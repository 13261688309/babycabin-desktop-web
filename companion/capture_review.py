"""Check a saved serial_probe bundle and diagnose report sequencing offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .telemetry import TelemetryDecoder, finite_values, reject_constant, unique_object

LIMITS = {'summary.json': 65536, 'received.bin': 16 * 1024 * 1024,
          'decoded.jsonl': 64 * 1024 * 1024}
FALSE_FLAGS = ('deviceIdentityVerified', 'baudTimingOrFreshnessQualified',
               'informationChainComplete', 'hardwareValidated',
               'actuatorControlEnabled', 'powerOnApproved')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_json(text: bytes | str) -> object:
    data = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    require(finite_values(data), 'Nonfinite JSON value')
    return data


def canonical(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_record(path: Path, limit: int) -> dict:
    size = path.stat().st_size
    require(size <= limit, 'Capture file exceeds review size limit')
    result = hashlib.sha256()
    total = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            total += len(chunk)
            require(total <= limit, 'Capture file grew beyond review size limit')
            result.update(chunk)
    require(total == size, 'Capture file changed during review')
    return {'bytes': size, 'sha256': result.hexdigest()}


def review_capture(folder: Path, *, max_report_gap_ms: int = 3000) -> dict:
    require(type(max_report_gap_ms) is int and 1 <= max_report_gap_ms < 0x80000000,
            'Gap comparison must be an integer in 1..2147483647 ms')
    folder = Path(folder)
    inputs = {name: file_record(folder / name, limit) for name, limit in LIMITS.items()}
    with (folder / 'summary.json').open('rb') as handle:
        summary_bytes = handle.read(LIMITS['summary.json'] + 1)
    require(len(summary_bytes) <= LIMITS['summary.json'], 'Capture summary grew beyond size limit')
    summary = strict_json(summary_bytes)
    require(isinstance(summary, dict) and type(summary.get('schemaVersion')) is int
            and summary['schemaVersion'] == 1, 'Unsupported capture summary')
    mode = summary.get('mode')
    require(mode in ('offline_replay', 'live_receive_only'), 'Unknown capture mode')
    require(canonical(summary.get('files')) == canonical({n: inputs[n] for n in ('received.bin', 'decoded.jsonl')}),
            'Summary does not bind the saved raw and decoded files')
    for flag in FALSE_FLAGS:
        require(summary.get(flag) is False, 'Capture summary claims unverified hardware qualification')
    elapsed = summary.get('elapsedSeconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            'Invalid capture elapsed time')
    require(type(summary.get('maxBytes')) is int and 1024 <= summary['maxBytes'] <= LIMITS['received.bin']
            and inputs['received.bin']['bytes'] <= summary['maxBytes'], 'Invalid byte limit')
    require(type(summary.get('requestedSeconds')) in (int, float)
            and math.isfinite(summary['requestedSeconds']) and .1 <= summary['requestedSeconds'] <= 300,
            'Invalid requested duration')
    termination = summary.get('termination')
    require(termination in ('duration_limit', 'byte_limit', 'end_of_file', 'interrupted', 'io_error'),
            'Unknown capture termination')
    require(termination != 'end_of_file' or mode == 'offline_replay', 'Live capture cannot report EOF')
    error = summary.get('errorType')
    require((isinstance(error, str) and bool(error)) if termination == 'io_error' else error is None,
            'Error and termination disagree')
    normal = termination == ('end_of_file' if mode == 'offline_replay' else 'duration_limit')
    if termination == 'byte_limit':
        require(inputs['received.bin']['bytes'] == summary['maxBytes'], 'Byte limit termination mismatch')
    if termination == 'duration_limit':
        # elapsedSeconds and receivedOffsetMs were independently rounded by the producer.
        require(elapsed + .000001 >= summary['requestedSeconds'], 'Duration limit termination mismatch')
    decoder = TelemetryDecoder()
    reports = alerts = decoded_count = 0
    first_tick = previous_tick = None
    previous_offset = first_report_offset = last_report_offset = None
    duplicates = discontinuities = wraps = firmware_gaps = host_gaps = 0
    max_firmware_gap = max_host_gap = None
    trailing = 0
    examples: list[dict] = []

    def event(kind: str, tick: int, previous: int | None, delta: int | None) -> None:
        if len(examples) < 32:
            examples.append({'kind': kind, 'reportIndex': reports - 1, 'previousTick': previous,
                             'tick': tick, 'deltaMs': delta})

    with (folder / 'received.bin').open('rb') as raw, (folder / 'decoded.jsonl').open('rb') as decoded:
        while chunk := raw.read(512):
            trailing = len(chunk.rsplit(b'\n', 1)[-1]) if b'\n' in chunk else trailing + len(chunk)
            for record in decoder.feed(chunk):
                line = decoded.readline(65537)
                require(bool(line) and len(line) <= 65536 and line.endswith(b'\n'),
                        'Missing, oversized or unterminated decoded row')
                row = strict_json(line)
                require(isinstance(row, dict) and set(row) == {'receivedOffsetMs', 'type', 'data'},
                        'Decoded row fields differ from serial_probe schema')
                require(canonical({k: row[k] for k in ('type', 'data')}) == canonical(record),
                        'Decoded row does not match raw bytes')
                offset = row['receivedOffsetMs']
                if mode == 'offline_replay':
                    require(offset is None, 'Offline replay cannot have reception timing')
                else:
                    require(type(offset) is int and 0 <= offset <= elapsed * 1000 + .501,
                            'Invalid saved receive offset')
                    require(previous_offset is None or offset >= previous_offset,
                            'Receive offsets moved backwards')
                    previous_offset = offset
                decoded_count += 1
                if record['type'] == 'alert':
                    alerts += 1
                    continue
                reports += 1
                tick = record['data']['ts']
                if reports == 1:
                    first_report_offset = offset
                if first_tick is None:
                    first_tick = tick
                if previous_tick is not None:
                    delta = (tick - previous_tick) & 0xFFFFFFFF
                    if delta == 0:
                        duplicates += 1
                        event('duplicate_firmware_tick', tick, previous_tick, 0)
                    elif delta >= 0x80000000:
                        discontinuities += 1
                        event('ambiguous_tick_discontinuity', tick, previous_tick, None)
                    else:
                        max_firmware_gap = max(delta, max_firmware_gap or 0)
                        if tick < previous_tick:
                            wraps += 1
                            event('wrap_compatible_forward_step', tick, previous_tick, delta)
                        if delta > max_report_gap_ms:
                            firmware_gaps += 1
                            event('firmware_tick_gap', tick, previous_tick, delta)
                if offset is not None and last_report_offset is not None:
                    host_delta = offset - last_report_offset
                    max_host_gap = max(host_delta, max_host_gap or 0)
                    if host_delta > max_report_gap_ms:
                        host_gaps += 1
                        event('saved_reception_gap', tick, previous_tick, host_delta)
                previous_tick = tick
                last_report_offset = offset
        require(not decoded.read(1), 'Extra decoded rows without matching raw data')
    expected = {'bytesReceived': inputs['received.bin']['bytes'], 'reportCount': reports,
                'alertCount': alerts, 'rejectedProtocolLines': decoder.rejected,
                'firstFirmwareTick': first_tick, 'lastFirmwareTick': previous_tick}
    for key, value in expected.items():
        require(canonical(summary.get(key)) == canonical(value), 'Capture summary count/tick mismatch: ' + key)
    require(summary.get('liveSerialReportObserved') is (mode == 'live_receive_only' and reports > 0),
            'Live observation flag disagrees with recorded mode/count')
    require(summary.get('receiveOrReplayCheckPassed') is (normal and reports > 0),
            'Producer pass flag disagrees with termination/count')
    # A second digest pass catches ordinary edits during analysis, not malicious forgery.
    require(inputs == {name: file_record(folder / name, limit) for name, limit in LIMITS.items()},
            'Saved capture changed during review')
    leading_silence = first_report_offset
    trailing_silence = None if last_report_offset is None else max(0., elapsed * 1000 - last_report_offset)
    boundary_gaps = sum(value is not None and value > max_report_gap_ms for value in (leading_silence, trailing_silence))
    anomalies = duplicates + discontinuities + firmware_gaps + host_gaps + boundary_gaps + decoder.rejected
    status = 'insufficient_reports' if reports < 2 else 'attention_required' if anomalies or trailing or not normal else 'no_sequence_anomalies_found'
    return {
        'schemaVersion': 1, 'type': 'offline_capture_review', 'captureMode': mode,
        'inputFiles': inputs, 'bundleInternallyConsistent': True,
        'inputAuthenticityOrDeviceIdentityVerified': False,
        'captureTerminatedNormally': normal, 'captureTermination': termination,
        'reportCount': reports, 'alertCount': alerts, 'decodedCount': decoded_count,
        'rejectedProtocolLines': decoder.rejected, 'unterminatedTrailingBytes': trailing,
        'sequenceDiagnostic': status, 'maxReportGapComparisonMs': max_report_gap_ms,
        'duplicateFirmwareTicks': duplicates, 'ambiguousTickDiscontinuities': discontinuities,
        'wrapCompatibleForwardSteps': wraps, 'firmwareTickGaps': firmware_gaps,
        'savedReceptionGaps': host_gaps, 'maxForwardFirmwareTickDeltaMs': max_firmware_gap,
        'savedReceptionBoundaryGaps': boundary_gaps,
        'savedLeadingReportSilenceMs': leading_silence, 'savedTrailingReportSilenceMs': trailing_silence,
        'maxSavedReceptionDeltaMs': max_host_gap, 'examples': examples,
        'exampleLimit': 32, 'hardwareAccessPerformed': False,
        'baudTimingOrFreshnessQualified': False, 'informationChainComplete': False,
        'hardwareValidated': False, 'actuatorControlEnabled': False, 'powerOnApproved': False,
        'limitations': [
            'Hashes and matching decode prove internal consistency, not authenticity or device identity.',
            'A modular tick jump can mean reset, loss, replay or reordering; cause is not established.',
            'Receive offsets are chunk-completion times, not per-byte UART or sample timestamps.',
            'Gap threshold is a diagnostic comparison, not guaranteed firmware cadence.',
            'No electrical, sensor, end-to-end freshness or actuator qualification is inferred.',
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='saved serial_probe output directory')
    parser.add_argument('--output', type=Path, required=True, help='new JSON file outside the input bundle')
    parser.add_argument('--max-report-gap-ms', type=int, default=3000)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), 'Output already exists')
        require(args.capture.resolve() not in args.output.resolve().parents,
                'Keep analysis output outside the input capture bundle')
        result = review_capture(args.capture, max_report_gap_ms=args.max_report_gap_ms)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
    except (ValueError, OSError, RecursionError):
        print('Capture review failed; check saved bundle integrity, options and new output path.')
        return 2
    print(json.dumps(result, allow_nan=False, separators=(',', ':')))
    return 0 if result['sequenceDiagnostic'] == 'no_sequence_anomalies_found' else 1


if __name__ == '__main__':
    raise SystemExit(main())
