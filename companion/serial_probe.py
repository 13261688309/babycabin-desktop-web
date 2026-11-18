"""Capture bounded receive-only UART evidence, or decode a saved capture offline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

from .telemetry import TelemetryDecoder


@contextmanager
def serial_reader(port: str, baud: int):
    import serial

    link = serial.Serial(port=None, baudrate=baud, timeout=.2, write_timeout=.2)
    try:
        link.dtr = False
        link.rts = False
        link.port = port
        link.open()
        yield link
    finally:
        link.close()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(65536), b''):
            result.update(chunk)
    return result.hexdigest()


def probe(output: Path, *, port: str | None = None, replay: Path | None = None,
          seconds: float = 15., baud: int = 115200, max_bytes: int = 1024 * 1024,
          open_serial=None, clock=None) -> dict:
    if bool(port) == (replay is not None):
        raise ValueError('Choose exactly one live port or replay file')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not .1 <= seconds <= 300:
        raise ValueError('Duration must be finite and in 0.1..300 seconds')
    if type(baud) is not int or not 1 <= baud <= 4000000:
        raise ValueError('Baud must be an integer in 1..4000000')
    if type(max_bytes) is not int or not 1024 <= max_bytes <= 16 * 1024 * 1024:
        raise ValueError('Byte limit must be an integer in 1024..16777216')
    if port is not None and (not isinstance(port, str) or not port.strip()):
        raise ValueError('Nonempty port required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    clock = clock or time.monotonic
    open_serial = open_serial or serial_reader
    decoder = TelemetryDecoder()
    raw_path, decoded_path = output / 'received.bin', output / 'decoded.jsonl'
    received = reports = alerts = 0
    first_report = last_report = None
    termination, error_type = 'not_started', None
    started = clock()
    try:
        with raw_path.open('xb') as raw, decoded_path.open('x', encoding='utf-8') as decoded:
            resource = open_serial(port, baud) if replay is None else Path(replay).open('rb')
            with resource as source:
                while True:
                    if clock() - started >= seconds:
                        termination = 'duration_limit'
                        break
                    if received >= max_bytes:
                        termination = 'byte_limit'
                        break
                    chunk = source.read(min(512, max_bytes - received))
                    if not isinstance(chunk, bytes):
                        raise TypeError('Reader must return bytes')
                    if len(chunk) > min(512, max_bytes - received):
                        raise ValueError('Reader exceeded requested byte count')
                    if not chunk:
                        if replay is not None:
                            termination = 'end_of_file'
                            break
                        continue
                    raw.write(chunk)
                    received += len(chunk)
                    offset = round((clock() - started) * 1000) if replay is None else None
                    for record in decoder.feed(chunk):
                        decoded.write(json.dumps({'receivedOffsetMs': offset, **record},
                                                 allow_nan=False, separators=(',', ':')) + '\n')
                        if record['type'] == 'report':
                            reports += 1
                            first_report = record['data']['ts'] if first_report is None else first_report
                            last_report = record['data']['ts']
                        else:
                            alerts += 1
    except KeyboardInterrupt:
        termination = 'interrupted'
    except Exception as error:
        # Serial/backend errors can contain identifiers or private configuration.
        termination, error_type = 'io_error', type(error).__name__
    elapsed = max(0., clock() - started)
    successful = reports > 0 and termination == ('duration_limit' if replay is None else 'end_of_file')
    result = {
        'schemaVersion': 1,
        'mode': 'live_receive_only' if replay is None else 'offline_replay',
        'termination': termination,
        'errorType': error_type,
        'elapsedSeconds': round(elapsed, 6),
        'requestedSeconds': seconds,
        'maxBytes': max_bytes,
        'bytesReceived': received,
        'reportCount': reports,
        'alertCount': alerts,
        'rejectedProtocolLines': decoder.rejected,
        'firstFirmwareTick': first_report,
        'lastFirmwareTick': last_report,
        'receiveOrReplayCheckPassed': successful,
        'liveSerialReportObserved': replay is None and reports > 0,
        'deviceIdentityVerified': False,
        'baudTimingOrFreshnessQualified': False,
        'informationChainComplete': False,
        'hardwareValidated': False,
        'actuatorControlEnabled': False,
        'powerOnApproved': False,
        'files': {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                  for path in (raw_path, decoded_path) if path.is_file()},
    }
    with (output / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--port', help='actual UART adapter port; TX/power must remain disconnected')
    source.add_argument('--replay', type=Path, help='offline binary capture; does not open a serial port')
    parser.add_argument('--output', type=Path, required=True, help='new directory; existing output is never replaced')
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--max-bytes', type=int, default=1024 * 1024)
    args = parser.parse_args(argv)
    try:
        result = probe(args.output, port=args.port, replay=args.replay, seconds=args.seconds,
                       baud=args.baud, max_bytes=args.max_bytes)
    except (ValueError, OSError):
        print('Probe could not reserve a new output or validate its options.')
        return 2
    print(json.dumps(result, allow_nan=False, separators=(',', ':')))
    if result['termination'] == 'interrupted':
        return 130
    return 0 if result['receiveOrReplayCheckPassed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
