"""Build a local, read-only historical viewer without hardware or model access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .capture_review import file_record, require, review_capture, strict_json
from .state import Detection, LABELS

MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 50000
MAX_LINE = 65536
SAFE_INT = (1 << 53) - 1
TEMPLATE = Path(__file__).with_name('replay_view.html')


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= SAFE_INT


def browser_safe_numbers(value):
    if type(value) in (int, float):
        return -SAFE_INT <= value <= SAFE_INT
    if isinstance(value, dict):
        return all(browser_safe_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(browser_safe_numbers(item) for item in value)
    return True


def bounded_rows(path: Path):
    total = 0
    with path.open('rb') as handle:
        for index in range(MAX_ROWS + 1):
            line = handle.readline(MAX_LINE + 1)
            if not line:
                return
            total += len(line)
            require(index < MAX_ROWS and len(line) <= MAX_LINE and line.endswith(b'\n'),
                    'Too many, oversized or unterminated rows')
            require(total <= MAX_BYTES, 'Input grew beyond export size limit')
            row = strict_json(line)
            require(isinstance(row, dict), 'Expected a JSON object per row')
            require(browser_safe_numbers(row), 'Recorded number exceeds browser precision range')
            yield row


def vision_session(path: Path) -> dict:
    identity = file_record(path, MAX_BYTES)
    events = []
    first = previous_time = previous_sequence = None
    for row in bounded_rows(path):
        require(row.get('type') == 'vision_observation' and type(row.get('schemaVersion')) is int
                and row['schemaVersion'] == 1, 'Unsupported vision record')
        require(row.get('actuatorControlEnabled') is False, 'Only observation-only logs are accepted')
        stamp, sequence = row.get('timestampMs'), row.get('sequence')
        require(integer(stamp) and integer(sequence), 'Invalid vision timestamp or sequence')
        require(previous_time is None or stamp >= previous_time, 'Vision timestamps moved backwards')
        require(previous_sequence is None or sequence == previous_sequence + 1, 'Vision sequence is discontinuous')
        observation = row.get('observation')
        require(isinstance(observation, dict), 'Missing recorded observation')
        require(type(observation.get('state')) is str and observation['state'] in LABELS | {'Unknown'}
                and (observation.get('candidate') is None or (type(observation['candidate']) is str
                     and observation['candidate'] in LABELS)), 'Unknown recorded class')
        confidence = observation.get('confidence')
        require(confidence is None or (type(confidence) in (int, float) and 0 <= confidence <= 1),
                'Invalid confidence')
        require(integer(observation.get('confirmed_frames')) and type(observation.get('reason')) is str
                and 0 < len(observation['reason']) <= 128, 'Invalid recorded observation fields')
        require((observation['state'] == 'Unknown' and observation['reason'] != 'confirmed') or
                (observation['state'] in LABELS and observation['reason'] == 'confirmed'
                 and observation['candidate'] == observation['state'] and confidence is not None
                 and observation['confirmed_frames'] >= 2), 'Contradictory recorded confirmation')
        detections = row.get('detections')
        require(isinstance(detections, list) and len(detections) <= 1000, 'Invalid detection list')
        for item in detections:
            require(isinstance(item, dict) and type(item.get('label')) is str
                    and type(item.get('confidence')) in (int, float)
                    and isinstance(item.get('xyxy'), list) and len(item['xyxy']) == 4
                    and all(type(v) in (int, float) for v in item['xyxy']), 'Invalid detection fields')
            require(Detection(item.get('label'), item['confidence'], tuple(item['xyxy'])).valid(),
                    'Invalid recorded detection')
        captured, frame_age, inference = row.get('frameCapturedMs'), row.get('frameAgeMs'), row.get('inferenceMs')
        require(captured is None or (integer(captured) and captured <= stamp), 'Invalid captured timestamp')
        require(frame_age is None or integer(frame_age), 'Invalid recorded frame age')
        require((captured is None and frame_age is None) or
                (captured is not None and frame_age == stamp - captured), 'Frame age disagrees with captured timestamp')
        require(inference is None or (type(inference) in (int, float) and inference >= 0), 'Invalid inference time')
        utc, kind = row.get('utc'), row.get('sourceKind')
        require(type(utc) is str and len(utc) <= 64 and kind in {'network', 'camera', 'file'},
                'Invalid source metadata')
        # Export only observation fields; arbitrary extra input fields are not copied into HTML.
        snapshot = {'sequence': sequence, 'timestampMs': stamp, 'utc': utc, 'sourceKind': kind,
                    'observation': {key: observation[key] for key in
                                    ('state', 'reason', 'candidate', 'confidence', 'confirmed_frames')},
                    'detectionCount': len(detections), 'frameAgeMs': frame_age,
                    'inferenceMs': inference, 'actuatorControlEnabled': False}
        model = row.get('model')
        require(isinstance(model, dict) and 'sha256' in model
                and type(model.get('name')) is str and len(model['name']) <= 256
                and (model.get('sha256') is None or (type(model['sha256']) is str
                     and re.fullmatch('[0-9a-f]{64}', model['sha256']))), 'Invalid recorded model identity')
        snapshot['model'] = {key: model[key] for key in ('name', 'sha256')}
        first = stamp if first is None else first
        events.append({'position': stamp - first, 'record': snapshot})
        previous_time, previous_sequence = stamp, sequence
    require(events, 'Vision log contains no records')
    require(identity == file_record(path, MAX_BYTES), 'Vision log changed during export')
    return {'kind': 'vision', 'name': path.name, 'timeAxis': 'host_monotonic_relative_ms',
            'duration': events[-1]['position'], 'files': {path.name: identity}, 'events': events}


def telemetry_session(folder: Path) -> dict:
    review = review_capture(folder)
    with (folder / 'summary.json').open('rb') as handle:
        summary_bytes = handle.read(65537)
    require(len(summary_bytes) <= 65536, 'Capture summary grew beyond export size limit')
    summary = strict_json(summary_bytes)
    timed = review['captureMode'] == 'live_receive_only'
    events = []
    for index, row in enumerate(bounded_rows(folder / 'decoded.jsonl')):
        fields = ('schemaVersion', 'ts', 'status', 'temp', 'hum', 'sensorHealth', 'alert',
                  'severity', 'suggestion', 'esp32Online', 'motion', 'wet', 'battery')
        events.append({'position': row['receivedOffsetMs'] if timed else index,
                       'record': {'type': row['type'], 'data': {key: row['data'][key] for key in fields
                                                             if key in row['data']}}})
    require(events, 'Telemetry capture contains no decoded records')
    # Recheck the bound raw bytes as well as their decoded rows after consuming them.
    require(review == review_capture(folder), 'Telemetry capture changed during export')
    duration = max(events[-1]['position'], round(summary['elapsedSeconds'] * 1000)) if timed else len(events) - 1
    require(integer(duration), 'Session duration exceeds browser precision range')
    return {'kind': 'telemetry', 'name': folder.name,
            'timeAxis': 'saved_receive_offsets_ms' if timed else 'ordinal_only', 'duration': duration,
            'files': review['inputFiles'], 'diagnostic': review['sequenceDiagnostic'],
            'events': events}


def build_replay(*, vision_log: Path | None = None, telemetry_capture: Path | None = None) -> dict:
    require(vision_log is not None or telemetry_capture is not None, 'At least one saved source is required')
    sessions = []
    if vision_log is not None:
        sessions.append(vision_session(Path(vision_log)))
    if telemetry_capture is not None:
        sessions.append(telemetry_session(Path(telemetry_capture)))
    return {'schemaVersion': 1, 'type': 'offline_history_replay', 'sessions': sessions,
            'timeAxesSynchronized': False, 'inputAuthenticityOrDeviceIdentityVerified': False,
            'hardwareAccessPerformed': False, 'modelInferencePerformed': False,
            'actuatorControlEnabled': False, 'informationChainComplete': False,
            'hardwareValidated': False, 'powerOnApproved': False}


def render_html(report: dict) -> str:
    payload = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    # JSON script contents must not contain an HTML closing-script token.
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = TEMPLATE.read_text(encoding='utf-8')
    require(template.count('__REPLAY_DATA__') == 1, 'Invalid viewer template')
    return template.replace('__REPLAY_DATA__', payload)


def export_replay(output: Path, *, vision_log: Path | None = None, telemetry_capture: Path | None = None) -> dict:
    output = Path(output)
    require(not output.exists(), 'Existing output is never overwritten')
    if telemetry_capture is not None:
        require(not output.resolve().is_relative_to(Path(telemetry_capture).resolve()),
                'Keep output outside the capture bundle')
    report = build_replay(vision_log=vision_log, telemetry_capture=telemetry_capture)
    html = render_html(report)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'index.html').write_text(html, encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + '\n',
                                      encoding='utf-8')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-log', type=Path)
    parser.add_argument('--telemetry-capture', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new directory for index.html and report.json')
    args = parser.parse_args(argv)
    try:
        result = export_replay(args.output, vision_log=args.vision_log, telemetry_capture=args.telemetry_capture)
    except (ValueError, OSError, RecursionError, TypeError):
        print('Replay export failed; check saved input integrity and the new output directory.')
        return 2
    print(json.dumps({'sessions': len(result['sessions']),
                      'records': [len(s['events']) for s in result['sessions']],
                      'hardwareAccessPerformed': False, 'timeAxesSynchronized': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
