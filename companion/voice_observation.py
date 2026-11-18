"""Strict voice diagnostics and conservative deltas, never physical stop proof."""

import re

BASE_FIELDS = frozenset(('frames', 'stops', 'alarms', 'last_ms', 'stop_ms', 'errors', 'dropped'))
V2_FIELDS = BASE_FIELDS | {'v', 'applied', 'apply_ms', 'now_ms'}
COUNTERS = ('frames', 'stops', 'alarms', 'errors', 'dropped', 'applied')
TERMINAL = {'stopped', 'interrupted', 'runtime_error', 'end_of_stream'}


def validate_stats(data):
    if not isinstance(data, dict) or set(data) not in (BASE_FIELDS, V2_FIELDS):
        raise ValueError('Incomplete voice diagnostic record')
    if any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in data.values()):
        raise ValueError('Voice diagnostic fields must be uint32')
    if 'v' in data and data['v'] != 2:
        raise ValueError('Unsupported voice diagnostic version')
    return data


def parse_voice_rx(line):
    text = line.decode('ascii')
    if not text.startswith('voice_rx ') or len(text) > 512:
        raise ValueError('Invalid voice diagnostic prefix/length')
    fields = {}
    for part in text[len('voice_rx '):].split(' '):
        if not re.fullmatch(r'[a-z_]+=[0-9]{1,10}', part):
            raise ValueError('Invalid voice diagnostic token')
        key, value = part.split('=')
        if key in fields:
            raise ValueError('Duplicate voice diagnostic key')
        fields[key] = int(value)
    return dict(validate_stats(fields))


class VoiceCounterTracker:
    """Only count deltas within a fresh, continuous telemetry interval.

    Uptime/counter continuity is evidence, not authenticated boot identity.
    First reports, reconnects, expiry and clock/counter regressions rebaseline.
    """

    def __init__(self):
        self.previous = None
        self.link_generation = None
        self.seen = False

    def observe(self, record):
        device = record.get('device') or {}
        item = device.get('voiceRx') or {}
        active = (record.get('observation') or {}).get('reason') not in TERMINAL
        try:
            data = dict(validate_stats(item.get('data')))
        except ValueError:
            data = {}
        if not (active and device.get('online') is True and item.get('fresh') is True and data.get('v') == 2):
            self.previous = None
            return None
        self.seen = True
        old, self.previous = self.previous, data
        generation = (device.get('transport'), device.get('reconnectAttempts', 0))
        old_generation, self.link_generation = self.link_generation, generation
        if old is None or generation != old_generation:
            return None
        elapsed = (data['now_ms'] - old['now_ms']) & 0xFFFFFFFF
        if elapsed == 0:
            # Repeated snapshots are not new telemetry; conflicts force a new baseline.
            self.previous = old if data == old else None
            return None
        limit = device.get('staleMs', 6000)
        if type(limit) is not int or not 0 < limit <= 300000:
            limit = 6000
        if elapsed > limit:
            return None
        deltas = {key: (data[key] - old[key]) & 0xFFFFFFFF for key in COUNTERS}
        if any(value >= 0x80000000 for value in deltas.values()):
            return None
        if deltas['stops'] + deltas['alarms'] > deltas['frames']:
            return None
        return dict(deltas, elapsedMs=elapsed, fromMs=old['now_ms'], toMs=data['now_ms'])
