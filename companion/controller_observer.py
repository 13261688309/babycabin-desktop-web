"""Explicit, identity-gated native USB reader. No command or write method exists."""
import copy
import re

from .gimbal import parse_control
from .native_usb import NativeStatusDecoder, native_wiring_scope

BENCH_FIELDS = frozenset(('v', 'epoch', 'lease', 'seq', 'now_ms', 'accepted', 'rejected',
                         'pan', 'tilt', 'remaining', 'pwm', 'fault', 'rock', 'motor', 'external'))


def parse_bench(line):
    if not isinstance(line, bytes) or not line.startswith(b'bench_link ') or len(line) > 256:
        raise ValueError('Not a bench capability')
    result = {}
    for token in line.decode('ascii').strip().split(' ')[1:]:
        key, sep, value = token.partition('=')
        if not sep or key in result or not re.fullmatch(r'[0-9]{1,10}', value) or int(value) > 0xffffffff:
            raise ValueError('Invalid bench status')
        result[key] = int(value)
    if (set(result) != BENCH_FIELDS or result['v'] != 1 or result['pwm'] not in (0, 1)
            or result['fault'] not in (0, 1) or result['external'] != 0
            or result['rock'] != 0 or result['motor'] != 0
            or not 60 <= result['pan'] <= 120 or not 75 <= result['tilt'] <= 105
            or result['remaining'] > 10000):
        raise ValueError('Unsupported bench status')
    return result


class ControllerDecoder(NativeStatusDecoder):
    def _decode(self, line):
        if line.startswith((b'bench_link ', b'gimbal_link ')):
            try:
                bench = line.startswith(b'bench_link ')
                return dict(type='controller', profile='GB01' if bench else 'GC03',
                            data=parse_bench(line) if bench else parse_control(line))
            except (ValueError, UnicodeError):
                self.rejected += 1
                return None
        return super()._decode(line)


class ControllerObserver:
    STALE_MS = 6000

    def __init__(self, port, serial_number, location, **confirmations):
        self.scope = native_wiring_scope(**confirmations)
        if any(type(value) is not str or not value.strip() for value in (port, serial_number, location)):
            raise ValueError('Explicit native endpoint identity required')
        from tools.capture_mcu_usb import native_reader
        self.decoder = ControllerDecoder()
        self.latest = {}
        self.connected = False
        self.error = None
        self._context = native_reader(port, 115200, expected_serial=serial_number, expected_location=location)
        self._link = self._context.__enter__()
        try:
            self._link.timeout = 0
            self.connected = True
        except BaseException:
            self._context.__exit__(None, None, None)
            raise

    def poll(self, now):
        if not self.connected:
            return
        try:
            for record in self.decoder.feed(self._link.read(4096)):
                if record['type'] == 'controller':
                    previous = self.latest.get('controller')
                    if previous and (record['profile'] != previous['profile']
                                     or record['data']['now_ms'] < previous['data']['now_ms']):
                        self.latest.clear()
                self.latest[record['type']] = dict(record, receivedMs=now)
        except (OSError, ValueError):
            self.error = 'Native USB read failed; no automatic reopen'
            self.close()

    def snapshot(self, now):
        result = dict(connected=self.connected, staleMs=self.STALE_MS, readOnly=True,
                      transport='h743_native_usb_cdc_read_only', rejectedLines=self.decoder.rejected,
                      physicalStateMeasured=False, applicationBytesWritten=0,
                      wiringScope=self.scope, error=self.error)
        bench = self.latest.get('controller', {}).get('profile') == 'GB01'
        for name, record in self.latest.items():
            # GB01 explicitly disables these modules. Never reuse old full-system sensor lines.
            if bench and name != 'controller':
                continue
            age = now - record['receivedMs']
            result[name] = dict(copy.deepcopy(record), ageMs=max(0, age),
                                fresh=self.connected and 0 <= age < self.STALE_MS)
        return result

    def close(self):
        self.connected = False
        self.latest.clear()
        context, self._context = self._context, None
        if context is not None:
            context.__exit__(None, None, None)
