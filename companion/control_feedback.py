"""Read-only feedback sharing the control port; never opens a device or writes bytes."""
import copy

from .native_usb import NativeStatusDecoder


class ControlFeedback:
    TOPICS = frozenset(('report', 'status', 'motion', 'voiceRx', 'alert'))
    STALE_MS = 6000

    def __init__(self):
        self.decoder = NativeStatusDecoder()
        self.latest = {}

    def feed(self, chunk, now):
        for record in self.decoder.feed(chunk):
            if record['type'] in self.TOPICS:
                self.latest[record['type']] = dict(record, receivedMs=now)

    def snapshot(self, now, *, connected):
        result = dict(connected=connected, staleMs=self.STALE_MS,
                      rejectedLines=self.decoder.rejected,
                      physicalStateMeasured=False)
        for name, record in self.latest.items():
            age = now - record['receivedMs']
            result[name] = dict(copy.deepcopy(record), ageMs=max(0, age),
                                fresh=connected and 0 <= age < self.STALE_MS)
        return result
