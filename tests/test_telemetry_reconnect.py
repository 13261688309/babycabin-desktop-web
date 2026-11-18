"""Connection recovery tests use in-memory links, never physical ports."""

from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from companion.runtime import main, parse_args
from companion.telemetry import SerialTelemetry


def report(tick):
    return ('POST /api/device/report ' + json.dumps({'schemaVersion': 2, 'ts': tick,
             'temp': None, 'hum': None, 'sensorHealth': {'dht11': 'read_failed'}}) + '\n').encode()


class Clock:
    def __init__(self):
        self.now = 0.0
    def advance(self, seconds):
        self.now += seconds


class Stop:
    def __init__(self, clock):
        self.clock, self.stopped = clock, False
    def is_set(self):
        return self.stopped
    def set(self):
        self.stopped = True
    def wait(self, seconds):
        if not self.stopped:
            self.clock.advance(seconds)
        return self.stopped


class Link:
    def __init__(self, chunks=(), *, open_error=None, close_error=None, open_action=None):
        self.chunks = deque(chunks)
        self.open_error, self.close_error, self.open_action = open_error, close_error, open_action
        self.events, self.close_calls = [], 0
    def __setattr__(self, name, value):
        if name in ('dtr', 'rts', 'port'):
            self.events.append((name, value))
        object.__setattr__(self, name, value)
    def open(self):
        self.events.append(('open', None))
        if self.open_action:
            self.open_action()
        if self.open_error:
            raise self.open_error
    def read(self, count):
        if not self.chunks:
            raise OSError('injected disconnect')
        chunk = self.chunks.popleft()
        if callable(chunk):
            chunk = chunk()
        if isinstance(chunk, Exception):
            raise chunk
        assert len(chunk) <= count
        return chunk
    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


class TelemetryReconnectTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.worker = Mock()
        self.worker.is_alive.return_value = True
        self.snapshots = []

    @contextmanager
    def backend(self, links, seconds=1):
        queue = deque(links)
        def factory(**kwargs):
            if not queue:
                raise OSError('injected missing port')
            return queue.popleft()
        self.factory = Mock(side_effect=factory)
        with patch.dict(sys.modules, {'serial': SimpleNamespace(Serial=self.factory)}), \
             patch('companion.telemetry.threading.Thread', return_value=self.worker), \
             patch('companion.telemetry.time.monotonic', side_effect=lambda: self.clock.now), \
             patch('companion.telemetry.time.monotonic_ns', side_effect=lambda: round(self.clock.now * 1e9)):
            self.link = SerialTelemetry('EXPLICIT_TEST_PORT', reconnect_seconds=seconds)
            self.link._stop = Stop(self.clock)
            try:
                yield self.link
            finally:
                self.link.close()

    def sample(self, *, stop=False):
        self.snapshots.append(self.link.snapshot(round(self.clock.now * 1000)))
        if stop:
            self.link._stop.set()
        return b''

    def test_default_keeps_no_retry_behavior_and_old_data_is_not_fresh(self):
        first = Link([report(1), OSError()])
        with self.backend([first], seconds=0) as link:
            link._run()
            state = link.snapshot(0)
            self.assertEqual(self.factory.call_count, 1)
            self.assertFalse(state['online'])
            self.assertFalse(state['report']['fresh'])
            self.assertEqual(state['report']['data']['ts'], 1)

    def test_reopens_only_requested_port_with_inactive_control_lines(self):
        first = Link([OSError()])
        second = Link([report(2), lambda: self.sample(stop=True)])
        with self.backend([first, second]) as link:
            link._run()
        for item in (first, second):
            self.assertEqual(item.events[:4], [('dtr', False), ('rts', False),
                                               ('port', 'EXPLICIT_TEST_PORT'), ('open', None)])
            self.assertEqual(item.close_calls, 1)
            self.assertFalse(hasattr(item, 'write'))
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(self.snapshots[0]['report']['data']['ts'], 2)
        self.assertTrue(self.snapshots[0]['report']['fresh'])

    def test_old_report_is_absent_until_a_new_complete_record_arrives(self):
        first = Link([report(1), OSError()])
        second = Link([self.sample, report(2), lambda: self.sample(stop=True)])
        with self.backend([first, second]) as link:
            link._run()
        before, after = self.snapshots
        self.assertFalse(before['online'])
        self.assertEqual(before['connectionState'], 'awaiting_data')
        self.assertNotIn('report', before)
        self.assertTrue(after['online'])
        self.assertEqual(after['report']['data']['ts'], 2)

    def test_partial_lines_are_not_spliced_across_connections(self):
        original = report(77)
        first = Link([original[:35], OSError()])
        second = Link([original[35:], self.sample, report(88), lambda: self.sample(stop=True)])
        with self.backend([first, second]) as link:
            link._run()
        self.assertNotIn('report', self.snapshots[0])
        self.assertEqual(self.snapshots[1]['report']['data']['ts'], 88)

    def test_unavailable_port_attempts_are_bounded_by_one_window(self):
        with self.backend([Link([OSError()])], seconds=1) as link:
            link._run()
            state = link.snapshot(1000)
            self.assertEqual(state['connectionState'], 'reconnect_expired')
            self.assertFalse(state['online'])
            self.assertEqual(state['reconnectAttempts'], 4)
            self.assertEqual(self.clock.now, 1)
            self.assertIsNone(link._serial)

    def test_open_read_failure_loops_do_not_restart_the_outage_deadline(self):
        links = [Link([OSError()]) for _ in range(20)]
        with self.backend(links, seconds=1) as link:
            link._run()
            self.assertEqual(link.snapshot(1000)['connectionState'], 'reconnect_expired')
            self.assertEqual(self.factory.call_count, 5)
            self.assertEqual(self.clock.now, 1)

    def test_empty_open_connection_expires_without_new_valid_protocol_data(self):
        def empty():
            self.clock.advance(.2)
            return b''
        second = Link([empty] * 10)
        with self.backend([Link([OSError()]), second], seconds=.5) as link:
            link._run()
            self.assertEqual(link.snapshot(600)['connectionState'], 'reconnect_expired')
            self.assertFalse(link.snapshot(600)['online'])
        self.assertEqual(second.close_calls, 1)

    def test_valid_bytes_returned_after_deadline_are_not_published(self):
        def late():
            self.clock.advance(2)
            return report(9)
        with self.backend([Link([OSError()]), Link([late])], seconds=1) as link:
            link._run()
            self.assertNotIn('report', link.snapshot(2000))
            self.assertEqual(link.snapshot(2000)['connectionState'], 'reconnect_expired')

    def test_slow_open_past_deadline_is_closed_and_never_accepted(self):
        second = Link(open_action=lambda: self.clock.advance(2))
        with self.backend([Link([OSError()]), second], seconds=1) as link:
            link._run()
            self.assertIsNone(link._serial)
            self.assertFalse(link.snapshot(2000)['online'])
        self.assertEqual(second.close_calls, 1)

    def test_cancel_during_open_closes_new_handle_without_publishing_it(self):
        second = Link(open_action=lambda: self.link._stop.set())
        with self.backend([Link([OSError()]), second]) as link:
            link._run()
            self.assertFalse(link.snapshot(0)['online'])
            self.assertIsNone(link._serial)
        self.assertEqual(second.close_calls, 1)

    def test_recovered_report_can_age_out_normally_without_new_data(self):
        def later():
            self.clock.advance(7)
            return self.sample(stop=True)
        with self.backend([Link([OSError()]), Link([report(2), later])]) as link:
            link._run()
        self.assertFalse(self.snapshots[0]['report']['fresh'])
        self.assertEqual(self.snapshots[0]['report']['ageMs'], 7000)

    def test_successful_new_record_gives_later_outage_its_own_window(self):
        def advance():
            self.clock.advance(2)
            return OSError()
        first = Link([OSError()])
        second = Link([report(2), self.sample, advance])
        third = Link([report(3), lambda: self.sample(stop=True)])
        with self.backend([first, second, third], seconds=1) as link:
            link._run()
        self.assertEqual([s['report']['data']['ts'] for s in self.snapshots], [2, 3])
        self.assertTrue(all(s['online'] for s in self.snapshots))

    def test_close_failure_stops_recovery_instead_of_opening_another_handle(self):
        first = Link([OSError()], close_error=OSError())
        with self.backend([first, Link()]) as link:
            link._run()
            self.assertEqual(link.snapshot(0)['connectionState'], 'cleanup_failed')
            self.assertEqual(self.factory.call_count, 1)

    def test_failed_partial_open_is_closed_before_another_attempt(self):
        failed = Link(open_error=OSError())
        second = Link([report(2), lambda: self.sample(stop=True)])
        with self.backend([Link([OSError()]), failed, second]) as link:
            link._run()
        self.assertEqual(failed.close_calls, 1)
        self.assertEqual(self.snapshots[-1]['report']['data']['ts'], 2)

    def test_failed_partial_open_cleanup_failure_stops_further_attempts(self):
        failed = Link(open_error=OSError(), close_error=OSError())
        with self.backend([Link([OSError()]), failed, Link()]) as link:
            link._run()
            self.assertEqual(link.snapshot(0)['connectionState'], 'cleanup_failed')
            self.assertEqual(self.factory.call_count, 2)

    def test_initial_open_failure_remains_explicit_not_background_retry(self):
        first = Link(open_error=OSError())
        with self.assertRaises(OSError):
            with self.backend([first]):
                pass
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(self.factory.call_count, 1)

    def test_invalid_windows_fail_before_serial_allocation(self):
        factory = Mock()
        with patch.dict(sys.modules, {'serial': SimpleNamespace(Serial=factory)}):
            for value in (-1, 301, float('inf'), float('nan'), True, '1'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    SerialTelemetry('TEST_PORT', reconnect_seconds=value)
        factory.assert_not_called()

    def test_cli_requires_explicit_port_and_preserves_disabled_default(self):
        self.assertEqual(parse_args(['--source', '0']).serial_reconnect_seconds, 0)
        self.assertEqual(parse_args(['--source', '0', '--serial-port', 'TEST_PORT',
                                     '--serial-reconnect-seconds', '60']).serial_reconnect_seconds, 60)
        with redirect_stderr(io.StringIO()):
            for value in ('-1', '301', 'nan', 'inf'):
                with self.assertRaises(SystemExit):
                    parse_args(['--source', '0', '--serial-port', 'TEST_PORT', '--serial-reconnect-seconds', value])
            with self.assertRaises(SystemExit):
                parse_args(['--source', '0', '--serial-reconnect-seconds', '1'])

    def test_real_worker_wait_can_be_cancelled_without_waiting_for_full_window(self):
        first = Link([OSError()])
        retry_attempted = threading.Event()
        calls = 0
        def factory(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return first
            retry_attempted.set()
            raise OSError('injected absent port')
        with patch.dict(sys.modules, {'serial': SimpleNamespace(Serial=factory)}):
            link = SerialTelemetry('TEST_PORT', reconnect_seconds=60)
            self.assertTrue(retry_attempted.wait(1), 'worker never reached the retry phase')
            started = time.monotonic()
            link.close()
            self.assertLess(time.monotonic() - started, 1.2)
            self.assertFalse(link._thread.is_alive())
            self.assertFalse(link.snapshot(0)['online'])
        self.assertEqual(first.close_calls, 1)

    def test_runtime_passes_opt_in_window_and_closes_the_observer(self):
        model = SimpleNamespace(task='detect', names={0: 'Cry', 1: 'Happy', 2: 'Neutral', 3: 'Sleep'},
                                predict=lambda *a, **kw: [SimpleNamespace(boxes=None)])
        observer = Mock()
        observer.snapshot.return_value = {'online': False, 'staleMs': 6000}
        modules = {'ultralytics': SimpleNamespace(YOLO=lambda *a: model),
                   'cv2': SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda *a: object()),
                   'numpy': SimpleNamespace(uint8=object(), fromfile=lambda *a, **kw: object())}
        with tempfile.TemporaryDirectory() as folder:
            weights, image = Path(folder) / 'weights.pt', Path(folder) / 'image.jpg'
            weights.write_bytes(b'synthetic test weights, never loaded')
            image.write_bytes(b'synthetic test image, never decoded')
            with patch.dict(sys.modules, modules), \
                 patch('companion.telemetry.SerialTelemetry', return_value=observer) as factory, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(['--source', str(image), '--weights', str(weights), '--serial-port', 'TEST_PORT',
                               '--serial-reconnect-seconds', '60'])
            self.assertEqual(status, 0)
            factory.assert_called_once_with('TEST_PORT', 115200, reconnect_seconds=60)
            observer.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
