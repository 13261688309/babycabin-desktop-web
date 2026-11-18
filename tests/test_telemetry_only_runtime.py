"""Headless device observation must not require vision or relax USB identity gates."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from companion.dashboard import DashboardState
from companion.runtime import main, parse_args

NATIVE = ['--telemetry-only', '--serial-port', 'TEST_PORT', '--serial-transport', 'native-usb',
          '--usb-serial-number', 'TEST123', '--usb-location', '1-5',
          '--native-board-confirmed', '--isolated-usb-confirmed']


class TelemetryOnlyRuntimeTests(unittest.TestCase):
    def test_source_requirements_and_native_scope_gates(self):
        self.assertTrue(parse_args(NATIVE).telemetry_only)
        voice = [x for x in NATIVE if x != '--isolated-usb-confirmed'] + [
            '--voice-rx-harness-confirmed', '--actuators-disconnected-confirmed',
            '--separate-usb-power-confirmed']
        self.assertTrue(parse_args(voice).voice_rx_harness_confirmed)
        cases = [[], ['--telemetry-only'], NATIVE + ['--source', '0'],
                 NATIVE + ['--voice-rx-harness-confirmed'],
                 ['--source', '0', '--voice-rx-harness-confirmed']]
        for flag in ('--voice-rx-harness-confirmed', '--actuators-disconnected-confirmed',
                     '--separate-usb-power-confirmed'):
            cases.append([x for x in voice if x != flag])
        with redirect_stderr(io.StringIO()):
            for args in cases:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    parse_args(args)

    def run_fake(self, extra=(), *, interrupt=False):
        observer = Mock()
        observer.snapshot.return_value = {'online': True, 'staleMs': 6000,
            'status': {'fresh': True, 'ageMs': 0, 'data': {'voice': 1, 'il_en': 0}}}
        output = io.StringIO()
        with patch('companion.native_usb.NativeUsbTelemetry', return_value=observer) as factory, \
             patch.dict(sys.modules, {'ultralytics': None, 'cv2': None, 'numpy': None}), \
             patch('companion.runtime.file_digest', side_effect=AssertionError('No model should be read')), \
             patch('companion.runtime.time.sleep', side_effect=KeyboardInterrupt if interrupt else None), \
             redirect_stdout(output), redirect_stderr(io.StringIO()):
            code = main(NATIVE + ['--max-frames', '2', '--weights', 'nonexistent.pt'] + list(extra))
        return code, [json.loads(line) for line in output.getvalue().splitlines()], observer, factory

    def test_no_model_or_camera_import_and_metadata_only_records(self):
        code, rows, observer, factory = self.run_fake()
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['type'] == 'device_observation' for row in rows))
        self.assertTrue(all(row['sourceKind'] == 'serial_only' for row in rows))
        self.assertTrue(all(row['model'] is None and not row['detections'] for row in rows))
        self.assertTrue(all(row['frameCapturedMs'] is None for row in rows))
        self.assertEqual(rows[0]['observation']['state'], 'Unknown')
        self.assertEqual(rows[0]['observation']['reason'], 'telemetry_only')
        self.assertEqual(rows[-1]['observation']['reason'], 'stopped')
        self.assertTrue(all(row['actuatorControlEnabled'] is False for row in rows))
        observer.close.assert_called_once()
        factory.assert_called_once_with('TEST_PORT', 115200, expected_serial='TEST123',
            expected_location='1-5', native_board_confirmed=True, isolated_usb_confirmed=True)

    def test_voice_scope_passes_all_flags_without_claiming_isolation(self):
        args = [x for x in NATIVE if x != '--isolated-usb-confirmed'] + [
            '--voice-rx-harness-confirmed', '--actuators-disconnected-confirmed',
            '--separate-usb-power-confirmed', '--max-frames', '1']
        observer = Mock()
        observer.snapshot.return_value = {}
        with patch('companion.native_usb.NativeUsbTelemetry', return_value=observer) as factory, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual(factory.call_args.kwargs['isolated_usb_confirmed'], False)
        for key in ('voice_rx_harness_confirmed', 'actuators_disconnected_confirmed',
                    'separate_usb_power_confirmed'):
            self.assertIs(factory.call_args.kwargs[key], True)

    def test_interrupt_closes_the_observer(self):
        code, rows, observer, _ = self.run_fake(interrupt=True)
        self.assertEqual(code, 130)
        self.assertEqual(rows[-1]['observation']['reason'], 'interrupted')
        observer.close.assert_called_once()

    def test_dht_only_scope_is_passed_without_isolation_or_voice_claims(self):
        args = [x for x in NATIVE if x != '--isolated-usb-confirmed'] + [
            '--dht11-3v3-harness-confirmed', '--actuators-disconnected-confirmed',
            '--max-frames', '1']
        observer = Mock()
        observer.snapshot.return_value = {}
        with patch('companion.native_usb.NativeUsbTelemetry', return_value=observer) as factory, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertIs(factory.call_args.kwargs['isolated_usb_confirmed'], False)
        self.assertIs(factory.call_args.kwargs['dht11_3v3_harness_confirmed'], True)
        self.assertIs(factory.call_args.kwargs['actuators_disconnected_confirmed'], True)
        self.assertNotIn('voice_rx_harness_confirmed', factory.call_args.kwargs)
        with redirect_stderr(io.StringIO()):
            for extra in ('--isolated-usb-confirmed', '--voice-rx-harness-confirmed',
                          '--separate-usb-power-confirmed'):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_args(args + [extra])

    def test_metadata_log_is_new_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'device.jsonl'
            code, rows, _, _ = self.run_fake(['--jsonl', str(path)])
            self.assertEqual(code, 0)
            original = path.read_bytes()
            self.assertEqual(len(original.splitlines()), len(rows))
            code, _, _, factory = self.run_fake(['--jsonl', str(path)])
            self.assertEqual(code, 2)
            factory.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_joint_voice_dht_scope_keeps_separate_power_and_actuator_gates(self):
        args = [x for x in NATIVE if x != '--isolated-usb-confirmed'] + [
            '--voice-and-dht11-harness-confirmed', '--actuators-disconnected-confirmed',
            '--separate-usb-power-confirmed', '--max-frames', '1']
        observer = Mock()
        observer.snapshot.return_value = {}
        with patch('companion.native_usb.NativeUsbTelemetry', return_value=observer) as factory, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertIs(factory.call_args.kwargs['isolated_usb_confirmed'], False)
        for key in ('voice_and_dht11_harness_confirmed', 'actuators_disconnected_confirmed',
                    'separate_usb_power_confirmed'):
            self.assertIs(factory.call_args.kwargs[key], True)
        self.assertNotIn('voice_rx_harness_confirmed', factory.call_args.kwargs)
        self.assertNotIn('dht11_3v3_harness_confirmed', factory.call_args.kwargs)
        with redirect_stderr(io.StringIO()):
            for extra in ('--isolated-usb-confirmed', '--voice-rx-harness-confirmed',
                          '--dht11-3v3-harness-confirmed'):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_args(args + [extra])

    def test_dashboard_status_ages_without_mutating_published_record(self):
        record = {'observation': {'state': 'Unknown', 'reason': 'telemetry_only'},
            'device': {'online': True, 'staleMs': 100,
                       'status': {'ageMs': 0, 'fresh': True, 'data': {'voice': 1, 'il_en': 0}}}}
        state = DashboardState()
        with patch('companion.dashboard.time.monotonic_ns', return_value=0):
            state.publish(record)
        with patch('companion.dashboard.time.monotonic_ns', return_value=101_000_000):
            result = state.snapshot()
        self.assertFalse(result['record']['device']['status']['fresh'])
        self.assertEqual(result['record']['device']['status']['ageMs'], 101)
        self.assertTrue(record['device']['status']['fresh'])


if __name__ == '__main__':
    unittest.main()
