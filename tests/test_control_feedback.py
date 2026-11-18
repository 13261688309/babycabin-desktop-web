import json
import unittest

from companion.control_feedback import ControlFeedback
from companion.gimbal_console import ControlLink, GimbalSession
from test_gimbal import FakeLink, capability


REPORT = {'schemaVersion': 2, 'ts': 1000, 'temp': 25.5, 'hum': 60,
          'sensorHealth': {'dht11': 'ok'}}
STATUS = b'status mode=idle voice=0x01 il_stop=1 il_arm=1 il_ready=1 il_en=0 il_pg=0 il_pending=0 il_trip=0\r\n'
VOICE = b'voice_rx v=2 frames=8 stops=2 alarms=0 last_ms=900 stop_ms=900 errors=0 dropped=0 applied=2 apply_ms=901 now_ms=1000\r\n'


def report_line(data=None):
    return b'POST /api/device/report '+json.dumps(REPORT if data is None else data).encode()+b'\r\n'


class FeedbackTests(unittest.TestCase):
    def test_fragmented_mixed_topics_are_read_only_and_independent(self):
        feedback = ControlFeedback()
        data = report_line()+STATUS+VOICE+b'POST /api/device/alert {"ts":1000,"alert":"visual_cry"}\n'
        for index in range(0, len(data), 7):
            feedback.feed(data[index:index+7], 2000)
        state = feedback.snapshot(2000, connected=True)
        self.assertEqual(state['report']['data'], REPORT)
        self.assertEqual(state['voiceRx']['data']['applied'], 2)
        self.assertEqual(state['status']['data']['voice'], 1)
        self.assertEqual(state['alert']['data']['alert'], 'visual_cry')
        self.assertTrue(state['report']['fresh'])
        self.assertFalse(state['physicalStateMeasured'])
        self.assertFalse(hasattr(feedback, 'serial'))
        self.assertFalse(hasattr(feedback, 'write'))

    def test_stale_or_disconnected_records_never_present_as_fresh(self):
        feedback = ControlFeedback(); feedback.feed(report_line(), 1000)
        self.assertTrue(feedback.snapshot(6999, connected=True)['report']['fresh'])
        for now, connected in ((7000, True), (2000, False), (999, True)):
            self.assertFalse(feedback.snapshot(now, connected=connected)['report']['fresh'])

    def test_partial_oversized_malformed_and_unknown_frames(self):
        feedback = ControlFeedback()
        feedback.feed(b'POST /api/device/report '+b'x'*9000+b'\n', 1000)
        feedback.feed(b'status mode=not_a_mode\nvoice_rx v=2\nnoise\n', 1000)
        feedback.feed(b'POST /api/device/report {"ts":1,"ts":2}\n', 1000)
        self.assertGreaterEqual(feedback.decoder.rejected, 4)
        self.assertEqual(feedback.latest, {})
        feedback.feed(report_line()[:-1], 1000)
        self.assertEqual(feedback.latest, {})
        feedback.feed(b'\n', 1001)
        self.assertEqual(feedback.snapshot(1001, connected=True)['report']['data'], REPORT)

    def test_report_health_not_rewritten_and_snapshot_is_detached(self):
        feedback = ControlFeedback()
        report = {**REPORT, 'temp': None, 'hum': None, 'sensorHealth': {'dht11': 'read_failed'}}
        feedback.feed(report_line(report), 1000)
        first = feedback.snapshot(1000, connected=True)
        first['report']['data']['sensorHealth']['dht11'] = 'ok'
        second = feedback.snapshot(1000, connected=True)
        self.assertIsNone(second['report']['data']['temp'])
        self.assertEqual(second['report']['data']['sensorHealth']['dht11'], 'read_failed')

    def test_control_port_reads_once_and_decodes_both_protocols(self):
        class Serial:
            reads = writes = 0
            def read(self, size):
                self.reads += 1
                return (report_line()+STATUS+VOICE+
                    ('gimbal_link '+' '.join(f'{k}={v}' for k,v in capability().items())+'\n').encode()) if self.reads == 1 else b''
        link = ControlLink.__new__(ControlLink)
        link.serial = Serial(); link.feedback = ControlFeedback()
        link.buffer = bytearray(); link.status = None; link.received = 0
        self.assertEqual(link.poll(1000), capability())
        self.assertEqual(link.serial.reads, 1)
        self.assertEqual(link.serial.writes, 0)
        self.assertTrue(link.feedback.snapshot(1000, connected=True)['report']['fresh'])
        self.assertIsNone(link.poll(1400))

    def test_session_feedback_never_changes_motion_or_start_authorization(self):
        link = FakeLink(); link.feedback = ControlFeedback()
        link.feedback.feed(report_line()+STATUS+VOICE, 1000)
        session = GimbalSession(link); session.tick(1000)
        state = session.snapshot(1000)
        self.assertEqual(state['controlState'], 'idle')
        self.assertTrue(state['device']['report']['fresh'])
        self.assertEqual(link.writes, [])
        link.status = None; session.tick(1100)
        self.assertFalse(session.snapshot(1100)['device']['report']['fresh'])
        self.assertEqual(session.mode, 'idle')
        self.assertEqual(GimbalSession().snapshot(1000)['device'], {'connected': False})

    def test_execution_display_distinguishes_pending_from_accepted(self):
        link = FakeLink(); session = GimbalSession(link); session.tick(1000)
        session.command(dict(action='rock',client='test',generation=0,speed=20,duration=3000), 1000)
        session.tick(1000)
        self.assertEqual(session.snapshot(1000)['controlState'], 'pending')
        link.status['lease'] = session.lease; session.tick(1100)
        self.assertEqual(session.snapshot(1100)['controlState'], 'accepted')
        session.command(dict(action='stop',client='test',generation=session.generation), 1100)
        self.assertEqual(session.snapshot(1100)['controlState'], 'stopping')
        session.tick(1100)
        self.assertEqual(session.snapshot(1100)['controlState'], 'idle')


if __name__ == '__main__':
    unittest.main()
