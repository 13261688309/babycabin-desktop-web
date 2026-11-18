import copy
import http.client
import json
import unittest

from companion.care_workflow import CareWorkflow
from companion.gimbal_console import ConsoleServer, GimbalSession
from companion.state import Detection
from test_gimbal import FakeLink


BOX = (25, 25, 75, 75)


class CareTests(unittest.TestCase):
    def sample(self, care, at, label='Cry', *, items=None, captured=None):
        care.observe([Detection(label, .99, BOX)] if items is None else items,
                     at if captured is None else captured, at, 100, 100)
        care.tick(at, video_fresh=True, device=dict(connected=False))

    def baseline(self):
        care = CareWorkflow()
        for at in range(1000, 4200, 200): self.sample(care, at)
        return care

    def following(self, hardware=False, accepted=False):
        care = self.baseline()
        care.begin(4000, source='manual', hardware=hardware)
        for at in range(4200, 6200, 200): self.sample(care, at)
        care.end(6000, completed=True, accepted=accepted)
        return care

    def test_confirmed_reminder_ack_and_no_flood(self):
        care = self.baseline()
        self.assertEqual(care.snapshot(4000)['alertInbox']['pending'], 1)
        care.acknowledge('a1', 4100); care.acknowledge('a1', 4101)
        for at in range(4200, 10000, 200): self.sample(care, at)
        result = care.snapshot(10000)
        self.assertEqual(result['alertInbox']['pending'], 0)
        self.assertEqual(len(result['alertInbox']['items']), 1)
        self.assertEqual(result['alertInbox']['items'][0]['condition'], 'active')
        self.assertEqual([event['kind'] for event in result['events']], ['reminder', 'ack'])
        self.assertFalse(result['automaticActionOnAcknowledgement'])

    def test_arm_does_not_erase_reminder_and_ack_never_changes_control(self):
        link = FakeLink(); session = GimbalSession(link); session.tick(900)
        for at in range(1000, 4200, 200):
            session.frame([BOX], 100, 100, at, at)
            session.emotion_frame([Detection('Cry', .99, BOX)], at, at, 100, 100)
        session.command(dict(action='arm_soothe', client='owner', generation=session.generation), 4000)
        before = (session.generation, session.mode, session.lease, session.owner, session.heartbeat)
        session.snapshot(4000)
        session.care_command(dict(action='ack', id='a1'), 4100)
        self.assertEqual(before, (session.generation, session.mode, session.lease, session.owner, session.heartbeat))
        self.assertEqual(link.writes, [])
        self.assertEqual(session.care.snapshot(4100)['alertInbox']['pending'], 0)

    def test_confirmed_recovery_and_new_reminder(self):
        care = self.baseline(); care.acknowledge('a1', 4100)
        for at in range(4200, 12600, 200): self.sample(care, at, 'Neutral')
        self.assertEqual(care.inbox.items[0]['condition'], 'recovered')
        for at in range(12600, 16000, 200): self.sample(care, at)
        self.assertEqual(care.inbox.snapshot()['pending'], 1)
        self.assertEqual(care.inbox.items[-1]['id'], 'a2')

    def test_stale_or_unknown_not_recovery_and_not_blanket_normal(self):
        care = self.baseline()
        care.tick(6000, video_fresh=False, device=dict(connected=False))
        self.assertEqual(care.inbox.items[0]['condition'], 'unverified')
        self.assertEqual(care.inbox.items[-1]['key'], 'video_lost')
        self.sample(care, 6200, items=[])
        self.assertEqual(care.inbox.items[0]['condition'], 'unverified')

    def test_mcu_reminders_and_lost_link_share_inbox(self):
        care = CareWorkflow()
        device = dict(connected=True, report=dict(fresh=True, data=dict(status='sensor_fault')))
        care.tick(1000, video_fresh=False, device=device)
        self.assertEqual(care.inbox.items[0]['key'], 'mcu:sensor_fault')
        care.acknowledge('a1', 1001)
        care.tick(8000, video_fresh=False, device=dict(connected=False))
        self.assertEqual(care.inbox.items[0]['condition'], 'unverified')
        self.assertEqual(care.inbox.items[-1]['key'], 'mcu_lost')

    def test_followup_changed_label_is_not_efficacy_or_retry(self):
        care = self.following()
        self.assertEqual(care.run['state'], 'following')
        for at in range(6200, 26200, 200): self.sample(care, at, 'Neutral')
        result = care.snapshot(26000)
        self.assertEqual(result['followup']['state'], 'label_changed')
        self.assertEqual((result['followup']['before'], result['followup']['after']), ('Cry', 'Neutral'))
        self.assertFalse(result['followup']['physicalMotionMeasured'])
        self.assertIn('不等于', result['followup']['reason'])
        self.assertFalse(result['automaticRetry'])
        self.assertEqual(result['followup']['samples'], 100)

    def test_followup_persistent_cry_never_retries(self):
        care = self.following()
        for at in range(6200, 26200, 200): self.sample(care, at)
        self.assertEqual(care.run['state'], 'cry_persists')
        self.assertIn('不会再次', care.run['reason'])
        previous = copy.deepcopy(care.run)
        self.sample(care, 27000)
        self.assertEqual(previous, care.run)

    def test_repeated_poll_and_pre_end_inference_do_not_add_samples(self):
        care = self.following()
        for at in range(6000, 7200, 10):
            self.sample(care, at, 'Neutral', captured=6000)
        self.assertEqual(care.run['samples'], 0)
        self.sample(care, 7300, 'Neutral')
        self.assertEqual(care.run['samples'], 1)
        self.assertEqual(care.run['after'], 'Unknown')
        care.tick(8800, video_fresh=True, device=dict(connected=False))
        self.assertEqual(care.run['state'], 'insufficient')

    def test_followup_lost_multiple_changed_or_stale_finishes_without_reacquire(self):
        for items in ([], [Detection('Cry', .99, BOX)]*2, [Detection('Cry', .99, (0, 0, 10, 10))],
                      [Detection('Cry', .79, BOX)], [Detection('Cry', .99, (0, 0, 101, 101))]):
            care = self.following(); self.sample(care, 6200, items=items)
            self.assertEqual(care.run['state'], 'insufficient')
            self.sample(care, 6400)
            self.assertEqual(care.run['state'], 'insufficient')

    def test_completed_requires_continuous_target_and_hardware_acceptance(self):
        for hardware, accepted in ((True, False),):
            care = self.following(hardware=hardware, accepted=accepted)
            self.assertEqual(care.run['state'], 'unavailable')
        self.assertEqual(self.following(hardware=True, accepted=True).run['state'], 'following')
        care = self.baseline(); care.begin(4000, source='manual', hardware=False)
        self.sample(care, 4200, items=[]); self.sample(care, 4400)
        care.end(4500, completed=True, accepted=False)
        self.assertEqual(care.run['state'], 'unavailable')
        care = CareWorkflow(); care.begin(1000, source='manual', hardware=False)
        care.end(3000, completed=True, accepted=False)
        self.assertEqual(care.run['state'], 'unavailable')

    def test_interruption_and_new_action_do_not_create_post_result(self):
        care = self.following(); care.interrupt('停止', 6100)
        self.assertEqual(care.run['state'], 'interrupted')
        care.begin(6200, source='manual', hardware=False)
        care.end(6300, completed=False, accepted=False)
        self.assertEqual(care.run['state'], 'interrupted')

    def test_events_and_reminders_bounded_and_detached(self):
        care = self.baseline()
        for at in range(100): care.event('test', str(at), at)
        self.assertEqual(len(care.snapshot(4000)['events']), 50)
        snapshot = care.snapshot(4000); snapshot['events'].clear(); snapshot['alertInbox']['items'].clear()
        self.assertEqual(len(care.events), 50); self.assertEqual(len(care.inbox.items), 1)
        for payload in ({'action':'ack','id':None}, {'action':'ack','id':'missing'},
                        {'action':'rock','id':'a1'}, {'action':'ack','id':'a1','enable':True}):
            with self.assertRaises(ValueError): GimbalSession().care_command(payload, 1000)

    def test_camera_reset_inflight_frame_and_resolution_change_cannot_compare(self):
        care = self.following(); care.invalidate('camera_changed', 6200)
        self.assertEqual(care.run['state'], 'insufficient')
        self.sample(care, 6300, captured=6100)
        self.assertIsNone(care.captured)
        for at in range(6400, 9400, 200): self.sample(care, at)
        self.assertEqual(care.visual.current.state, 'Unknown')
        care = self.following()
        care.observe([Detection('Neutral', .99, BOX)], 6200, 6200, 200, 200)
        self.assertEqual(care.run['state'], 'insufficient')

    def test_session_natural_completion_is_read_only_after_stop(self):
        link = FakeLink(); session = GimbalSession(link); session.tick(900)
        for at in range(1000, 4200, 200):
            session.frame([BOX], 100, 100, at, at)
            session.emotion_frame([Detection('Cry', .99, BOX)], at, at, 100, 100)
        def command(action, now, **extra):
            session.command(dict(action=action, client='owner', generation=session.generation, **extra), now)
        command('rock', 4000, speed=10, duration=2000); session.tick(4000)
        link.status['lease'] = session.lease
        for at in range(4200, 26200, 200):
            if session.mode != 'idle': command('heartbeat', at)
            session.frame([BOX], 100, 100, at, at)
            session.emotion_frame([Detection('Cry', .99, BOX)], at, at, 100, 100)
            session.tick(at)
        self.assertEqual(session.mode, 'idle')
        self.assertEqual(session.care.run['state'], 'cry_persists')
        self.assertEqual([entry['op'] for entry in link.writes].count(4), 1)
        self.assertEqual(link.writes[-1]['op'], 3)
        writes = copy.deepcopy(link.writes); session.tick(27000)
        self.assertEqual(writes, link.writes)


class CareHttpTests(unittest.TestCase):
    def setUp(self):
        self.session = GimbalSession()
        self.session.care.inbox._raise('visual_cry', '<img src=x>', 'test', 'fixture', 1, 'now')
        self.server = ConsoleServer(self.session); self.server.start()

    def tearDown(self): self.server.close()

    def post(self, data, **headers):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server.server_port)
        connection.request('POST', '/api/care', data, headers={
            'Origin':self.server.url.rstrip('/'), 'Content-Type':'application/json',
            'X-Control-Token':self.session.token, **headers})
        response = connection.getresponse(); body = response.read(); connection.close()
        return response.status, body

    def test_ack_endpoint_idempotent_and_no_control_generation_change(self):
        before = self.session.generation
        for _ in range(2):
            status, body = self.post(b'{"action":"ack","id":"a1"}')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['care']['alertInbox']['pending'], 0)
        self.assertEqual(self.session.generation, before)
        self.assertEqual(self.session.mode, 'idle')
        self.assertEqual(self.session.lease, 0)

    def test_ack_endpoint_origin_host_token_and_schema_enforced(self):
        for headers in ({'Origin':'null'}, {'Host':'example.com'}, {'X-Control-Token':'bad'}):
            self.assertEqual(self.post(b'{"action":"ack","id":"a1"}', **headers)[0], 403)
        for body in (b'[]', b'{"action":"ack","id":1}', b'{"action":"ack","id":"a1","id":"a1"}',
                     b'{"action":"rock","id":"a1"}', b'{"action":"ack","id":"missing"}', b'{"action":"ack","id":"a1","enable":true}'):
            self.assertEqual(self.post(body)[0], 400)
        self.assertEqual(self.session.care.inbox.snapshot()['pending'], 1)


if __name__ == '__main__': unittest.main()
