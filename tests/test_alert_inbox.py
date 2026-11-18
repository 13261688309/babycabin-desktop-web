"""Offline behaviour and HTTP boundary tests for local acknowledgement."""

import copy
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from companion.alert_inbox import AlertInbox
from companion.dashboard import DashboardServer, DashboardState


def record(label="Cry", reason="confirmed", frame_age=10, status="normal"):
    return dict(sourceKind="network", frameAgeMs=frame_age,
                video={"connectionState": "streaming"},
                observation={"state": label, "reason": reason},
                device={"online": True, "staleMs": 6000,
                        "report": {"fresh": True, "ageMs": 0,
                                   "data": {"ts": 100, "status": status}},
                        "status": {"fresh": True, "ageMs": 0,
                                   "data": {"il_en": 0, "mode": "idle"}}})


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.inbox = AlertInbox()

    def observe(self, now=0, **changes):
        self.inbox.observe({"record": record(**changes), "live": True}, now, str(now))
        return self.inbox.snapshot()

    def test_only_confirmed_fresh_cry_raises(self):
        for reason in ("confirming", "ambiguous_class", "low_confidence", "no_detection"):
            self.assertEqual(self.observe(reason=reason)["pending"], 0)
        self.inbox = AlertInbox()
        self.assertEqual(self.observe(frame_age=None)["pending"], 0)
        result = self.observe()
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["items"][0]["key"], "visual_cry")
        result["items"].clear()
        self.assertEqual(len(self.inbox.snapshot()["items"]), 1)

    def test_ack_is_idempotent_does_not_clear_or_rearm_continuous_cry(self):
        first = self.observe()["items"][0]
        self.assertTrue(self.inbox.acknowledge(first["id"], "first"))
        self.assertTrue(self.inbox.acknowledge(first["id"], "second"))
        for now in range(100, 10000, 100):
            result = self.observe(now)
        self.assertEqual(result["pending"], 0)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["condition"], "active")
        self.assertEqual(result["items"][0]["acknowledgedUtc"], "first")
        self.assertFalse(self.inbox.acknowledge("unknown", "now"))

    def test_confirmed_recovery_then_new_cry_is_new_reminder(self):
        self.observe()
        self.inbox.acknowledge("a1", "ack")
        for now in range(1000, 6000, 1000):
            self.assertNotEqual(self.observe(now, label="Neutral")["items"][0]["condition"], "recovered")
        self.assertEqual(self.observe(6000, label="Neutral")["items"][0]["condition"], "recovered")
        result = self.observe(7000)
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["items"][-1]["id"], "a2")

    def test_unknown_and_gaps_do_not_prove_recovery(self):
        self.observe()
        for now in (1000, 2000, 3000):
            self.observe(now, label="Neutral")
        self.observe(4000, label="Unknown", reason="no_detection")
        for now in (5000, 6000, 7000, 8000):
            result = self.observe(now, label="Happy")
        self.assertEqual(result["items"][0]["condition"], "unverified")
        result = self.observe(20000, label="Happy")
        self.assertEqual(result["items"][0]["condition"], "unverified")

    def test_stale_capture_does_not_create_alarm_and_missing_stream_not_normal(self):
        state = DashboardState()
        with patch("companion.dashboard.time.monotonic_ns", return_value=0):
            state.publish(record(frame_age=1501))
            self.assertEqual(state.snapshot()["alertInbox"]["pending"], 0)
            state.publish(record())
        with patch("companion.dashboard.time.monotonic_ns", return_value=2_000_000_000):
            snapshot = state.snapshot()
            items = {item["key"]: item for item in snapshot["alertInbox"]["items"]}
            self.assertEqual(items["visual_cry"]["condition"], "unverified")
            self.assertEqual(items["video_lost"]["condition"], "active")
            self.assertNotIn("mcu_lost", items)
            self.assertEqual(snapshot["record"]["observation"]["state"], "Unknown")
        with patch("companion.dashboard.time.monotonic_ns", return_value=7_000_000_000):
            self.assertIn("mcu_lost", [x["key"] for x in state.snapshot()["alertInbox"]["items"]])

    def test_disconnect_only_after_connected_and_does_not_flood(self):
        state = DashboardState()
        state.publish({"sourceKind": "serial_only", "observation": {"reason": "telemetry_only"}})
        self.assertEqual(state.snapshot()["alertInbox"]["pending"], 0)
        state.publish(record(label="Neutral"))
        offline = record(label="Unknown", reason="stream_lost")
        offline["device"]["online"] = False
        offline["video"]["connectionState"] = "reconnecting"
        for _ in range(10):
            state.publish(offline)
        items = state.snapshot()["alertInbox"]["items"]
        self.assertEqual({x["key"] for x in items}, {"video_lost", "mcu_lost"})
        state.publish(record(label="Neutral"))
        self.assertTrue(all(x["condition"] == "recovered" for x in state.snapshot()["alertInbox"]["items"]))

    def test_stop_preserves_history_without_claiming_live_alarms(self):
        state = DashboardState()
        state.publish(record(status="sensor_fault"))
        state.publish(record(reason="stopped"))
        items = state.snapshot()["alertInbox"]["items"]
        self.assertEqual(len(items), 2)
        self.assertTrue(all(x["condition"] == "unverified" for x in items))

    def test_mcu_report_not_ack_resolves_condition(self):
        self.observe(label="Neutral", status="sensor_fault")
        self.inbox.acknowledge("a1", "ack")
        self.assertEqual(self.observe(100, label="Neutral", status="sensor_fault")["items"][0]["condition"], "active")
        self.assertEqual(self.observe(200, label="Neutral", status="unexpected")["items"][0]["condition"], "unverified")
        self.assertEqual(self.observe(300, label="Neutral")["items"][0]["condition"], "recovered")
        self.assertEqual(self.observe(400, label="Neutral", status="sensor_fault")["pending"], 1)

    def test_malformed_mcu_status_never_crashes_or_clears_reminder(self):
        self.observe(label="Neutral", status="sensor_fault")
        for value in ([], {}, None, 0, True):
            result = self.observe(100, label="Neutral", status=value)
            self.assertEqual(result["pending"], 1)
            self.assertEqual(result["items"][0]["condition"], "unverified")

    def test_bounded_history_and_offline_source_are_explicit(self):
        inbox = AlertInbox(capacity=2, clear_ms=0)
        for now, status in enumerate(("sensor_fault", "normal", "wet", "normal", "env_warning")):
            data = record(label="Neutral", status=status)
            inbox.observe({"record": data, "live": True}, now, str(now))
        self.assertEqual(len(inbox.items), 2)
        self.assertEqual(inbox.dropped, 1)
        data = record()
        data["sourceKind"] = "file"
        inbox.observe({"record": data, "live": True}, 20, "20")
        self.assertEqual(inbox.items[-1]["source"], "离线文件识别")


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.server = DashboardServer(0)
        self.server.publish(record())
        self.server.start()
        self.headers = {"Origin": self.server.url.rstrip("/"),
                        "X-Alert-Token": self.server.state.ack_token,
                        "Content-Type": "application/json"}

    def tearDown(self):
        self.server.close()

    def post(self, body=b'{"id":"a1"}', headers=None, path="api/alerts/ack"):
        return urlopen(Request(self.server.url + path, data=body,
                               headers=self.headers if headers is None else headers), timeout=3)

    def test_ack_changes_local_state_only_and_survives_poll(self):
        before = copy.deepcopy(self.server.state._record)
        with self.post() as response:
            self.assertEqual(json.load(response), {"acknowledged": True, "deviceCommandSent": False})
        self.assertEqual(self.server.state._record, before)
        with urlopen(self.server.url + "api/state") as response:
            self.assertEqual(json.load(response)["alertInbox"]["pending"], 0)
        with urlopen(self.server.url + "api/report.json") as response:
            report = json.load(response)
            self.assertNotIn("ackToken", report)
            self.assertEqual(report["alertInbox"]["items"][0]["condition"], "active")

    def test_origin_token_host_required(self):
        for changed in ({"Origin": "http://example.invalid"}, {"X-Alert-Token": "bad"},
                        {"Host": "example.invalid"}, {"Origin": "null"}):
            with self.subTest(changed=changed), self.assertRaises(HTTPError) as result:
                self.post(headers={**self.headers, **changed})
            self.assertEqual(result.exception.code, 403)
        with self.assertRaises(HTTPError) as result:
            urlopen(Request(self.server.url + "api/state", headers={"Host": "example.invalid"}))
        self.assertEqual(result.exception.code, 403)

    def test_invalid_body_and_no_motion_route(self):
        for body in (b'{"id":"a1","enable":1}', b'{"id":"a1","id":"a1"}',
                     b'[]', b'{"id":1}', b'NaN', b'0' * 257, b'{broken'):
            with self.subTest(body=body), self.assertRaises(HTTPError) as result:
                self.post(body)
            self.assertEqual(result.exception.code, 400)
        for path in ("api/control", "api/actuator", "api/arm", "api/stop"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as result:
                self.post(path=path)
            self.assertEqual(result.exception.code, 501)
        with self.assertRaises(HTTPError) as result:
            self.post(b'{"id":"missing"}')
        self.assertEqual(result.exception.code, 404)
        self.assertEqual(self.server.state.snapshot()["alertInbox"]["pending"], 1)


if __name__ == "__main__":
    unittest.main()
