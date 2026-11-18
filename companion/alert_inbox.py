"""Session-local reminders. Acknowledgement never calls a device command API."""

from __future__ import annotations

import copy
from datetime import datetime, timezone


MCU_NOTICES = {
    "sensor_fault": ("传感器故障", "检查主控报告的故障传感器。"),
    "data_unavailable": ("主控数据不足", "部分判断所需传感器尚不可用，不能据此判断状态正常。"),
    "temperature_alert": ("主控温度告警", "查看温度读数并现场核实；本系统不提供医疗判断。"),
    "env_warning": ("主控环境提醒", "查看环境传感器读数并现场核实。"),
    "presence_alert": ("主控在位提醒", "现场核实目标位置和在位传感器。"),
    "abnormal_crying": ("主控哭声提醒", "现场核实声音来源；不自动启动安抚机构。"),
    "uncomfortable": ("主控综合状态提醒", "查看相关传感器和实时画面，现场确认。"),
    "wet": ("主控尿湿提醒", "现场核实探头状态。"),
    "hungry": ("主控需求推测", "仅为主控推测，需人工核实。"),
}
KNOWN_MCU_STATES = frozenset(MCU_NOTICES) | {"normal", "sleepy"}


class AlertInbox:
    def __init__(self, capacity: int = 100, clear_ms: int = 5000) -> None:
        if capacity < 1 or clear_ms < 0:
            raise ValueError("invalid inbox limits")
        self.capacity = capacity
        self.clear_ms = clear_ms
        self.items: list[dict] = []
        self._next_id = 1
        self._seen_video = False
        self._seen_mcu = False
        self._clear_since: int | None = None
        self._last_clear_ms: int | None = None
        self.dropped = 0

    def _active(self, key: str) -> dict | None:
        return next((item for item in reversed(self.items)
                     if item["key"] == key and item["condition"] != "recovered"), None)

    def _raise(self, key: str, title: str, detail: str, source: str,
               now_ms: int, utc: str, *, verified: bool = True) -> None:
        item = self._active(key)
        if item is None:
            if len(self.items) >= self.capacity:
                oldest = next((i for i, old in enumerate(self.items)
                               if old["acknowledged"] and old["condition"] == "recovered"), 0)
                self.items.pop(oldest)
                self.dropped += 1
            item = dict(id=f"a{self._next_id}", key=key, title=title, detail=detail,
                        source=source, firstUtc=utc, firstMs=now_ms,
                        acknowledged=False, acknowledgedUtc=None)
            self._next_id += 1
            self.items.append(item)
        item.update(condition="active" if verified else "unverified",
                    lastMs=now_ms, lastUtc=utc)

    def _recover(self, key: str, now_ms: int, utc: str) -> None:
        item = self._active(key)
        if item is not None:
            item.update(condition="recovered", recoveredUtc=utc, recoveredMs=now_ms)

    def observe(self, snapshot: dict, now_ms: int, utc: str) -> None:
        record = snapshot.get("record") or {}
        if not record:
            return
        # No longer seeing a condition is not evidence that it recovered.
        for item in self.items:
            if item["condition"] != "recovered":
                item["condition"] = "unverified"
        observation = record.get("observation") or {}
        stopped = observation.get("reason") in {
            "stopped", "interrupted", "runtime_error", "end_of_stream"
        }
        frame_age = record.get("frameAgeMs")
        video = record.get("video") or {}
        frame_fresh = (snapshot.get("live") is True and type(frame_age) in (int, float)
                       and video.get("connectionState") == "streaming")
        if frame_fresh:
            self._seen_video = True
            self._recover("video_lost", now_ms, utc)
        elif self._seen_video and not stopped:
            self._raise("video_lost", "画面不可用", "实时画面已中断或过期，请检查相机连接。",
                        "连接状态", now_ms, utc)

        confirmed = frame_fresh and observation.get("reason") == "confirmed"
        label = observation.get("state")
        if confirmed and label == "Cry":
            source = "离线文件识别" if record.get("sourceKind") == "file" else "相机视觉识别"
            self._raise("visual_cry", "视觉 Cry 待确认",
                        "画面连续判为 Cry，请查看画面并现场确认；这不是哭声检测。",
                        source, now_ms, utc)
        if confirmed and label in {"Happy", "Neutral", "Sleep"}:
            if self._last_clear_ms is None or now_ms - self._last_clear_ms > 1500:
                self._clear_since = now_ms
            self._last_clear_ms = now_ms
            if now_ms - self._clear_since >= self.clear_ms:
                self._recover("visual_cry", now_ms, utc)
        else:
            self._clear_since = self._last_clear_ms = None

        device = record.get("device") or {}
        report = device.get("report") or {}
        status = device.get("status") or {}
        online = not stopped and device.get("online") is True
        report_fresh = online and report.get("fresh") is True
        if report_fresh or online and status.get("fresh") is True:
            self._seen_mcu = True
            self._recover("mcu_lost", now_ms, utc)
        elif self._seen_mcu and not stopped:
            self._raise("mcu_lost", "主控数据不可用", "主控数据中断或过期，不能确认当前设备状态。",
                        "连接状态", now_ms, utc)
        if report_fresh:
            code = (report.get("data") or {}).get("status")
            if isinstance(code, str) and code in KNOWN_MCU_STATES:
                for item in self.items:
                    if item["key"].startswith("mcu:") and item["key"] != f"mcu:{code}":
                        self._recover(item["key"], now_ms, utc)
                if code in MCU_NOTICES:
                    title, detail = MCU_NOTICES[code]
                    self._raise(f"mcu:{code}", title, detail, "主控报告", now_ms, utc)

    def acknowledge(self, alert_id: str, utc: str) -> bool:
        for item in self.items:
            if item["id"] == alert_id:
                if not item["acknowledged"]:
                    item.update(acknowledged=True, acknowledgedUtc=utc)
                return True
        return False

    def apply_rules(self, results: list[dict], now_ms: int, utc: str) -> None:
        for rule in results:
            if rule["active"]:
                self._raise(rule["key"], rule["title"], rule["detail"], "自定义监测规则",
                            now_ms, utc, verified=rule["state"] in {"active", "recovering"})
            elif rule["state"] == "normal":
                self._recover(rule["key"], now_ms, utc)

    def snapshot(self) -> dict:
        items = copy.deepcopy(self.items)
        return dict(items=items, pending=sum(not item["acknowledged"] for item in items),
                    attention=sum(not item["acknowledged"] and item["condition"] != "recovered"
                                  for item in items), dropped=self.dropped,
                    scope="session_local_acknowledgement_only")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
