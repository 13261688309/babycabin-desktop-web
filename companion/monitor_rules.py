"""Configurable, time-qualified observation rules; never actuator commands."""

import copy
import math


DEFAULT_RULES = {
    "temperature": dict(enabled=False, low=15.0, high=35.0, hysteresis=1.0, holdSeconds=10),
    "humidity": dict(enabled=False, low=20.0, high=80.0, hysteresis=3.0, holdSeconds=10),
    "no_target": dict(enabled=False, holdSeconds=30),
}
TITLES = {"temperature": "环境温度持续越界", "humidity": "环境湿度持续越界",
          "no_target": "画面持续未检测到目标"}


def validate_rules(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_RULES):
        raise ValueError("Expected all three monitoring rules")
    for name, rule in value.items():
        if not isinstance(rule, dict) or set(rule) != set(DEFAULT_RULES[name]):
            raise ValueError("Unexpected rule fields")
        if type(rule["enabled"]) is not bool or type(rule["holdSeconds"]) is not int:
            raise ValueError("Invalid rule types")
        if not 5 <= rule["holdSeconds"] <= 600:
            raise ValueError("Hold duration must be 5..600 seconds")
        if name != "no_target":
            numbers = [rule[key] for key in ("low", "high", "hysteresis")]
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in numbers):
                raise ValueError("Expected finite rule bounds")
            low, high, hysteresis = numbers
            maximum = 50 if name == "temperature" else 100
            if not (0 <= low < high <= maximum and 0 < hysteresis < (high - low) / 2):
                raise ValueError("Invalid range or recovery hysteresis")
    return copy.deepcopy(value)


class MonitorRules:
    def __init__(self):
        self.config = copy.deepcopy(DEFAULT_RULES)
        self.revision = 1
        self._track = {}

    def configure(self, value):
        value = validate_rules(value)
        if value != self.config:
            self.config = value
            self.revision += 1
            self._track.clear()

    @staticmethod
    def _sample(name, snapshot, now):
        record = snapshot.get("record") or {}
        if name == "no_target":
            observation = record.get("observation") or {}
            reason = observation.get("reason")
            captured = record.get("frameCapturedMs")
            if (snapshot.get("live") is not True or record.get("sourceKind") not in {"network", "camera"}
                    or (record.get("video") or {}).get("connectionState") != "streaming"
                    or type(captured) is not int or not 0 <= now - captured <= 1500):
                return None
            if reason == "no_detection" and not record.get("detections"):
                return captured, captured, True, 1500
            if reason == "confirmed" and observation.get("state") in {"Cry", "Happy", "Neutral", "Sleep"}:
                return captured, captured, False, 1500
            return None
        device = record.get("device") or {}
        report = device.get("report") or {}
        data = report.get("data") or {}
        age = report.get("ageMs")
        value = data.get("temp" if name == "temperature" else "hum")
        if (device.get("online") is not True or report.get("fresh") is not True
                or (data.get("sensorHealth") or {}).get("dht11") != "ok"
                or type(data.get("ts")) is not int or type(age) not in (int, float)
                or not 0 <= age <= 6000 or type(value) not in (int, float)
                or not math.isfinite(value) or not 0 <= value <= (50 if name == "temperature" else 100)):
            return None
        return data["ts"], now - age, value, 6000

    def observe(self, snapshot, now):
        results = []
        for name, rule in self.config.items():
            track = self._track.setdefault(name, dict(active=False, token=None, last=None,
                since=None, clear=None, state="waiting", elapsed=0, value=None, edge=None))
            sample = self._sample(name, snapshot, now) if rule["enabled"] else None
            if sample is None:
                track.update(token=None, last=None, since=None, clear=None, elapsed=0,
                             state=("unverified" if track["active"] else "waiting") if rule["enabled"] else "disabled")
            else:
                token, sample_ms, value, gap = sample
                if token != track["token"]:
                    if (track["last"] is None or not 0 <= sample_ms - track["last"] <= gap
                            or track["token"] is not None and token < track["token"]):
                        track.update(since=None, clear=None, edge=None)
                    track.update(token=token, last=sample_ms, value=value)
                    bad = value if name == "no_target" else value < rule["low"] or value > rule["high"]
                    normal = not value if name == "no_target" else rule["low"] + rule["hysteresis"] <= value <= rule["high"] - rule["hysteresis"]
                    if track["active"]:
                        if normal:
                            if track["clear"] is None:
                                track["clear"] = sample_ms
                            track["state"] = "recovering"
                            if sample_ms - track["clear"] >= 5000:
                                track.update(active=False, state="normal", since=None, clear=None, elapsed=0)
                        else:
                            track.update(clear=None, state="active")
                    elif bad:
                        edge = "absent" if name == "no_target" else "low" if value < rule["low"] else "high"
                        if track["since"] is None or track["edge"] != edge:
                            track.update(since=sample_ms, edge=edge)
                        track["elapsed"] = max(0, sample_ms - track["since"])
                        track["active"] = track["elapsed"] >= rule["holdSeconds"] * 1000
                        track["state"] = "active" if track["active"] else "confirming"
                    else:
                        track.update(state="normal", since=None, elapsed=0, edge=None)
            detail = ("检测模型持续没有找到目标，请现场核实；遮挡或模型漏检也可能造成此提醒。"
                      if name == "no_target" else
                      f"环境读数持续超出自定义区间 {rule['low']:g}–{rule['high']:g}"
                      + (" °C。" if name == "temperature" else " %RH。"))
            results.append(dict(name=name, key=f"rule:{self.revision}:{name}", title=TITLES[name],
                detail=detail, state=track["state"], active=track["active"], value=track["value"],
                elapsedMs=track["elapsed"], holdMs=rule["holdSeconds"] * 1000))
        return dict(revision=self.revision, config=copy.deepcopy(self.config), results=results,
                    recoverySeconds=5, persistence="current_process_only", deviceCommands=False)
