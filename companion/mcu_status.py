"""Validate software status lines without inferring physical actuator state."""

import re

BITS = ("il_stop", "il_arm", "il_ready", "il_en", "il_pg", "il_pending")
MODES = {"idle", "sleep", "soothe", "alert", "fault"}


def parse_status(line):
    if len(line) > 2048 or not line.endswith(b"\n"):
        raise ValueError("Incomplete or oversized status line")
    tokens = line.rstrip(b"\r\n").decode("ascii").split()
    if not tokens or tokens[0] != "status":
        raise ValueError("Not a status line")
    fields = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator != "=" or not key or not value or key in fields:
            raise ValueError("Invalid or duplicate status field")
        fields[key] = value
    if fields.get("mode") not in MODES:
        raise ValueError("Unknown mode")
    if re.fullmatch(r"0x[0-9A-Fa-f]{2}", fields.get("voice", "")) is None:
        raise ValueError("Invalid voice ID")
    if any(fields.get(key) not in ("0", "1") for key in BITS):
        raise ValueError("Missing or invalid interlock bit")
    if fields.get("il_trip") not in ("0", "1", "2", "3", "4"):
        raise ValueError("Missing or invalid trip reason")
    result = {"mode": fields["mode"], "voice": int(fields["voice"], 16),
              **{key: int(fields[key]) for key in (*BITS, "il_trip")}}
    if "dht_n" in fields or "dht_raw" in fields:
        if fields.get("dht_n") not in ("0", "1", "2", "3", "4", "5") or \
                re.fullmatch(r"[0-9a-fA-F]{10}", fields.get("dht_raw", "")) is None:
            raise ValueError("Invalid DHT11 diagnostic frame")
        received = int(fields["dht_n"])
        raw = bytes.fromhex(fields["dht_raw"])
        if any(raw[received:]):
            raise ValueError("DHT11 diagnostics include unreceived bytes")
        result["dhtRaw"] = {"bytesReceived": received, "bytes": list(raw)}
        if fields.get("dht") in ("0", "1", "2", "3", "4", "5"):
            result["dhtResult"] = int(fields["dht"])
    return result


def parse_motion(line):
    """Parse requested outputs, never measured motor speed or physical position."""
    if len(line) > 320 or not line.endswith(b"\n"):
        raise ValueError("Incomplete or oversized motion line")
    tokens = line.rstrip(b"\r\n").decode("ascii").split()
    if not tokens or tokens[0] != "motion":
        raise ValueError("Not a motion line")
    limits = {"v": (1, 1), "action": (0, 3), "motor_cmd": (-20, 20),
              "pan_target": (0, 180), "tilt_target": (0, 180),
              "remaining_ms": (0, 10000), "rejected": (0, 0xFFFFFFFF)}
    fields = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator != "=" or key not in limits or key in fields or \
                re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None:
            raise ValueError("Invalid or duplicate motion field")
        number = int(value)
        low, high = limits[key]
        if not low <= number <= high:
            raise ValueError("Motion field out of range")
        fields[key] = number
    if fields.keys() != limits.keys():
        raise ValueError("Missing motion field")
    if fields['action'] in (0, 2) and fields['motor_cmd'] != 0:
        raise ValueError("Motor target contradicts motion owner")
    if fields['action'] == 0 and fields['remaining_ms'] != 0:
        raise ValueError("Idle motion cannot have a remaining duration")
    return fields
