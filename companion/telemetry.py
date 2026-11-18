"""Bounded decoder for the firmware's USART1 report and alert lines."""

from __future__ import annotations

import copy
import json
import math
import threading
import time

PREFIXES = {
    b"POST /api/device/report ": "report",
    b"POST /api/device/alert ": "alert",
}


def reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON number: {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def finite_values(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    return True


class TelemetryDecoder:
    def __init__(self, max_line: int = 2048) -> None:
        if max_line < 128:
            raise ValueError("max_line too small")
        self.max_line = max_line
        self._buffer = bytearray()
        self._discarding = False
        self.rejected = 0

    def feed(self, chunk: bytes) -> list[dict]:
        records = []
        for byte in chunk:
            if byte == 10:
                if not self._discarding and self._buffer:
                    record = self._decode(bytes(self._buffer).rstrip(b"\r"))
                    if record is not None:
                        records.append(record)
                self._buffer.clear()
                self._discarding = False
            elif not self._discarding:
                if len(self._buffer) >= self.max_line:
                    self._buffer.clear()
                    self._discarding = True
                    self.rejected += 1
                else:
                    self._buffer.append(byte)
        return records

    def _decode(self, line: bytes) -> dict | None:
        for prefix, kind in PREFIXES.items():
            if not line.startswith(prefix):
                continue
            try:
                data = json.loads(
                    line[len(prefix):],
                    parse_constant=reject_constant,
                    object_pairs_hook=unique_object,
                )
                if not isinstance(data, dict) or not finite_values(data):
                    raise ValueError("expected finite JSON object")
                if type(data.get("ts")) is not int or not 0 <= data["ts"] <= 0xFFFFFFFF:
                    raise ValueError("invalid firmware timestamp")
                if kind == "report":
                    if type(data.get("schemaVersion")) is not int or data["schemaVersion"] != 2:
                        raise ValueError("unsupported report version")
                    if not isinstance(data.get("sensorHealth"), dict):
                        raise ValueError("missing sensor health")
                    for key in ("temp", "hum", "sound", "irObjectTemp", "irAmbientTemp"):
                        value = data.get(key)
                        if value is not None and type(value) not in (int, float):
                            raise ValueError("invalid observation type")
                elif not isinstance(data.get("alert"), str):
                    raise ValueError("missing alert name")
                return {"type": kind, "data": data}
            except (ValueError, UnicodeError, RecursionError):
                self.rejected += 1
                return None
        return None


class SerialTelemetry:
    """Read-only link: this class deliberately has no command/write API."""

    def __init__(self, port: str, baud: int = 115200, stale_ms: int = 6000,
                 reconnect_seconds: float = 0) -> None:
        if (type(reconnect_seconds) not in (int, float) or not math.isfinite(reconnect_seconds)
                or not 0 <= reconnect_seconds <= 300):
            raise ValueError("reconnect seconds must be finite and in 0..300")
        import serial

        self.decoder = self._new_decoder()
        self.stale_ms = stale_ms
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._port = port
        self._baud = baud
        self._serial_factory = serial.Serial
        self._reconnect_seconds = reconnect_seconds
        self._connected = False
        self._connection_state = "disconnected"
        self._reconnect_attempts = 0
        self._next_retry_at = 0.0
        self._open_cleanup_failed = False
        self._serial = self._open_link()
        try:
            self._connected = True
            self._connection_state = "connected"
            self._thread = threading.Thread(target=self._run, name="telemetry-reader", daemon=True)
            self._thread.start()
        except BaseException:
            self._stop.set()
            try:
                self._serial.close()
            except Exception:
                # Preserve the startup failure even if backend cleanup also fails.
                pass
            raise

    def _new_decoder(self):
        return TelemetryDecoder()

    def _open_link(self):
        link = self._serial_factory(port=None, baudrate=self._baud, timeout=0.2)
        try:
            link.dtr = False
            link.rts = False
            link.port = self._port
            link.open()
            return link
        except BaseException:
            try:
                link.close()
            except Exception:
                self._open_cleanup_failed = True
            raise

    def _drop_link(self, state: str) -> bool:
        with self._lock:
            link, self._serial = self._serial, None
            self._latest.clear()
            self._connected = False
            self._connection_state = state
        self.decoder = self._new_decoder()
        if link is not None:
            try:
                link.close()
            except Exception:
                with self._lock:
                    self._connection_state = "cleanup_failed"
                return False
        return True

    def _recover(self, deadline: float) -> bool:
        if not self._drop_link("reconnecting"):
            return False
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= deadline:
                break
            delay = min(max(0.0, self._next_retry_at - now), deadline - now)
            if delay and self._stop.wait(delay):
                return False
            if self._stop.is_set() or time.monotonic() >= deadline:
                break
            self._next_retry_at = time.monotonic() + 0.25
            with self._lock:
                self._reconnect_attempts += 1
            try:
                link = self._open_link()
            except (OSError, ValueError):
                if self._open_cleanup_failed:
                    with self._lock:
                        self._connection_state = "cleanup_failed"
                    return False
                continue
            # A slow OS open must not publish a link after cancellation/expiry.
            with self._lock:
                accepted = not self._stop.is_set() and time.monotonic() < deadline
                if accepted:
                    self._serial = link
                    self._connection_state = "awaiting_data"
            if accepted:
                return True
            try:
                link.close()
            except Exception:
                with self._lock:
                    self._connection_state = "cleanup_failed"
                return False
            break
        with self._lock:
            self._connection_state = "closed" if self._stop.is_set() else "reconnect_expired"
        return False

    def _run(self) -> None:
        deadline = None
        while not self._stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                self._drop_link("reconnect_expired")
                return
            try:
                with self._lock:
                    link = self._serial
                if link is None:
                    return
                chunk = link.read(512)
                if deadline is not None and time.monotonic() >= deadline:
                    self._drop_link("reconnect_expired")
                    return
                records = self.decoder.feed(chunk)
                for record in records:
                    with self._lock:
                        if self._stop.is_set():
                            return
                        self._latest[record["type"]] = {
                            **record,
                            "received_ms": time.monotonic_ns() // 1000000,
                        }
                        self._connected = True
                        self._connection_state = "connected"
                    deadline = None
                    self._next_retry_at = 0.0
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                with self._lock:
                    self._connected = False
                    self._connection_state = "disconnected"
                if not self._reconnect_seconds:
                    return
                if deadline is None:
                    deadline = time.monotonic() + self._reconnect_seconds
                if not self._recover(deadline):
                    return

    def snapshot(self, now_ms: int) -> dict:
        with self._lock:
            latest = copy.deepcopy(self._latest)
            online = self._connected and not self._stop.is_set() and self._thread.is_alive()
            state = self._connection_state
            attempts = self._reconnect_attempts
        for record in latest.values():
            record["ageMs"] = max(0, now_ms - record.pop("received_ms"))
            record["fresh"] = online and record["ageMs"] <= self.stale_ms
        return {"online": online, "staleMs": self.stale_ms, "connectionState": state,
                "reconnectAttempts": attempts, "transport": "uart_adapter_read_only",
                "dtrRequested": False, "rtsRequested": False, "applicationBytesTransmitted": 0, **latest}

    def close(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        finally:
            with self._lock:
                link, self._serial = self._serial, None
                self._connected = False
                self._connection_state = "closed"
            if link is not None:
                link.close()
