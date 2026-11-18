"""Drain live captures into one timestamped slot, never a growing frame queue."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class CapturedFrame:
    image: object
    captured_ms: int
    jpeg: bytes | None = None


class LiveFrameReader:
    def __init__(self, capture, *, shutdown_timeout: float = 2.0) -> None:
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("capture shutdown timeout must be finite and positive")
        self._capture = capture
        self._shutdown_timeout = shutdown_timeout
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: CapturedFrame | None = None
        self._failed = False
        self._thread = threading.Thread(target=self._run, name="live-frame-reader", daemon=True)
        self._thread.start()

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._failed

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, image = self._capture.read()
                captured_ms = time.monotonic_ns() // 1_000_000
                if not ok or image is None:
                    break
                with self._condition:
                    if self._stop.is_set():
                        break
                    self._latest = image if isinstance(image, CapturedFrame) else CapturedFrame(image, captured_ms)
                    self._condition.notify_all()
        except Exception:
            # Backend messages may contain stream credentials; expose only failure.
            pass
        finally:
            try:
                self._capture.release()
            except Exception:
                pass
            finally:
                with self._condition:
                    self._latest = None
                    self._failed = True
                    self._condition.notify_all()

    def next_frame(self, timeout: float = 0.25) -> CapturedFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None or self._failed or self._stop.is_set(), timeout)
            if self._stop.is_set():
                return None
            frame = self._latest
            self._latest = None
            return frame

    def release(self) -> None:
        self._stop.set()
        with self._condition:
            self._latest = None
            self._condition.notify_all()
        request_stop = getattr(self._capture, 'request_stop', None)
        if callable(request_stop):
            request_stop()
        self._thread.join(timeout=self._shutdown_timeout)
        if self._thread.is_alive():
            raise RuntimeError("live capture backend did not stop within its timeout")
