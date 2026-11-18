"""One upstream camera connection, bounded fallback, and measured frame rates."""
from collections import deque
import threading
import time
from urllib.parse import urlsplit

from .http_mjpeg import HttpMjpegCapture
from .http_snapshot import HttpSnapshotCapture

STREAM_RECOVERY_FRAMES = 3
QUICK_STREAM_RETRY_SECONDS = 3
FAILED_STREAM_RETRY_SECONDS = 30


class FrameRate:
    def __init__(self):
        self.samples = deque(maxlen=180)

    def add(self, now):
        self.samples.append(now)

    def value(self, now):
        while self.samples and now - self.samples[0] > 2000:
            self.samples.popleft()
        if len(self.samples) < 2 or now - self.samples[-1] >= 400:
            return 0.0
        elapsed = self.samples[-1] - self.samples[0]
        return round((len(self.samples) - 1) * 1000 / elapsed, 1) if elapsed > 0 else 0.0


class CameraCapture:
    """An explicit /capture stays compatible; /stream falls back then retries."""
    def __init__(self, source, *, read_timeout_ms=1000, open_timeout_ms=1000,
                 stream_factory=HttpMjpegCapture, snapshot_factory=HttpSnapshotCapture,
                 clock=time.monotonic):
        self.source = source
        url = urlsplit(source)
        port = None if url.port == 81 else url.port
        self.snapshot_source = f'http://{url.hostname}' + (f':{port}' if port else '') + '/capture'
        self.factories = {'mjpeg': stream_factory, 'snapshot': snapshot_factory}
        self.options = dict(read_timeout_ms=read_timeout_ms, open_timeout_ms=open_timeout_ms)
        self.preferred = 'mjpeg' if url.path == '/stream' else 'snapshot'
        self.transport = self.preferred
        self.fallback_reason = None
        self.retry_at = 0.0
        self.stream_frames = 0
        self.clock = clock
        self.backend = None
        self.lock = threading.Lock()
        self.stopped = False

    def read(self):
        with self.lock:
            if self.stopped:
                return False, None
            if self.transport == 'snapshot' and self.preferred == 'mjpeg' and self.clock() >= self.retry_at:
                if self.backend:
                    self.backend.release()
                self.backend = None
                self.transport = 'mjpeg'
            if self.backend is None:
                source = self.source if self.transport == self.preferred else self.snapshot_source
                self.backend = self.factories[self.transport](source, **self.options)
                self.stream_frames = 0
            backend = self.backend
        try:
            ok, frame = backend.read()
            if not ok:
                raise OSError('Camera stream ended')
            if self.transport == 'mjpeg':
                self.stream_frames = min(STREAM_RECOVERY_FRAMES, self.stream_frames + 1)
                self.fallback_reason = None
            return ok, frame
        except Exception as error:
            with self.lock:
                backend.release()
                self.backend = None
                if self.transport == 'mjpeg':
                    self.transport = 'snapshot'
                    delay = QUICK_STREAM_RETRY_SECONDS if self.stream_frames >= STREAM_RECOVERY_FRAMES else FAILED_STREAM_RETRY_SECONDS
                    self.retry_at = self.clock() + delay
                    self.fallback_reason = type(error).__name__
            # The caller must invalidate tracking before accepting fallback frames.
            raise

    def release(self):
        with self.lock:
            self.stopped = True
            backend = self.backend
        if backend:
            backend.release()
