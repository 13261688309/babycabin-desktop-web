"""Explicit, bounded single-JPEG camera requests; never retain a failed frame."""

from __future__ import annotations

import http.client
import threading
import time
from urllib.parse import urlsplit

from .capture import CapturedFrame
from .http_mjpeg import MAX_FRAME_BYTES, decode_jpeg_frame
from .http_deadline import HttpReadDeadline


def snapshot_url(source):
    url = urlsplit(source)
    if (url.scheme.lower() not in ('http', 'https') or not url.hostname
            or url.username is not None or url.password is not None or url.fragment
            or any(ord(char) <= 32 or ord(char) == 127 for char in source)):
        raise ValueError('Snapshot requires an HTTP(S) URL without userinfo or fragment')
    # Accessing port also rejects malformed or out-of-range values before any I/O.
    _ = url.port
    return url


class HttpSnapshotCapture:
    def __init__(self, source, *, read_timeout_ms=3000, open_timeout_ms=3000, interval_ms=200):
        self._url = snapshot_url(source)
        if any(type(value) is not int or not 250 <= value <= 10000
               for value in (read_timeout_ms, open_timeout_ms)):
            raise ValueError('Snapshot timeouts must be bounded integer milliseconds')
        if type(interval_ms) is not int or not 200 <= interval_ms <= 10000:
            raise ValueError('Snapshot interval must be in 200..10000 milliseconds')
        self._read_timeout = read_timeout_ms / 1000
        self._open_timeout = open_timeout_ms / 1000
        self._interval = interval_ms / 1000
        self._next_request = 0.0
        self._stop = threading.Event()
        self._read_deadline = HttpReadDeadline()

    @property
    def shutdown_timeout(self):
        return self._interval + 2 * self._open_timeout + self._read_timeout + 1

    def read(self):
        if self._stop.wait(max(0, self._next_request - time.monotonic())):
            return False, None
        factory = http.client.HTTPSConnection if self._url.scheme.lower() == 'https' else http.client.HTTPConnection
        connection = factory(self._url.hostname, self._url.port, timeout=self._open_timeout)
        connection.response_class = self._read_deadline.response
        response = None
        # Conservatively include transfer time in frame age; this is not sensor time.
        captured_ms = time.monotonic_ns() // 1_000_000
        try:
            self._read_deadline.begin(self._open_timeout)
            connection.connect()
            path = (self._url.path or '/') + ('?' + self._url.query if self._url.query else '')
            connection.request('GET', path, headers={
                'Accept': 'image/jpeg', 'Cache-Control': 'no-cache, no-store',
                'Pragma': 'no-cache', 'Connection': 'close',
            })
            response = connection.getresponse()
            self._read_deadline.remaining()
            if response.status != 200 or response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'image/jpeg':
                raise ValueError('Expected a JPEG response; redirects are not followed')
            if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('Encoded snapshot bodies are not supported')
            lengths = response.headers.get_all('Content-Length', [])
            if lengths and response.getheader('Transfer-Encoding') is not None:
                raise ValueError('Ambiguous snapshot transfer framing')
            expected = None
            if lengths:
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 10:
                    raise ValueError('Invalid snapshot content length')
                expected = int(lengths[0])
                if not 4 <= expected <= MAX_FRAME_BYTES:
                    raise ValueError('Snapshot content length exceeds limits')
            body = bytearray()
            self._read_deadline.begin(self._read_timeout)
            while not self._stop.is_set():
                self._read_deadline.remaining()
                chunk = response.read1(min(8192, MAX_FRAME_BYTES + 1 - len(body)))
                self._read_deadline.remaining()
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > MAX_FRAME_BYTES:
                    raise ValueError('Snapshot byte limit exceeded')
            if self._stop.is_set():
                return False, None
            if ((expected is not None and len(body) != expected) or not body.startswith(b'\xff\xd8')
                    or not body.endswith(b'\xff\xd9')):
                raise ValueError('Incomplete snapshot JPEG')
            encoded = bytes(body)
            return True, CapturedFrame(decode_jpeg_frame(encoded), captured_ms, encoded)
        finally:
            self._next_request = time.monotonic() + self._interval
            try:
                if response is not None:
                    response.close()
            finally:
                connection.close()

    def request_stop(self):
        self._stop.set()
        self._read_deadline.cancel()

    def release(self):
        self.request_stop()
