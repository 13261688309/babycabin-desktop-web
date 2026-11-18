"""Bounded HTTP/1.1 MJPEG reader for the factory camera's multipart stream."""

from __future__ import annotations

import http.client
import io
import time
from urllib.parse import urlsplit

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from .capture import CapturedFrame
from .http_deadline import HttpReadDeadline

MAX_FRAME_BYTES = 2 * 1024 * 1024


def decode_jpeg_frame(data):
    from PIL import Image
    import cv2
    import numpy as np

    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if image.format != 'JPEG' or not 0 < width <= 4096 or not 0 < height <= 4096 or width * height > 8_000_000:
            raise ValueError('HTTP camera decoded dimension limit exceeded')
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('HTTP camera decode returned no image')
    return frame


class HttpMjpegCapture:
    def __init__(self, source: str, *, read_timeout_ms: int = 3000, open_timeout_ms: int = 3000):
        url = urlsplit(source)
        if url.scheme.lower() not in ('http', 'https') or not url.hostname or url.username is not None or url.password is not None or url.fragment:
            raise ValueError('HTTP MJPEG requires an HTTP(S) source without userinfo or fragment')
        if any(type(value) is not int or not 250 <= value <= 10000 for value in (read_timeout_ms, open_timeout_ms)):
            raise ValueError('HTTP MJPEG timeouts must be bounded integer milliseconds')
        self._host = url.hostname
        self._port = url.port
        self._https = url.scheme.lower() == 'https'
        self._path = (url.path or '/') + ('?' + url.query if url.query else '')
        self._read_timeout = read_timeout_ms / 1000
        self._open_timeout = open_timeout_ms / 1000
        self._connection = None
        self._response = None
        self._parser = None
        self._closed = False
        self._ended = False
        self._pending = None
        self._chunk_ms = 0
        self._part_ms = 0
        self._headers = {}
        self._field = bytearray()
        self._value = bytearray()
        self._body = bytearray()
        self._expected = None
        self._part_emitted = False
        self._read_deadline = HttpReadDeadline()

    @property
    def shutdown_timeout(self) -> float:
        # Opening includes connect and response headers; body reads have their own deadline.
        return 2 * self._open_timeout + self._read_timeout + 1

    def _part_begin(self):
        self._part_ms = self._chunk_ms
        self._headers = {}
        self._body.clear()
        self._expected = None
        self._part_emitted = False

    def _header_begin(self):
        self._field.clear()
        self._value.clear()

    def _header_field(self, data, start, end):
        self._field.extend(data[start:end])

    def _header_value(self, data, start, end):
        self._value.extend(data[start:end])

    def _header_end(self):
        name = bytes(self._field).lower()
        if name in self._headers:
            raise ValueError('Duplicate MJPEG part header')
        self._headers[name] = bytes(self._value).strip()

    def _headers_finished(self):
        media, _ = parse_options_header(self._headers.get(b'content-type'))
        length = self._headers.get(b'content-length', b'')
        if media.lower() != b'image/jpeg' or not length.isdigit() or len(length) > 10:
            raise ValueError('MJPEG part needs JPEG type and a bounded content length')
        self._expected = int(length)
        if not 4 <= self._expected <= MAX_FRAME_BYTES:
            raise ValueError('MJPEG frame byte limit exceeded')
        if self._headers.get(b'content-transfer-encoding', b'binary').lower() not in (b'binary', b'8bit'):
            raise ValueError('Unsupported MJPEG transfer encoding')

    def _part_data(self, data, start, end):
        if self._expected is None or len(self._body) + end - start > self._expected:
            raise ValueError('MJPEG frame exceeds its declared length')
        self._body.extend(data[start:end])
        if len(self._body) == self._expected:
            # Content-Length closes this JPEG; waiting for the next boundary
            # would add a camera-frame interval without making it any fresher.
            self._validate_part()
            self._pending = (bytes(self._body), self._part_ms)
            self._part_emitted = True

    def _validate_part(self):
        if len(self._body) != self._expected or not self._body.startswith(b'\xff\xd8') or not self._body.endswith(b'\xff\xd9'):
            raise ValueError('Incomplete MJPEG frame')

    def _part_end(self):
        self._validate_part()
        if not self._part_emitted:
            self._pending = (bytes(self._body), self._part_ms)
        self._body.clear()

    def _end(self):
        self._ended = True

    def _start_parser(self, content_type):
        if not isinstance(content_type, str) or len(content_type) > 1024 or any(char in content_type for char in ('\r', '\n', '\x00')):
            raise ValueError('Invalid HTTP MJPEG content type')
        media, params = parse_options_header(content_type)
        boundary = params.get(b'boundary')
        if media.lower() != b'multipart/x-mixed-replace' or not boundary or not 1 <= len(boundary) <= 70 or not boundary.isascii():
            raise ValueError('Missing bounded HTTP MJPEG boundary')
        self._parser = MultipartParser(boundary, {
            'on_part_begin': self._part_begin, 'on_header_begin': self._header_begin,
            'on_header_field': self._header_field, 'on_header_value': self._header_value,
            'on_header_end': self._header_end, 'on_headers_finished': self._headers_finished,
            'on_part_data': self._part_data, 'on_part_end': self._part_end, 'on_end': self._end,
        }, max_header_count=8, max_header_size=1024)

    def _open(self):
        factory = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        self._read_deadline.begin(self._open_timeout)
        self._connection = factory(self._host, self._port, timeout=self._open_timeout)
        self._connection.response_class = self._read_deadline.response
        self._connection.request('GET', self._path, headers={'Accept': 'multipart/x-mixed-replace'})
        self._response = self._connection.getresponse()
        self._read_deadline.remaining()
        if self._response.status != 200 or self._response.version != 11 or self._connection.sock is None:
            raise ValueError('Expected an open HTTP/1.1 MJPEG response; redirects are not followed')
        self._start_parser(self._response.getheader('Content-Type'))

    def _feed(self, data, received_ms):
        self._chunk_ms = received_ms
        try:
            if self._parser.write(data) != len(data):
                raise ValueError('MJPEG parser did not consume the complete input')
        except Exception:
            self._pending = None
            raise

    def read(self):
        if self._closed:
            return False, None
        if self._connection is None:
            self._open()
        self._read_deadline.begin(self._read_timeout)
        while self._pending is None:
            if self._ended:
                return False, None
            self._read_deadline.remaining()
            data = self._response.read1(8192)
            self._read_deadline.remaining()
            if not data:
                return False, None
            self._feed(data, time.monotonic_ns() // 1_000_000)
        data, captured_ms = self._pending
        self._pending = None
        return True, CapturedFrame(decode_jpeg_frame(data), captured_ms, data)

    def request_stop(self):
        self._read_deadline.cancel()

    def release(self):
        self.request_stop()
        if self._closed:
            return
        self._closed = True
        self._pending = None
        self._body.clear()
        try:
            if self._response is not None:
                self._response.close()
        finally:
            if self._connection is not None:
                self._connection.close()
