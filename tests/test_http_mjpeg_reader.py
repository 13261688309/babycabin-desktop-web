from contextlib import contextmanager, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image
from python_multipart.exceptions import MultipartParseError

from companion.capture import CapturedFrame, LiveFrameReader
from companion.http_mjpeg import HttpMjpegCapture, MAX_FRAME_BYTES
from companion.runtime import default_weights, parse_args


TYPE = 'multipart/x-mixed-replace; boundary="frame"'


def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (64, 48), (80, 120, 160)).save(stream, format='JPEG')
    return stream.getvalue()


def part(data=None, *, length=None, extra=b''):
    data = jpeg() if data is None else data
    size = len(data) if length is None else length
    return (b'\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
            + str(size).encode() + b'\r\n' + extra + b'\r\n' + data)


@contextmanager
def server(body=None, *, silent=False, drip=False, repeat=False, response_type=TYPE, status=200):
    body = part() + (b'' if repeat else b'\r\n--frame--\r\n') if body is None else body
    stop = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def handle(self):
            try:
                super().handle()
            except ConnectionError:
                pass

        def do_GET(self):
            self.close_connection = True
            requests.append(self.path)
            self.send_response(status)
            self.send_header('Content-Type', response_type)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            self.wfile.flush()
            if silent:
                stop.wait(timeout=3)
                return
            if drip:
                chunks = [part(b'\xff\xd8', length=100000)] + [b'x'] * 100
            else:
                chunks = [body[index:index + 37] for index in range(0, len(body), 37)]
            while True:
                for chunk in chunks:
                    if stop.is_set():
                        return
                    self.wfile.write(f'{len(chunk):x}\r\n'.encode() + chunk + b'\r\n')
                    self.wfile.flush()
                    if drip and stop.wait(0.02):
                        return
                if not repeat or stop.wait(0.05):
                    break
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    http.daemon_threads = True
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}/stream', requests, body
    finally:
        stop.set()
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)


class MjpegParserTests(unittest.TestCase):
    def capture(self):
        capture = HttpMjpegCapture('http://camera.invalid/stream')
        self.addCleanup(capture.release)
        capture._start_parser(TYPE)
        return capture

    def test_fragmented_headers_and_binary_jpeg_use_streaming_parser(self):
        capture = self.capture()
        encoded = part() + b'\r\n--frame--\r\n'
        for index in range(0, len(encoded), 7):
            capture._feed(encoded[index:index + 7], 1000 + index)
        self.assertEqual(capture._pending[0], jpeg())
        self.assertTrue(capture._ended)
        self.assertLess(capture._pending[1], 1100)

    def test_late_boundary_does_not_refresh_the_older_frames_timestamp(self):
        capture = self.capture()
        capture._feed(part(), 1000)
        capture._feed(b'\r\n--frame\r\n', 5000)
        self.assertEqual(capture._pending, (jpeg(), 1000))

    def test_many_parts_keep_only_latest_encoded_frame(self):
        capture = self.capture()
        first = jpeg()
        capture._feed(part(first), 1000)
        capture._feed(part(first) + b'\r\n--frame--\r\n', 2000)
        self.assertEqual(capture._pending, (first, 2000))
        self.assertEqual(len(capture._body), 0)

    def test_incomplete_or_overlong_frames_are_rejected(self):
        for payload in [part(length=len(jpeg()) + 1), part(length=len(jpeg()) - 1),
                        part(b'not-jpeg'), part(length=MAX_FRAME_BYTES + 1)]:
            with self.subTest(length=len(payload)):
                capture = self.capture()
                with self.assertRaises((ValueError, MultipartParseError)):
                    capture._feed(payload + b'\r\n--frame--\r\n', 1000)
                self.assertIsNone(capture._pending)

    def test_duplicate_headers_and_nonbinary_encoding_are_rejected(self):
        for extra in [b'Content-Length: 100\r\n', b'Content-Transfer-Encoding: base64\r\n']:
            with self.subTest(extra=extra):
                capture = self.capture()
                with self.assertRaises((ValueError, MultipartParseError)):
                    capture._feed(part(extra=extra) + b'\r\n--frame--\r\n', 1000)

    def test_header_count_and_header_size_are_bounded(self):
        for extra in [b'X-Long: ' + b'x' * 1100 + b'\r\n',
                      b''.join(f'X-{index}: value\r\n'.encode() for index in range(9))]:
            with self.subTest(length=len(extra)):
                capture = self.capture()
                with self.assertRaises((ValueError, MultipartParseError)):
                    capture._feed(part(extra=extra), 1000)

    def test_invalid_top_level_types_or_boundaries_are_rejected(self):
        capture = self.capture()
        for value in [None, 'image/jpeg', 'multipart/x-mixed-replace',
                      'multipart/x-mixed-replace; boundary=' + 'x' * 71, TYPE + '\r\nInjected: yes']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                capture._start_parser(value)

    def test_construction_does_not_open_a_network_connection(self):
        with patch('companion.http_mjpeg.http.client.HTTPConnection') as factory:
            HttpMjpegCapture('http://camera.invalid/stream')
            factory.assert_not_called()

    def test_source_and_timeout_arguments_are_validated_before_network_access(self):
        for source in ['rtsp://camera.invalid/stream', 'http://user:password@camera.invalid/stream',
                       'http://camera.invalid/stream#fragment', 'http:///stream']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                HttpMjpegCapture(source)
        for value in [0, 249, 10001, True, float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                HttpMjpegCapture('http://camera.invalid', read_timeout_ms=value)

    def test_runtime_selects_reader_only_for_explicit_http_video(self):
        self.assertEqual(parse_args(['--source', '0']).network_reader, 'ffmpeg')
        self.assertEqual(parse_args(['--source', 'http://camera.invalid/stream',
            '--network-reader', 'http-mjpeg']).network_reader, 'http-mjpeg')
        for args in [['--source', '0'], ['--source', 'clip.mp4'],
                     ['--source', 'rtsp://camera.invalid/live'],
                     ['--source', 'http://user:password@camera.invalid/stream'],
                     ['--telemetry-only', '--serial-port', 'TEST_ONLY']]:
            with self.subTest(args=args), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parse_args(args + ['--network-reader', 'http-mjpeg'])

    def test_live_slot_preserves_a_backend_supplied_capture_timestamp(self):
        items = queue.Queue()
        backend = SimpleNamespace(read=lambda: items.get(timeout=2), release=Mock())
        reader = LiveFrameReader(backend)
        try:
            items.put((True, CapturedFrame('synthetic-frame', 123)))
            frame = reader.next_frame(timeout=1)
            self.assertEqual(frame.captured_ms, 123)
            self.assertEqual(frame.image, 'synthetic-frame')
        finally:
            items.put((False, None))
            reader.release()
        backend.release.assert_called_once()


class MjpegHttpTests(unittest.TestCase):
    def test_real_model_runtime_on_new_http_reader_uses_only_synthetic_localhost(self):
        root = Path(__file__).resolve().parents[1]
        weights = root / 'models/baby-state.pt'
        if not weights.is_file():
            weights = default_weights()
        if not weights.is_file() or not importlib.util.find_spec('ultralytics'):
            self.skipTest('requires trusted existing weights and optional local model runtime')
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
        with server(repeat=True) as (url, _, _):
            result = subprocess.run([sys.executable, '-m', 'companion', '--source', url,
                '--network-reader', 'http-mjpeg', '--weights', str(weights), '--device', 'cpu',
                '--max-frames', '3', '--max-seconds', '20', '--retries', '0'],
                cwd=root, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        frames = [row for row in rows if row['inferenceMs'] is not None]
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(row['video']['networkReader'] == 'http-mjpeg' for row in rows))
        self.assertTrue(all(row['video']['reconnectAttempts'] == 0 for row in rows))
        self.assertTrue(all(row['actuatorControlEnabled'] is False and row['device'] is None for row in rows))
        self.assertTrue(all(row['frameAgeMs'] >= 0 for row in frames))
        self.assertEqual(rows[-1]['terminationReason'], 'frame_limit')

    def test_chunked_http_decodes_and_closes_without_model_or_serial(self):
        with server() as (url, requests, _):
            capture = HttpMjpegCapture(url)
            try:
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.image.shape, (48, 64, 3))
                self.assertLessEqual(frame.captured_ms, time.monotonic_ns() // 1000000)
                self.assertEqual(capture.read(), (False, None))
            finally:
                capture.release()
                capture.release()
            self.assertEqual(requests, ['/stream'])
            self.assertTrue(capture._response.closed)
            self.assertIsNone(capture._connection.sock)

    def test_same_encoded_jpeg_is_decoded_by_http_and_ffmpeg(self):
        with server() as (url, requests, _):
            capture = HttpMjpegCapture(url)
            ffmpeg = None
            try:
                ok, direct = capture.read()
                self.assertTrue(ok)
                capture.release()
                ffmpeg = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                ok, decoded = ffmpeg.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape, direct.image.shape)
                self.assertLessEqual(int(np.abs(decoded.astype(int) - direct.image.astype(int)).max()), 3)
                self.assertEqual(requests, ['/stream', '/stream'])
            finally:
                capture.release()
                if ffmpeg is not None:
                    ffmpeg.release()

    def test_partial_jpeg_is_not_accepted_on_eof(self):
        with server(part(b'\xff\xd8partial', length=100)) as (url, _, _):
            capture = HttpMjpegCapture(url)
            try:
                self.assertEqual(capture.read(), (False, None))
            finally:
                capture.release()

    def test_dripping_bytes_without_complete_frame_obey_total_read_deadline(self):
        with server(drip=True) as (url, _, _):
            capture = HttpMjpegCapture(url, read_timeout_ms=250)
            started = time.monotonic()
            try:
                with self.assertRaises(TimeoutError):
                    capture.read()
                self.assertLess(time.monotonic() - started, 1)
            finally:
                capture.release()

    def test_silent_http_read_closes_the_worker_within_declared_budget(self):
        with server(silent=True) as (url, _, _):
            backend = HttpMjpegCapture(url, read_timeout_ms=250, open_timeout_ms=250)
            reader = LiveFrameReader(backend, shutdown_timeout=backend.shutdown_timeout)
            started = time.monotonic()
            reader.release()
            self.assertLess(time.monotonic() - started, 2)
            self.assertTrue(reader.failed)
            self.assertTrue(backend._closed)

    def test_redirect_and_non_mjpeg_reply_are_not_followed(self):
        for options in [dict(status=302), dict(response_type='text/html')]:
            with self.subTest(options=options), server(**options) as (url, requests, _):
                capture = HttpMjpegCapture(url)
                try:
                    with self.assertRaises(ValueError):
                        capture.read()
                    self.assertEqual(requests, ['/stream'])
                finally:
                    capture.release()


if __name__ == '__main__':
    unittest.main()
