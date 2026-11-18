"""Single-JPEG fallback tests use synthetic localhost images, never devices."""

from contextlib import contextmanager, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from companion.capture import LiveFrameReader
from companion.http_mjpeg import MAX_FRAME_BYTES
from companion.http_snapshot import HttpSnapshotCapture
from companion.runtime import default_weights, parse_args


def jpeg():
    output = io.BytesIO()
    Image.new('RGB', (64, 48), (80, 120, 160)).save(output, format='JPEG')
    return output.getvalue()


@contextmanager
def snapshot_server(*, body=None, content_type='image/jpeg', status=200,
                    length='auto', chunked=False, silent=False, drip=False, delay=0,
                    stream_failure=False):
    body = jpeg() if body is None else body
    stop = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self):
            try:
                self.close_connection = True
                requests.append((self.path, dict(self.headers)))
                if stream_failure and self.path == '/stream':
                    self.send_response(503)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if delay and stop.wait(delay):
                    return
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                if chunked:
                    self.send_header('Transfer-Encoding', 'chunked')
                elif length is not None:
                    self.send_header('Content-Length', str(len(body) if length == 'auto' else length))
                self.end_headers()
                self.wfile.flush()
                if silent:
                    stop.wait(3)
                    return
                for index in range(0, len(body), 1 if drip else 127):
                    chunk = body[index:index + (1 if drip else 127)]
                    self.wfile.write((f'{len(chunk):x}\r\n'.encode() + chunk + b'\r\n') if chunked else chunk)
                    self.wfile.flush()
                    if drip and stop.wait(.02):
                        return
                if chunked:
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
            except (ConnectionError, OSError):
                pass

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    http.daemon_threads = True
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}/capture', requests
    finally:
        stop.set()
        http.shutdown()
        http.server_close()
        thread.join(2)


class SnapshotReaderTests(unittest.TestCase):
    def test_repeated_requests_decode_real_jpeg_and_disable_caching(self):
        with snapshot_server() as (url, requests):
            capture = HttpSnapshotCapture(url)
            try:
                frames = [capture.read()[1], capture.read()[1]]
                self.assertTrue(all(frame.image.shape == (48, 64, 3) for frame in frames))
                self.assertGreaterEqual(frames[1].captured_ms - frames[0].captured_ms, 190)
            finally:
                capture.release()
            self.assertEqual([item[0] for item in requests], ['/capture', '/capture'])
            self.assertTrue(all(item[1]['Connection'] == 'close' for item in requests))
            self.assertTrue(all('no-cache' in item[1]['Cache-Control'] for item in requests))

    def test_chunked_and_close_delimited_jpeg_bodies(self):
        for options in [dict(chunked=True), dict(length=None)]:
            with self.subTest(options=options), snapshot_server(**options) as (url, _):
                capture = HttpSnapshotCapture(url)
                try:
                    self.assertEqual(capture.read()[1].image.shape, (48, 64, 3))
                finally:
                    capture.release()

    def test_timestamp_precedes_response_latency(self):
        with snapshot_server(delay=.15) as (url, _):
            capture = HttpSnapshotCapture(url)
            try:
                frame = capture.read()[1]
                self.assertGreaterEqual(time.monotonic_ns() // 1_000_000 - frame.captured_ms, 140)
            finally:
                capture.release()

    def test_partial_body_and_wrong_length_are_not_returned(self):
        for options in [dict(body=b'\xff\xd8partial'), dict(length=len(jpeg()) + 1)]:
            with self.subTest(options=options), snapshot_server(**options) as (url, _):
                capture = HttpSnapshotCapture(url)
                try:
                    with self.assertRaises((ValueError, OSError)):
                        capture.read()
                finally:
                    capture.release()

    def test_redirect_non_jpeg_and_oversized_length_are_rejected(self):
        for options in [dict(status=302), dict(content_type='text/html'),
                        dict(length=MAX_FRAME_BYTES + 1), dict(length='invalid')]:
            with self.subTest(options=options), snapshot_server(**options) as (url, requests):
                capture = HttpSnapshotCapture(url)
                try:
                    with self.assertRaises(ValueError):
                        capture.read()
                    self.assertEqual(len(requests), 1)
                finally:
                    capture.release()

    def test_dripping_bytes_obey_total_body_deadline(self):
        with snapshot_server(drip=True) as (url, _):
            capture = HttpSnapshotCapture(url, read_timeout_ms=250)
            started = time.monotonic()
            try:
                with self.assertRaises(TimeoutError):
                    capture.read()
                self.assertLess(time.monotonic() - started, 1)
            finally:
                capture.release()

    def test_silent_request_worker_stops_within_budget(self):
        with snapshot_server(silent=True) as (url, _):
            capture = HttpSnapshotCapture(url, read_timeout_ms=250, open_timeout_ms=250)
            reader = LiveFrameReader(capture, shutdown_timeout=capture.shutdown_timeout)
            started = time.monotonic()
            reader.release()
            self.assertLess(time.monotonic() - started, 2)
            self.assertTrue(reader.failed)

    def test_released_capture_never_opens_another_connection(self):
        with patch('companion.http_snapshot.http.client.HTTPConnection') as factory:
            capture = HttpSnapshotCapture('http://camera.invalid/capture')
            factory.assert_not_called()
            capture.release()
            capture.release()
            self.assertEqual(capture.read(), (False, None))
            factory.assert_not_called()

    def test_bad_urls_timeouts_and_intervals_fail_without_network(self):
        for url in ['http:///capture', 'rtsp://camera.invalid/capture', 'http://u:p@camera.invalid/capture',
                    'http://camera.invalid/capture#fragment', 'http://camera.invalid:99999/capture',
                    'http://camera.invalid/cap\nture']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                HttpSnapshotCapture(url)
        for kwargs in [dict(read_timeout_ms=0), dict(open_timeout_ms=True), dict(interval_ms=199),
                       dict(interval_ms=float('inf'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HttpSnapshotCapture('http://camera.invalid/capture', **kwargs)

    def test_fallback_is_opt_in_and_same_host_only(self):
        source = ['--source', 'http://camera.invalid:81/stream', '--network-reader', 'http-mjpeg']
        self.assertIsNone(parse_args(source).snapshot_fallback_url)
        self.assertEqual(parse_args(['--source', 'http://camera.invalid/capture',
            '--network-reader', 'http-snapshot']).network_reader, 'http-snapshot')
        self.assertEqual(parse_args(source + ['--snapshot-fallback-url', 'http://camera.invalid/capture'])
                         .snapshot_fallback_url, 'http://camera.invalid/capture')
        for url in ['http://other.invalid/capture', 'https://camera.invalid/capture',
                    'http://camera.invalid:81/stream', 'http://u:p@camera.invalid/capture']:
            with self.subTest(url=url), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parse_args(source + ['--snapshot-fallback-url', url])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_args(['--source', 'http://camera.invalid/stream',
                        '--snapshot-fallback-url', 'http://camera.invalid/capture'])

    def test_real_model_runs_after_http_failure_using_only_synthetic_localhost(self):
        root = Path(__file__).resolve().parents[1]
        weights = root / 'models/baby-state.pt'
        if not weights.is_file():
            weights = default_weights()
        if not weights.is_file() or not importlib.util.find_spec('ultralytics'):
            self.skipTest('requires trusted existing weights and optional local model runtime')
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
        with snapshot_server(stream_failure=True) as (url, requests):
            result = subprocess.run([sys.executable, '-m', 'companion',
                '--source', url.replace('/capture', '/stream'), '--network-reader', 'http-mjpeg',
                '--snapshot-fallback-url', url, '--weights', str(weights), '--device', 'cpu',
                '--max-frames', '3', '--max-seconds', '20', '--retries', '1'],
                cwd=root, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        frames = [row for row in rows if row['inferenceMs'] is not None]
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(row['video']['networkReader'] == 'http-snapshot' for row in frames))
        self.assertTrue(all(row['video']['frameTimestampBasis'] == 'snapshot_request_started' for row in frames))
        self.assertTrue(all(row['device'] is None and row['actuatorControlEnabled'] is False for row in rows))
        self.assertTrue(all(0 <= row['frameAgeMs'] <= 1500 for row in frames))
        self.assertEqual(requests[0][0], '/stream')
        self.assertTrue(all(item[0] == '/capture' for item in requests[1:]))
        self.assertEqual(rows[-1]['terminationReason'], 'frame_limit')


if __name__ == '__main__':
    unittest.main()
