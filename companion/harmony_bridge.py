"""Opt-in TLS transport for the native HarmonyOS console; no new motion policy."""
import ipaddress
import json
import re
import secrets
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .gimbal_console import milliseconds
from .telemetry import unique_object


def local_ipv4(value):
    address = ipaddress.IPv4Address(value)
    allowed = (ipaddress.IPv4Network('10.0.0.0/8'), ipaddress.IPv4Network('172.16.0.0/12'),
               ipaddress.IPv4Network('192.168.0.0/16'), ipaddress.IPv4Network('127.0.0.0/8'))
    if not any(address in network for network in allowed):
        raise ValueError('Choose an explicit private LAN IPv4 or loopback address')
    return str(address)


class Pairing:
    def __init__(self, now):
        self.code = f'{secrets.randbelow(100000000):08d}'
        self.deadline = now + 300000
        self.attempts = 0
        self.token = None
        self.expires = 0
        self.client = 'harmony-' + secrets.token_hex(16)
        self.sequence = 0
        self.lock = threading.RLock()

    def pair(self, code, now):
        with self.lock:
            if self.token or now >= self.deadline or self.attempts >= 10:
                raise PermissionError('Pairing closed; restart the mobile gateway')
            self.attempts += 1
            if not isinstance(code, str) or not re.fullmatch(r'[0-9]{8}', code) or not secrets.compare_digest(code, self.code):
                raise PermissionError('Invalid pairing code')
            self.token = secrets.token_hex(32)
            self.expires = now + 3600000
            return dict(token=self.token, expiresInMs=3600000, protocol='harmony-control-1')

    def authorize(self, header, now):
        if (self.token is None or now >= self.expires or not isinstance(header, str)
                or not header.isascii() or not secrets.compare_digest(header, 'Bearer ' + self.token)):
            raise PermissionError('Pair again on the computer')

    def advance(self, sequence):
        if type(sequence) is not int or not self.sequence < sequence <= 2147483647:
            raise ValueError('Stale or invalid request sequence')
        self.sequence = sequence

    def display(self, now):
        with self.lock:
            remaining = max(0, self.deadline - now)
            opened = self.token is None and remaining > 0 and self.attempts < 10
            return dict(paired=self.token is not None, code=self.code if opened else None,
                        remainingMs=remaining if opened else 0)


class MobileServer:
    def __init__(self, session, camera, host, port, cert, key):
        self.session = session
        self.pairing = Pairing(milliseconds())
        host = local_ipv4(host)
        pairing = self.pairing

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def send(self, code, body, kind='application/json'):
                if not isinstance(body, bytes):
                    body = json.dumps(body, ensure_ascii=False).encode('utf-8')
                self.send_response(code)
                for name, value in {'Content-Type': kind, 'Content-Length': str(len(body)),
                                    'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}.items():
                    self.send_header(name, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionError):
                    pass

            def valid_host(self):
                if (self.headers.get('Host') != f'{host}:{self.server.server_port}'
                        or self.headers.get('Origin') is not None):
                    raise PermissionError('Native paired client required')

            def state(self):
                result = session.snapshot(milliseconds())
                result.pop('token', None)
                result['protocol'] = 'harmony-control-1'
                return result

            def do_GET(self):
                try:
                    self.valid_host()
                    pairing.authorize(self.headers.get('Authorization'), milliseconds())
                    url = urlsplit(self.path)
                    if url.path == '/api/state' and not url.query:
                        return self.send(200, self.state())
                    if url.path == '/frame.jpg':
                        query = parse_qs(url.query)
                        if set(query) != {'frame'} or len(query['frame']) != 1:
                            raise ValueError('Exact frame required')
                        with session.lock:
                            frame = next((item for item in session.frame_history if str(item['id']) == query['frame'][0]), None)
                            # Display may tolerate a short delay; motion still uses the session's 400 ms gate.
                            image = frame['jpeg'] if frame and session.captured is not None and 0 <= milliseconds()-frame['id'] < 2000 else None
                        return self.send(200 if image else 404, image or b'', 'image/jpeg')
                    return self.send(404, {'error': 'Not found'})
                except PermissionError as error:
                    return self.send(401, {'error': str(error)})
                except ValueError as error:
                    return self.send(400, {'error': str(error)})

            def do_POST(self):
                try:
                    self.valid_host()
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 1 <= size <= 2048 or self.headers.get('Content-Type') != 'application/json' or self.headers.get('Transfer-Encoding'):
                        raise ValueError('Invalid body')
                    data = json.loads(self.rfile.read(size), object_pairs_hook=unique_object)
                    if not isinstance(data, dict):
                        raise ValueError('Object required')
                    with pairing.lock:
                        if self.path == '/api/pair':
                            if set(data) != {'code'}:
                                raise ValueError('Invalid pairing fields')
                            return self.send(200, pairing.pair(data['code'], milliseconds()))
                        pairing.authorize(self.headers.get('Authorization'), milliseconds())
                        if 'sequence' not in data:
                            raise ValueError('Sequence required')
                        pairing.advance(data.pop('sequence'))
                        with session.lock:
                            if self.path == '/api/control':
                                if 'client' in data:
                                    raise ValueError('Client identity is server-assigned')
                                session.command(dict(data, client=pairing.client), milliseconds())
                            elif self.path == '/api/suspend':
                                if data:
                                    raise ValueError('Invalid suspend fields')
                                if session.owner == pairing.client:
                                    session.stop('手机已离开控制页，停止本次动作')
                            elif self.path == '/api/camera':
                                if camera is None:
                                    raise ValueError('Camera unavailable')
                                if session.mode != 'idle':
                                    raise ValueError('Stop the current action before changing camera')
                                camera.configure(data)
                            else:
                                return self.send(404, {'error': 'Not found'})
                        return self.send(200, self.state())
                except PermissionError as error:
                    return self.send(401, {'error': str(error)})
                except (ValueError, OSError, UnicodeError, RecursionError) as error:
                    return self.send(400, {'error': str(error)})

            def log_message(self, *args):
                pass

        # Bound TLS handshakes so an incomplete connection cannot block accept forever.
        class Server(ThreadingHTTPServer):
            daemon_threads = True
            def get_request(self):
                sock, address = super().get_request()
                sock.settimeout(2)
                try:
                    return context.wrap_socket(sock, server_side=True), address
                except Exception:
                    sock.close()
                    raise

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        self.server = Server((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f'https://{host}:{self.server.server_port}'

    def start(self):
        self.thread.start()

    def close(self):
        with self.session.lock:
            if self.session.owner == self.pairing.client:
                self.session.stop('手机网关关闭')
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
