import http.client
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import unittest

from companion.gimbal_console import ConsoleServer, GimbalSession, milliseconds
from companion.harmony_bridge import MobileServer, Pairing, local_ipv4


class HarmonyBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='harmony-tls-test-')
        folder = Path(cls.temp.name)
        cls.cert, cls.key = folder/'cert.pem', folder/'key.pem'
        (folder/'openssl.cnf').write_text('[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=localhost\n')
        openssl = shutil.which('openssl')
        if not openssl:
            raise unittest.SkipTest('OpenSSL required for real TLS transport tests')
        subprocess.run([openssl, 'req', '-config', 'openssl.cnf', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', 'key.pem', '-out', 'cert.pem', '-days', '1', '-subj', '/CN=localhost',
                        '-addext', 'subjectAltName=IP:127.0.0.1', '-addext', 'basicConstraints=critical,CA:TRUE'],
                       cwd=folder, capture_output=True, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.session = GimbalSession()
        self.server = MobileServer(self.session, None, '127.0.0.1', 0, self.cert, self.key)
        self.server.start()
        self.token = None
        self.sequence = 0

    def tearDown(self):
        self.server.close()

    def request(self, path, body=None, headers=None, raw=False):
        connection = http.client.HTTPSConnection('127.0.0.1', self.server.server.server_port,
                                                 context=ssl.create_default_context(cafile=str(self.cert)), timeout=3)
        options = {'Content-Type': 'application/json'}
        if self.token:
            options['Authorization'] = 'Bearer '+self.token
        options.update(headers or {})
        connection.request('POST' if body is not None else 'GET', path,
                           body if raw else json.dumps(body) if body is not None else None, options)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, json.loads(data) if response.getheader('Content-Type') == 'application/json' else data

    def pair(self):
        status, data = self.request('/api/pair', {'code': self.server.pairing.code})
        self.assertEqual(status, 200)
        self.token = data['token']

    def command(self, action, **extra):
        self.sequence += 1
        return self.request('/api/control', dict(action=action, generation=self.session.generation,
                                                sequence=self.sequence, **extra))

    def test_private_bind_and_pairing_window(self):
        for host in ('0.0.0.0', '8.8.8.8', '224.0.0.1', 'localhost', '::1'):
            with self.assertRaises(ValueError): local_ipv4(host)
        for host in ('127.0.0.1', '192.168.43.4', '10.0.0.2', '172.16.1.2'):
            self.assertEqual(local_ipv4(host), host)
        pairing = Pairing(100)
        with self.assertRaises(PermissionError): pairing.pair(pairing.code, 300100)
        pairing = Pairing(100)
        for _ in range(10):
            with self.assertRaises(PermissionError): pairing.pair('bad', 100)
        with self.assertRaises(PermissionError): pairing.pair(pairing.code, 100)

    def test_pairing_display_hides_closed_codes(self):
        pairing = Pairing(100)
        self.assertEqual(pairing.display(101)['code'], pairing.code)
        self.assertIsNone(pairing.display(300100)['code'])
        pairing.pair(pairing.code, 101)
        self.assertTrue(pairing.display(102)['paired'])
        self.assertIsNone(pairing.display(102)['code'])

    def test_pairing_page_is_loopback_only(self):
        desktop = ConsoleServer(self.session, mobile=self.server)
        desktop.start()
        def get(path, headers=None):
            connection = http.client.HTTPConnection('127.0.0.1', desktop.server.server_port, timeout=3)
            connection.request('GET', path, headers=headers or {})
            response = connection.getresponse()
            body = response.read()
            connection.close()
            return response.status, body
        try:
            self.assertEqual(get('/mobile')[0], 200)
            status, body = get('/api/mobile-pairing')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['code'], self.server.pairing.code)
            self.assertEqual(get('/api/mobile-pairing', {'Host': 'attacker.example'})[0], 403)
            self.assertEqual(get('/api/mobile-pairing', {'Origin': 'https://attacker.example'})[0], 403)
            self.assertEqual(get('/api/mobile-pairing', {'Sec-Fetch-Site': 'cross-site'})[0], 403)
            self.pair()
            self.assertIsNone(json.loads(get('/api/mobile-pairing')[1])['code'])
            self.assertEqual(self.request('/api/mobile-pairing')[0], 404)
        finally:
            desktop.close()

    def test_tls_verification_and_authentication(self):
        with self.assertRaises(ssl.SSLCertVerificationError):
            connection = http.client.HTTPSConnection('127.0.0.1', self.server.server.server_port, timeout=3)
            try: connection.request('GET', '/api/state')
            finally: connection.close()
        self.assertEqual(self.request('/api/state')[0], 401)
        self.assertEqual(self.request('/frame.jpg?frame=1')[0], 401)
        self.pair()
        status, state = self.request('/api/state')
        self.assertEqual(status, 200)
        self.assertNotIn('token', state)
        self.assertFalse(state['hardware'])
        self.assertEqual(state['mode'], 'idle')
        self.assertEqual(self.request('/api/pair', {'code': self.server.pairing.code})[0], 401)

    def test_host_origin_and_duplicate_json_rejected(self):
        self.pair()
        self.assertEqual(self.request('/api/state', headers={'Host': 'attacker.example'})[0], 401)
        self.assertEqual(self.request('/api/state', headers={'Origin': 'https://attacker.example'})[0], 401)
        self.assertEqual(self.request('/api/control', '{"sequence":1,"sequence":2}', raw=True)[0], 400)
        self.assertEqual(self.request('/api/control', '{}', headers={'Content-Type': 'text/plain'}, raw=True)[0], 400)

    def test_sequence_replay_and_server_assigned_owner(self):
        self.pair()
        status, _ = self.command('manual', pan=95, tilt=90)
        self.assertEqual(status, 200)
        self.assertEqual(self.session.owner, self.server.pairing.client)
        self.assertEqual(self.request('/api/control', dict(sequence=self.sequence, action='manual',
                         generation=self.session.generation, pan=100, tilt=90))[0], 400)
        self.assertEqual(self.command('heartbeat', client='browser')[0], 400)
        self.assertEqual(self.session.pan, 95)

    def test_suspend_cancels_only_phone_owner(self):
        self.pair()
        self.command('manual', pan=95, tilt=90)
        self.sequence += 1
        self.assertEqual(self.request('/api/suspend', dict(sequence=self.sequence))[0], 200)
        self.assertEqual(self.session.mode, 'idle')
        self.session.command(dict(action='manual', client='desktop', generation=self.session.generation, pan=100, tilt=90), milliseconds())
        self.sequence += 1
        self.assertEqual(self.request('/api/suspend', dict(sequence=self.sequence))[0], 200)
        self.assertEqual(self.session.owner, 'desktop')
        self.assertEqual(self.command('stop')[0], 200)
        self.assertEqual(self.session.mode, 'idle')

    def test_heartbeat_owner_and_disconnect_deadline(self):
        self.pair()
        self.command('manual', pan=95, tilt=90)
        self.assertEqual(self.command('heartbeat')[0], 200)
        last = self.session.heartbeat
        self.session.tick(last+600)
        self.assertEqual(self.session.mode, 'idle')
        self.command('heartbeat')
        self.assertEqual(self.session.mode, 'idle')

    def test_limits_generations_and_presets(self):
        self.pair()
        self.assertEqual(self.command('manual', pan=180, tilt=90)[0], 400)
        self.assertEqual(self.command('preset_save', slot='1', name='Desk', pan=100, tilt=90, panSign=1, tiltSign=-1)[0], 200)
        self.assertEqual(self.session.mode, 'idle')
        self.assertEqual(self.command('preset_apply', slot='1')[0], 200)
        self.assertEqual((self.session.pan, self.session.tilt), (100, 90))
        self.command('stop')
        self.sequence += 1
        self.assertEqual(self.request('/api/control', dict(sequence=self.sequence, generation=0, action='manual', pan=90, tilt=90))[0], 400)

    def test_token_expiry_does_not_renew_motion(self):
        self.pair()
        self.command('rock', speed=10, duration=2000)
        last = self.session.heartbeat
        self.server.pairing.expires = milliseconds()-1
        self.assertEqual(self.command('heartbeat')[0], 401)
        self.session.tick(last+600)
        self.assertEqual(self.session.mode, 'idle')

    def test_frames_match_authenticated_requested_frame(self):
        self.pair()
        now = milliseconds()
        self.session.frame([(1, 1, 20, 20)], 100, 100, now, now, b'frame-A')
        self.session.frame([(2, 1, 20, 20)], 100, 100, now+1, now+1, b'frame-B')
        self.assertEqual(self.request('/frame.jpg?frame='+str(now)), (200, b'frame-A'))
        self.assertEqual(self.request('/frame.jpg')[0], 400)
        self.assertEqual(self.request('/frame.jpg?frame=0')[0], 404)
        self.assertEqual(self.command('select_target', frameId=now, index=0)[0], 200)
        self.assertEqual(self.session.mode, 'idle')

    def test_no_hardware_or_camera_mode_escalation(self):
        self.pair()
        self.assertEqual(self.command('enable-hardware')[0], 400)
        self.assertEqual(self.request('/api/camera', dict(sequence=99, action='connect', generation=0, source='http://127.0.0.1/capture'))[0], 400)
        self.assertFalse(self.session.hardware)

    def test_local_telemetry_release_preserves_pairing_and_frame(self):
        class Observer:
            connected=True
            def snapshot(self,now): return dict(connected=self.connected,readOnly=True)
            def close(self): self.connected=False
        observer=Observer()
        self.session.observer=observer
        self.pair()
        pairing=self.server.pairing
        token,expires=pairing.token,pairing.expires
        now=milliseconds()
        self.session.frame([(1,1,20,20)],100,100,now,now,b'frame-kept')
        self.assertEqual(self.request('/api/controller',dict(sequence=1,action='disconnect',generation=0))[0],404)
        self.assertTrue(observer.connected)
        desktop=ConsoleServer(self.session,mobile=self.server)
        desktop.start()
        try:
            port=desktop.server.server_port
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=3)
            connection.request('POST','/api/controller',json.dumps(dict(action='disconnect',generation=0)),
                               {'Origin':f'http://127.0.0.1:{port}','X-Control-Token':self.session.token,
                                'Content-Type':'application/json'})
            response=connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status,200)
            self.assertFalse(observer.connected)
            self.assertEqual((pairing.token,pairing.expires),(token,expires))
            self.assertEqual(self.request('/api/state')[0],200)
            self.assertEqual(self.request('/frame.jpg?frame='+str(now)),(200,b'frame-kept'))
        finally:
            desktop.close()

    def test_delayed_display_does_not_relax_motion_freshness(self):
        self.pair()
        now = milliseconds()
        self.session.frame([(1, 1, 20, 20)], 100, 100, now-800, now, b'delayed-preview')
        self.assertEqual(self.request('/frame.jpg?frame='+str(now-800)), (200, b'delayed-preview'))
        self.assertFalse(self.request('/api/state')[1]['frameFresh'])
        self.assertEqual(self.command('select_target', frameId=now-800, index=0)[0], 400)
        self.assertEqual(self.command('auto', panSign=1, tiltSign=1)[0], 200)
        self.session.tick(milliseconds())
        self.assertFalse(self.session.sent)
        self.assertEqual((self.session.pan, self.session.tilt), (90, 90))


if __name__ == '__main__':
    unittest.main()
