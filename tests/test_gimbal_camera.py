import http.client
import json
import queue
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from companion.gimbal_camera import GimbalCamera,camera_source
from companion.gimbal_console import GimbalSession,ConsoleServer,milliseconds


class Capture:
    def __init__(self,source,**kwargs):
        self.source=source;self.queue=queue.Queue();self.released=False
    def read(self):
        try: value=self.queue.get(timeout=.05)
        except queue.Empty: return False,None
        if isinstance(value,Exception): raise value
        return True,SimpleNamespace(image=value,captured_ms=milliseconds())
    def release(self): self.released=True


class Detector:
    def detect(self,frame): return [(25,25,75,75)]


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.session=GimbalSession()
        self.camera=GimbalCamera(self.session,capture_factory=Capture,detector_factory=Detector)
    def tearDown(self): self.camera.close()
    def configure(self,source=None):
        data=dict(action='connect' if source else 'disconnect',generation=self.session.generation)
        if source: data['source']=source
        self.camera.configure(data)
    def wait(self,predicate):
        deadline=time.monotonic()+2
        while not predicate() and time.monotonic()<deadline: time.sleep(.01)
        self.assertTrue(predicate())

    def test_address_validation_without_network(self):
        self.assertEqual(camera_source('192.168.43.24'),'http://192.168.43.24:81/stream')
        self.assertEqual(camera_source('http://127.0.0.1:1234/stream'),'http://127.0.0.1:1234/stream')
        self.assertEqual(camera_source('http://127.0.0.1:1234/capture'),'http://127.0.0.1:1234/capture')
        for value in ('http://example.com/capture','http://8.8.8.8/capture','file:///x','http://192.168.1.2/control',
                      'http://u:p@192.168.1.2/capture','http://192.168.1.2:0/capture','192.168.1.2?x=1',
                      'http://192.168.1.2/capture#x','http://192.168.1.2/\ncapture','https://192.168.1.2/capture',True):
            with self.subTest(value=value),self.assertRaises(ValueError): camera_source(value)

    def test_default_does_not_open_camera(self):
        self.camera.start();time.sleep(.06)
        self.assertIsNone(self.camera.capture)
        self.assertEqual(self.session.camera['state'],'disconnected')
        self.assertIsNone(self.session.link)

    def test_connect_frame_disconnect_and_invalid_switch(self):
        self.camera.start();self.configure('192.168.43.24')
        old=self.camera.capture
        old.queue.put(np.zeros((100,100,3),dtype=np.uint8))
        self.wait(lambda:self.session.camera['frames']==1 and self.session.captured is not None)
        self.assertTrue(self.session.snapshot(milliseconds())['frameFresh'])
        self.assertEqual(self.session.faces,[(25,25,75,75)])
        with self.assertRaises(ValueError): self.configure('https://public.example/capture')
        self.assertIs(self.camera.capture,old)
        self.configure()
        self.assertTrue(old.released)
        self.assertFalse(self.session.snapshot(milliseconds())['frameFresh'])
        self.assertIsNone(self.session.jpeg)

    def test_inflight_old_camera_frame_is_not_published(self):
        entered=threading.Event();finish=threading.Event()
        class Delayed(Capture):
            def read(self):
                entered.set();finish.wait(1)
                return True,SimpleNamespace(image=np.zeros((100,100,3),dtype=np.uint8),captured_ms=milliseconds())
        self.camera.capture_factory=Delayed
        self.configure('192.168.43.24');self.camera.start()
        self.assertTrue(entered.wait(1))
        self.configure();finish.set();time.sleep(.1)
        self.assertEqual(self.session.camera['state'],'disconnected')
        self.assertEqual(self.session.camera['frames'],0)
        self.assertIsNone(self.session.jpeg)

    def test_camera_failure_stops_tracking_and_recovery_does_not_restart(self):
        self.configure('192.168.43.24');self.camera.start()
        self.session.command(dict(action='auto',client='test',generation=self.session.generation,panSign=1,tiltSign=1),milliseconds())
        self.camera.capture.queue.put(OSError('offline'))
        self.wait(lambda:self.session.camera['state']=='error')
        self.assertEqual(self.session.mode,'idle')
        self.camera.capture.queue.put(np.zeros((100,100,3),dtype=np.uint8))
        self.wait(lambda:self.session.camera['state']=='streaming')
        self.assertEqual(self.session.mode,'idle')

    def test_http_camera_configuration_is_authenticated_and_generation_bound(self):
        server=ConsoleServer(self.session,camera=self.camera);server.start()
        try:
            def post(data,headers=None):
                conn=http.client.HTTPConnection('127.0.0.1',server.server.server_port,timeout=2)
                merged={'Origin':server.url[:-1],'Content-Type':'application/json','X-Control-Token':self.session.token}
                merged.update(headers or {})
                conn.request('POST','/api/camera',json.dumps(data),merged)
                res=conn.getresponse();res.read();conn.close();return res.status
            data=dict(action='connect',source='192.168.43.24',generation=0)
            self.assertEqual(post(data,{'X-Control-Token':'bad'}),403)
            self.assertIsNone(self.camera.capture)
            self.assertEqual(post(data),200)
            self.assertEqual(post(data),400)
            self.assertEqual(post({**data,'generation':self.session.generation,'extra':1}),400)
        finally: server.close()


if __name__=='__main__': unittest.main()
