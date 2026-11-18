import http.client
import json
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from companion.camera_video import CameraCapture, FrameRate
from companion.gimbal_camera import GimbalCamera
from companion.gimbal_console import ConsoleServer, GimbalSession, milliseconds


class VideoTests(unittest.TestCase):
    def test_stream_failure_closes_before_fallback_and_retry(self):
        now=[0]; active=[]; history=[]
        class Backend:
            def __init__(self,source,**kwargs):
                self.source=source; self.closed=False
                assert not active
                active.append(self);history.append(source)
            def read(self):
                if self.source.endswith('/stream') and now[0]<30: raise TimeoutError()
                return True,'frame'
            def release(self):
                if not self.closed: active.remove(self);self.closed=True
        capture=CameraCapture('http://192.168.1.9:81/stream',stream_factory=Backend,
            snapshot_factory=Backend,clock=lambda:now[0])
        self.addCleanup(capture.release)
        with self.assertRaises(TimeoutError): capture.read()
        self.assertEqual(active,[])
        self.assertEqual(capture.read(),(True,'frame'))
        self.assertEqual(capture.transport,'snapshot')
        now[0]=31
        self.assertEqual(capture.read(),(True,'frame'))
        self.assertEqual(capture.transport,'mjpeg')
        self.assertIsNone(capture.fallback_reason)
        self.assertEqual(history,['http://192.168.1.9:81/stream','http://192.168.1.9/capture','http://192.168.1.9:81/stream'])

    def test_explicit_snapshot_never_probes_a_stream(self):
        calls=[]
        class Backend:
            def __init__(self,source,**kwargs): calls.append(source)
            def read(self): return True,'frame'
            def release(self): pass
        capture=CameraCapture('http://127.0.0.1:1234/capture',snapshot_factory=Backend,
            stream_factory=lambda *a,**k:self.fail('unexpected stream'))
        self.addCleanup(capture.release)
        capture.read();capture.read()
        self.assertEqual(calls,['http://127.0.0.1:1234/capture'])

    def test_established_stream_retries_early_but_unstable_retry_backs_off(self):
        now = [0]
        streams = iter([iter([True, True, True, False]), iter([True, False])])
        class Backend:
            def __init__(self, source, **kwargs):
                self.frames = next(streams) if source.endswith('/stream') else None

            def read(self):
                return (self.frames is None or next(self.frames)), 'frame'

            def release(self):
                pass

        capture = CameraCapture('http://192.168.1.9:81/stream', stream_factory=Backend,
                                snapshot_factory=Backend, clock=lambda: now[0])
        self.addCleanup(capture.release)
        for _ in range(3):
            self.assertEqual(capture.read(), (True, 'frame'))
        with self.assertRaises(OSError):
            capture.read()
        now[0] = 2
        capture.read()
        self.assertEqual(capture.transport, 'snapshot')
        now[0] = 3
        capture.read()
        self.assertEqual(capture.transport, 'mjpeg')
        with self.assertRaises(OSError):
            capture.read()
        now[0] = 32
        capture.read()
        self.assertEqual(capture.transport, 'snapshot')
        self.assertEqual(capture.retry_at, 33)

    def test_eof_is_failure_not_busy_loop(self):
        backend=SimpleNamespace(read=lambda:(False,None),release=lambda:None)
        capture=CameraCapture('http://127.0.0.1:81/stream',stream_factory=lambda *a,**k:backend)
        self.addCleanup(capture.release)
        with self.assertRaises(OSError):capture.read()
        self.assertEqual(capture.transport,'snapshot')
        capture.release()
        self.assertEqual(capture.read(),(False,None))

    def test_rates_measure_arrivals_and_expire_without_new_frames(self):
        rate=FrameRate()
        for value in range(0,1000,33):rate.add(value)
        self.assertAlmostEqual(rate.value(1000),30.3,places=1)
        self.assertEqual(rate.value(1500),0)
        self.assertEqual(rate.value(4000),0)

    def test_slow_detection_does_not_throttle_video_and_queues_stay_latest_only(self):
        ended=threading.Event()
        class Paced:
            def __init__(self,*a,**k):self.next=time.monotonic();self.jpeg=b'original-camera-jpeg'
            def read(self):
                ended.wait(max(0,self.next-time.monotonic()));self.next+=1/30
                return not ended.is_set(),SimpleNamespace(image=np.zeros((60,80,3),np.uint8),captured_ms=milliseconds(),jpeg=self.jpeg)
            def release(self):ended.set()
        class Slow:
            def detect(self,frame):time.sleep(.13);return []
        session=GimbalSession()
        camera=GimbalCamera(session,capture_factory=Paced,detector_factory=Slow,emotion_factory=Slow)
        camera.configure(dict(action='connect',source='127.0.0.1',generation=0));camera.start()
        try:
            time.sleep(2)
            state=session.snapshot(milliseconds())
            self.assertGreaterEqual(state['camera']['frames'],50)
            self.assertGreater(state['camera']['fps']['capture'],25)
            self.assertLess(state['camera']['fps']['face'],10)
            self.assertLess(state['camera']['fps']['emotion'],10)
            self.assertLessEqual(camera.face_jobs.qsize(),1)
            self.assertLessEqual(camera.emotion_jobs.qsize(),1)
            self.assertEqual(session.video['jpeg'],b'original-camera-jpeg')
            self.assertTrue(state['frameFresh'])
            self.assertEqual(session.mode,'idle')
        finally:camera.close()

    def test_video_endpoint_does_not_serve_duplicates_stale_images_or_foreign_hosts(self):
        session=GimbalSession();server=ConsoleServer(session);server.start()
        self.addCleanup(server.close)
        def get(path,headers=None):
            connection=http.client.HTTPConnection('127.0.0.1',server.server.server_port,timeout=2)
            connection.request('GET',path,headers=headers or {})
            response=connection.getresponse();body=response.read()
            result=response.status,dict(response.getheaders()),body
            connection.close();return result
        self.assertEqual(get('/video.jpg')[0],204)
        now=milliseconds();session.video=dict(id=now,width=80,height=60,jpeg=b'jpeg')
        status,headers,body=get('/video.jpg')
        self.assertEqual((status,body),(200,b'jpeg'))
        self.assertEqual(headers['X-Frame-Id'],str(now))
        self.assertEqual(get('/video.jpg?after='+str(now))[0],204)
        self.assertEqual(get('/video.jpg',{'Host':'elsewhere.invalid'})[0],403)
        session.video['id']=milliseconds()-500
        self.assertEqual(get('/video.jpg')[0],200)
        self.assertFalse(session.snapshot(milliseconds())['frameFresh'])
        session.video['id']=milliseconds()-2001
        self.assertEqual(get('/video.jpg')[0],204)

    def test_two_console_ports_share_the_same_camera_and_session(self):
        session=GimbalSession();calls=[]
        camera=SimpleNamespace(configure=lambda data:calls.append(data))
        servers=[ConsoleServer(session,camera=camera) for _ in range(2)]
        for server in servers:
            server.start();self.addCleanup(server.close)
        now=milliseconds();session.video=dict(id=now,width=80,height=60,jpeg=b'shared')
        for server in servers:
            connection=http.client.HTTPConnection('127.0.0.1',server.server.server_port,timeout=2)
            connection.request('GET','/api/state');response=connection.getresponse()
            self.assertEqual(json.loads(response.read())['token'],session.token)
            connection.request('GET','/video.jpg');response=connection.getresponse()
            self.assertEqual(response.getheader('X-Frame-Id'),str(now))
            self.assertEqual(response.read(),b'shared');connection.close()
        self.assertEqual(calls,[])

    def test_inflight_detection_cannot_return_after_capture_failure(self):
        entered=threading.Event();finish=threading.Event()
        class Slow:
            def detect(self,frame):entered.set();finish.wait(2);return [(0,0,20,20)]
        session=GimbalSession();camera=GimbalCamera(session,detector_factory=Slow)
        capture=SimpleNamespace(release=lambda:None)
        camera.capture=capture
        camera.face_jobs.put((capture,0,0,np.zeros((60,80,3),np.uint8),milliseconds(),b'jpeg'))
        camera.face_thread.start()
        try:
            self.assertTrue(entered.wait(1))
            camera._failed(capture,0,OSError('lost'))
            finish.set();time.sleep(.05)
            self.assertIsNone(session.captured)
            self.assertIsNone(session.video)
            self.assertEqual(session.faces,[])
        finally:finish.set();camera.close()


if __name__=='__main__':unittest.main()
