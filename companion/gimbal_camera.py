"""Explicit camera connection for the gimbal console; no discovery or device writes."""
import http.client
import ipaddress
import queue
import threading
import time
from urllib.parse import urlsplit

from .gimbal import FaceDetector
from .camera_video import CameraCapture, FrameRate
from .image_orientation import checked_rotation, rotate_frame, restore_box
from .state import Detection


def camera_source(value):
    if not isinstance(value,str) or not 1<=len(value)<=200:
        raise ValueError('请输入相机 IP、/stream 或 /capture 地址')
    value=value.strip()
    if any(ord(char)<=32 or ord(char)==127 for char in value):
        raise ValueError('相机地址不能包含空白字符')
    url=urlsplit(value if '://' in value else 'http://'+value)
    try:
        address=ipaddress.IPv4Address(url.hostname)
        port=url.port
    except (ValueError,TypeError):
        raise ValueError('请输入局域网 IPv4 地址，不使用域名') from None
    local=address.is_loopback or any(address in ipaddress.IPv4Network(network)
        for network in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16'))
    if (not local or url.scheme!='http' or url.username is not None or url.password is not None
            or url.fragment or url.query or url.path not in ('','/','/capture','/stream') or port==0):
        raise ValueError('仅支持本机或局域网 HTTP /stream、/capture 地址，不含账号和参数')
    path=url.path if url.path in ('/capture','/stream') else '/stream'
    if path=='/stream' and port is None: port=81
    return f'http://{address}'+(f':{port}' if port is not None else '')+path


class GimbalCamera:
    def __init__(self,session,*,capture_factory=CameraCapture,detector_factory=FaceDetector,emotion_factory=None):
        self.session=session
        self.capture_factory=capture_factory
        self.detector_factory=detector_factory
        self.emotion_factory=emotion_factory
        self.emotion_jobs=queue.Queue(maxsize=1)
        self.face_jobs=queue.Queue(maxsize=1)
        self.work_epoch=0
        self.face_thread=threading.Thread(target=self._faces,name='face-inference',daemon=True)
        self.emotion_thread=threading.Thread(target=self._emotions,name='emotion-inference',daemon=True) if emotion_factory else None
        self.capture=None
        self.stop_event=threading.Event()
        self.wakeup=threading.Event()
        self.thread=threading.Thread(target=self._run,name='gimbal-camera',daemon=True)

    def start(self):
        self.thread.start()
        self.face_thread.start()
        if self.emotion_thread: self.emotion_thread.start()

    def configure(self,data):
        if not isinstance(data,dict) or type(data.get('action')) is not str:
            raise ValueError('Invalid camera action')
        action=data['action']
        fields={'action','generation'}|({'source'} if action=='connect' else {'rotation'} if action=='analysis_rotation' else set())
        if set(data)!=fields or type(data['generation']) is not int or action not in ('connect','disconnect','analysis_rotation'):
            raise ValueError('Invalid camera fields')
        if action=='analysis_rotation':
            return self._rotate_analysis(data)
        source=camera_source(data['source']) if action=='connect' else None
        with self.session.lock:
            if self.stop_event.is_set() or data['generation']!=self.session.generation:
                raise ValueError('控制状态已改变，请重新操作')
            new=self.capture_factory(source,read_timeout_ms=1000,open_timeout_ms=1000) if source else None
            old,self.capture=self.capture,new
            self.work_epoch+=1
            self.session.stop('相机已切换，请重新启动跟踪' if source else '相机已断开')
            generation=self.session.camera['generation']+1
            self.session.camera=dict(source=source,state='connecting' if source else 'disconnected',
                error=None,generation=generation,frames=0,targetFps=30,
                transport=getattr(new,'transport',None),fallbackReason=None,
                analysisRotation=self.session.camera.get('analysisRotation',0))
            self.session.video=None
            self.session.video_condition.notify_all()
            self.session.camera_rates={key:FrameRate() for key in ('capture','face','emotion')}
            self.session.jpeg=None
            self.session.captured=None
            self.session.faces=[]
            self.session.frame_history.clear()
            self.session.width=self.session.height=0
            self.session.emotion.invalidate('camera_changed')
            self.session.emotion_captured=None
            self.session.emotion_preview=None
            self.session.care.invalidate('camera_changed',time.monotonic_ns()//1000000)
        if old: old.release()
        self.wakeup.set()

    def _rotate_analysis(self,data):
        rotation=checked_rotation(data['rotation'])
        with self.session.lock:
            session=self.session
            if self.stop_event.is_set() or data['generation']!=session.generation:
                raise ValueError('控制状态已改变，请重新操作')
            if (session.mode!='idle' or session.stop_requested or session.pending
                    or session.hardware and (session.status is None or session.status['lease']!=0)):
                raise ValueError('先停止控制并等待主控空闲，再修改识别方向')
            if rotation==session.camera.get('analysisRotation',0):
                return
            # Keep the upstream reader; reject in-flight work and old selectable boxes.
            self.work_epoch+=1
            session.stop('识别方向已更新，目标选择已清除')
            session.camera.update(analysisRotation=rotation,generation=session.camera['generation']+1)
            session.video=None
            session.video_condition.notify_all()
            session.jpeg=None
            session.captured=None
            session.faces=[]
            session.frame_history.clear()
            session.width=session.height=0
            session.emotion.invalidate('analysis_changed')
            session.emotion_captured=None
            session.emotion_preview=None
            session.camera_rates['face']=FrameRate()
            session.camera_rates['emotion']=FrameRate()
            session.care.invalidate('analysis_changed',time.monotonic_ns()//1000000)

    @staticmethod
    def _offer(jobs,item):
        try: jobs.get_nowait()
        except queue.Empty: pass
        jobs.put_nowait(item)

    def _current(self,capture,generation,epoch):
        return (capture is self.capture and generation==self.session.camera['generation']
                and epoch==self.work_epoch and not self.stop_event.is_set())

    def _run(self):
        while not self.stop_event.is_set():
            self.wakeup.clear()
            with self.session.lock:
                capture=self.capture
                generation=self.session.camera['generation']
            if capture is None:
                self.wakeup.wait(.2)
                continue
            try:
                import cv2
                ok,item=capture.read()
                if not ok:
                    self.wakeup.wait(.05)
                    continue
                frame=item.image
                jpeg=getattr(item,'jpeg',None)
                if jpeg is None:
                    good,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
                    if not good: raise ValueError('JPEG encoding failed')
                    jpeg=encoded.tobytes()
                with self.session.lock:
                    # A read that finishes after a switch cannot republish its old frame.
                    if capture is not self.capture or generation!=self.session.camera['generation']:
                        continue
                    if self.session.video and item.captured_ms<=self.session.video['id']: continue
                    now=time.monotonic_ns()//1000000
                    if not 0<=now-item.captured_ms<400: continue
                    self.session.video=dict(id=item.captured_ms,jpeg=jpeg,width=frame.shape[1],height=frame.shape[0])
                    self.session.video_condition.notify_all()
                    self.session.camera_rates['capture'].add(now)
                    self.session.camera.update(state='streaming',error=None,
                        frames=self.session.camera['frames']+1,transport=getattr(capture,'transport','snapshot'),
                        fallbackReason=getattr(capture,'fallback_reason',None))
                    job=(capture,generation,self.work_epoch,frame,item.captured_ms,jpeg)
                    self._offer(self.face_jobs,job)
                    if self.emotion_thread: self._offer(self.emotion_jobs,job)
            except (OSError,ValueError,RuntimeError,http.client.HTTPException,ImportError) as error:
                self._failed(capture,generation,error)
                self.wakeup.wait(.25)
            except Exception as error:
                # Native detector errors must fail closed instead of leaving the last image live.
                self._failed(capture,generation,error)
                self.wakeup.wait(.25)

    def _faces(self):
        detector=None
        while not self.stop_event.is_set():
            try: capture,generation,epoch,frame,captured,jpeg=self.face_jobs.get(timeout=.2)
            except queue.Empty: continue
            with self.session.lock:
                if not self._current(capture,generation,epoch): continue
                rotation=self.session.camera.get('analysisRotation',0)
            try:
                if detector is None: detector=self.detector_factory()
                faces=detector.detect(rotate_frame(frame,rotation))
                if rotation:
                    faces=[restore_box(box,rotation,frame.shape[1],frame.shape[0]) for box in faces]
                with self.session.lock:
                    if not self._current(capture,generation,epoch): continue
                    now=time.monotonic_ns()//1000000
                    if not 0<=now-captured<400: continue
                    self.session.frame(faces,frame.shape[1],frame.shape[0],captured,now,jpeg)
                    self.session.camera_rates['face'].add(now)
            except Exception as error:
                self._failed(capture,generation,error,epoch=epoch)
                self.stop_event.wait(.25)

    def _failed(self,capture,generation,error,epoch=None):
        with self.session.lock:
            if capture is not self.capture or generation!=self.session.camera['generation']: return
            if epoch is not None and epoch!=self.work_epoch: return
            self.work_epoch+=1
            self.session.video=None
            self.session.video_condition.notify_all()
            self.session.jpeg=None
            self.session.captured=None
            self.session.faces=[]
            self.session.frame_history.clear()
            self.session.selected_box=None
            if self.session.selection_required:self.session.selection_state='lost'
            self.session.camera.update(state='error',error=type(error).__name__,
                transport=getattr(capture,'transport',None),fallbackReason=getattr(capture,'fallback_reason',None))
            self.session.emotion.invalidate('camera_failed')
            self.session.emotion_captured=None
            self.session.emotion_preview=None
            self.session.care.invalidate('camera_failed',time.monotonic_ns()//1000000)
            if (self.session.mode in ('auto','observe') or (self.session.mode=='rock' and self.session.rock_source=='vision')
                    or self.session.mode=='patrol' and self.session.patrol.pause_on_face):
                self.session.stop('相机取图失败，自动动作已停止')

    def _emotions(self):
        model=None
        while not self.stop_event.is_set():
            try: capture,generation,epoch,frame,captured,_jpeg=self.emotion_jobs.get(timeout=.2)
            except queue.Empty: continue
            with self.session.lock:
                if not self._current(capture,generation,epoch): continue
                rotation=self.session.camera.get('analysisRotation',0)
            try:
                if model is None: model=self.emotion_factory()
                detections=model.detect(rotate_frame(frame,rotation))
                if rotation:
                    detections=[Detection(item.label,item.confidence,
                        restore_box(item.xyxy,rotation,frame.shape[1],frame.shape[0])) for item in detections]
                with self.session.lock:
                    if self._current(capture,generation,epoch):
                        now=time.monotonic_ns()//1000000
                        self.session.emotion_frame(detections,captured,now,frame.shape[1],frame.shape[0])
                        self.session.camera_rates['emotion'].add(now)
            except Exception as error:
                with self.session.lock:
                    if not self._current(capture,generation,epoch): continue
                    self.session.emotion_ready=False
                    self.session.emotion_error=type(error).__name__
                    self.session.emotion_preview=None
                    self.session.emotion.invalidate('model_error')
                    self.session.care.invalidate('model_error',time.monotonic_ns()//1000000)
                    if self.session.mode=='observe' or (self.session.mode=='rock' and self.session.rock_source=='vision'):
                        self.session.stop('识别模型异常，自动安抚已取消')
                if model is None: return

    def close(self):
        self.stop_event.set()
        with self.session.lock:
            capture,self.capture=self.capture,None
        if capture: capture.release()
        self.wakeup.set()
        if self.thread.is_alive(): self.thread.join(timeout=5)
        if self.thread.is_alive(): raise RuntimeError('Camera worker did not stop')
        if self.face_thread.is_alive(): self.face_thread.join(timeout=5)
        if self.face_thread.is_alive(): raise RuntimeError('Face worker did not stop')
        if self.emotion_thread and self.emotion_thread.is_alive(): self.emotion_thread.join(timeout=30)
        if self.emotion_thread and self.emotion_thread.is_alive(): raise RuntimeError('Emotion worker did not stop')
