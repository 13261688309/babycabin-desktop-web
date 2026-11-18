"""Standalone core-function console. Default is preview: no serial device access or output."""
import argparse
import json
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .gimbal import FaceTracker, encode_control, encode_rock_control, parse_control, match_selected, valid_face
from .gimbal_presets import PresetStore
from .gimbal_patrol import PatrolPlan
from .control_feedback import ControlFeedback
from .care_workflow import CareWorkflow
from .tracking_metrics import TrackingMetrics
from .camera_video import FrameRate
from .voice_control import VoiceWindow, parse_voice_control, COMMANDS
from .state import TemporalState
from .telemetry import unique_object


def milliseconds():
    return time.monotonic_ns() // 1000000


def parse_servo_outputs(line, now):
    prefix = b'servo_regs '
    if not line.startswith(prefix) or len(line) > 1536:
        raise ValueError('Invalid servo register report')
    row = json.loads(line[len(prefix):], object_pairs_hook=unique_object)
    if (not isinstance(row, dict) or set(row) != {'v','sample','now_ms','epoch','pwm','registers'}
            or row['v'] != 2 or type(row['pwm']) is not int or row['pwm'] not in (0, 1)):
        raise ValueError('Unsupported servo register report')
    regs = row['registers']
    fields = ('GPIOA_MODER','GPIOA_AFRL','TIM3_CR1','TIM3_CCER','TIM3_PSC','TIM3_ARR','TIM3_CCR1','TIM3_CCR2')
    if not isinstance(regs, dict):
        raise ValueError('Invalid servo register values')
    raw_values = [regs.get(field) for field in fields]
    if any(not isinstance(value, str) or len(value) != 10 or not value.startswith('0x')
           or any(digit not in '0123456789abcdefABCDEF' for digit in value[2:]) for value in raw_values):
        raise ValueError('Invalid servo register values')
    regs = {name:int(value, 16) for name, value in zip(fields, raw_values)}
    if any(type(row[k]) is not int or not 0 <= row[k] <= 0xffffffff for k in ('epoch','sample','now_ms')):
        raise ValueError('Invalid servo report counters')
    outputs = {'registerReportOnly':True, 'receivedMs':now, 'epoch':row['epoch'],
               'controllerMs':row['now_ms'], 'pwm':row['pwm'],
               'timerRunning':bool(regs['TIM3_CR1'] & 1), 'periodTicks':regs['TIM3_ARR'] + 1}
    for axis, pin, channel in (('pan', 6, 1), ('tilt', 7, 2)):
        outputs[axis] = {'pin':f'A{pin}', 'channel':channel,
                         'pinMuxTim3':(regs['GPIOA_MODER'] >> (pin * 2)) & 3 == 2
                                      and (regs['GPIOA_AFRL'] >> (pin * 4)) & 15 == 2,
                         'channelEnabled':bool(regs['TIM3_CCER'] & (1 << ((channel - 1) * 4))),
                         'compareTicks':regs[f'TIM3_CCR{channel}']}
    return outputs


class ControlLink:
    codec = None
    servo_outputs = None
    last_active_servo_outputs = None

    def __init__(self, port, serial_number, location, *, codec=None):
        from tools.capture_mcu_usb import require_endpoint
        import serial
        require_endpoint(port, serial_number, location)
        self.serial = serial.Serial(port=None, baudrate=115200, timeout=0, write_timeout=.15)
        self.serial.dtr = True
        self.serial.rts = False
        self.serial.port = port
        require_endpoint(port, serial_number, location)
        self.serial.open()
        self.buffer = bytearray()
        self.status = None
        self.received = 0
        self.sequence = 0
        self.feedback = ControlFeedback()
        self.voice_snapshot = None
        self.codec = codec

    def poll(self, now):
        chunk = self.serial.read(4096)
        self.feedback.feed(chunk, now)
        self.buffer.extend(chunk)
        while b'\n' in self.buffer:
            line, _, remainder = self.buffer.partition(b'\n')
            self.buffer = bytearray(remainder)
            if line.startswith(b'bench_link ' if self.codec else b'gimbal_link '):
                try:
                    self.status = (self.codec.parse_control if self.codec else parse_control)(bytes(line))
                    self.received = now
                except (ValueError,UnicodeError):
                    self.status = None
            elif line.startswith(b'voice_cmd '):
                try: self.voice_snapshot=dict(data=parse_voice_control(bytes(line)),receivedMs=now)
                except (ValueError,UnicodeError): self.voice_snapshot=None
            elif self.codec and line.startswith(b'servo_regs '):
                try:
                    self.servo_outputs = parse_servo_outputs(bytes(line), now)
                    if self.servo_outputs['pwm']:
                        self.last_active_servo_outputs = self.servo_outputs
                except (ValueError,UnicodeError):
                    self.servo_outputs = None
        if len(self.buffer)>4096:
            self.buffer.clear()
            self.status = None
        if now-self.received>=400:
            self.status = None
        return self.status

    def write(self, *, lease, op, pan=90, tilt=90, duration=0, speed=20, period=1000):
        if self.status is None:
            raise ValueError('Fresh GC03 firmware capability is required')
        control_encoder = self.codec.encode_control if self.codec else encode_control
        motor_encoder = self.codec.encode_rock_control if self.codec else encode_rock_control
        data = (motor_encoder(self.status,sequence=self.sequence,lease=lease,op=op,
                    speed=speed,period=period,duration=duration) if op in (4,5) else
                control_encoder(self.status, sequence=self.sequence, lease=lease, op=op,
                    pan=pan, tilt=tilt, duration=duration))
        if self.serial.write(data) != len(data):
            raise OSError('Partial command; no retry')
        self.sequence=(self.sequence+1)&0xffffffff

    def close(self):
        self.serial.close()


class GimbalSession:
    auto_frame_grace_ms = 0
    auto_max_frame_gap_ms = 400
    manual_pan_range = (60, 120)
    manual_tilt_range = (75, 105)
    def __init__(self, link=None, *, presets=None, observer=None):
        self.lock=threading.RLock()
        self.video_condition=threading.Condition(self.lock)
        self.link=link
        if link is not None and observer is not None:
            raise ValueError('Read-only observer and motion link are mutually exclusive')
        self.observer=observer
        self.hardware=link is not None
        self.token=secrets.token_hex(24)
        self.mode='idle'
        self.pan=self.tilt=90
        self.pan_sign=self.tilt_sign=1
        self.tracker=FaceTracker()
        self.lease=0
        self.started=self.last_write=self.heartbeat=0
        self.last_manual_target=None
        self.pending=False
        self.start_epoch=None
        self.sent=False
        self.stop_requested=False
        self.stop_lease=1
        self.stop_op=3
        self.status=None
        self.message='等待选择控制模式'
        self.faces=[]
        self.face_detector_name='haar'
        self.captured=None
        self.width=self.height=0
        self.jpeg=None
        self.error=None
        self.generation=0
        self.owner=None
        self.camera=dict(source=None,state='disconnected',error=None,generation=0,frames=0,
            targetFps=30,transport=None,fallbackReason=None,analysisRotation=0)
        self.video=None
        self.camera_rates={key:FrameRate() for key in ('capture','face','emotion')}
        self.emotion=TemporalState(confidence=.80,confirm_ms=3000,min_frames=5,max_gap_ms=1500)
        self.emotion_ready=False
        self.emotion_captured=None
        self.emotion_error=None
        self.emotion_preview=None
        self.rock_speed=20
        self.rock_duration=3000
        self.rock_period=1000
        self.rock_source=None
        self.duration=10000
        self.arm_epoch=None
        self.presets=presets if presets is not None else PresetStore()
        self.frame_history=deque(maxlen=32)
        self.selection_required=False
        self.selected_box=None
        self.selection_state='single'
        self.care=CareWorkflow()
        self.metrics=TrackingMetrics()
        self.patrol=None
        self.voice_window=VoiceWindow()
        self.simulated_voice=None

    def _voice_item(self,now):
        if self.hardware: return getattr(self.link,'voice_snapshot',None)
        if self.simulated_voice is None:
            self.simulated_voice=dict(v=1,frames=0,type=0,id=0,last_ms=0,now_ms=now&0xffffffff,
                                      stops=0,applied=0,alarms=0,errors=0,dropped=0,epoch=0)
        self.simulated_voice['now_ms']=now&0xffffffff
        return dict(data=dict(self.simulated_voice),receivedMs=now)

    def _voice_tick(self,now):
        window=self.voice_window
        if not window.armed: return
        owner=window.owner
        if self.hardware and (self.status is None or self.status['trip']):
            window.disarm('主控或保护未就绪，扩展语音已关闭')
        else:
            item=self._voice_item(now)
            command=window.poll(item,now,revision=self.presets.revision,
                epoch=self.status['epoch'] if self.status else 0,
                own_active_lease=bool(self.status and self.owner==window.owner and self.lease
                                      and self.status['lease']==self.lease))
            if command:
                action=command.pop('action'); slot=command.pop('slot')
                payload=dict(action=action,client=window.owner,generation=self.generation)
                if action=='preset_apply': payload['slot']=slot
                elif action=='auto': payload.update(panSign=self.pan_sign,tiltSign=self.tilt_sign)
                elif action=='patrol': payload.update(window.route)
                try:
                    if action=='auto' and (self.captured is None or not 0<=now-self.captured<400):
                        raise ValueError('无新鲜画面，不启动跟踪')
                    self.command(payload,now)
                    window.last_command['result']='电脑会话已接纳（非主控接纳、语音识别或机械实测证明）'
                except ValueError as error:
                    window.last_command['result']='拒绝：'+str(error)
                    window.disarm('扩展口令未通过控制校验，需重新允许')
        if not window.armed and self.mode!='idle' and self.owner==owner:
            owned=not self.hardware or bool(self.status and self.status['lease']==self.lease)
            self.stop('扩展语音窗口已关闭',now=now,notify_device=owned)

    patrol_limit_ms = PatrolPlan.MAX_MS

    def _patrol_command(self, data, now):
        action=data['action']
        if action=='patrol':
            if self.mode!='idle': raise ValueError('先停止当前动作，再启动巡航')
            if self.hardware and self.status['lease']!=0:
                raise ValueError('主控仍有其他控制会话，不能启动巡航')
            if data['pauseOnFace'] is True and (self.captured is None
                    or not 0<=now-self.captured<400 or self.faces):
                raise ValueError('发现人脸暂停功能需要新鲜、暂无人脸的相机画面')
            pan,tilt=(self.status['pan'],self.status['tilt']) if self.status else (self.pan,self.tilt)
            plan=PatrolPlan(self.presets,data['slots'],data['dwell'],pan=pan,tilt=tilt,
                            pause_on_face=data['pauseOnFace'],now=now,max_ms=self.patrol_limit_ms)
            self.patrol=plan
            self.mode='patrol'
            self.pan,self.tilt=pan,tilt
            self.started=self.heartbeat=now
            self.last_write=now-100
            self.duration=plan.max_ms
            self.owner=data['client']
            self.lease=secrets.randbelow(0xffffffff)+1
            self.pending=self.sent=False
            self.selected_box=None
            if self.selection_required:self.selection_state='reselect'
            self.care.interrupt('已切换到限时巡航',now)
            self.rock_source=None
        else:
            if self.mode!='patrol' or self.patrol is None:
                raise ValueError('当前没有可暂停或继续的巡航')
            if now-self.started>=self.duration or now-self.heartbeat>=600:
                self.stop('巡航期限或页面心跳已过期',now=now)
                raise ValueError('本轮已退出，不能继续')
            if action=='patrol_pause':
                self.patrol.pause(now)
                self.pan,self.tilt=self.patrol.target
            else:
                if self.patrol.pause_on_face and (self.captured is None
                        or not 0<=now-self.captured<400 or self.faces):
                    raise ValueError('当前画面过期或仍有人脸，保持暂停')
                self.patrol.resume(now)
            self.heartbeat=now
        self.generation+=1
        self.message=self.patrol.reason

    def controller_command(self, data, now):
        with self.lock:
            if (not isinstance(data,dict) or set(data)!={'action','generation'}
                    or data['action']!='disconnect' or type(data['generation']) is not int):
                raise ValueError('Invalid controller action')
            if self.hardware or self.observer is None:
                raise ValueError('Read-only controller connection required')
            if data['generation']!=self.generation:
                raise ValueError('Control state changed; refresh before disconnecting')
            if self.mode!='idle' or self.stop_requested:
                raise ValueError('Stop the current software session first')
            state=self.observer.snapshot(now)
            controller=state.get('controller',{})
            if controller.get('fresh') and any(controller.get('data',{}).get(key,0)
                    for key in ('lease','pwm','enabled','remaining','rock','motor')):
                raise ValueError('Controller reports an active action; not releasing telemetry')
            self.observer.close()
            self.generation+=1
            self.message='主控串口已释放；手机配对和相机保持连接'

    def care_command(self, data, now):
        with self.lock:
            if not isinstance(data,dict) or set(data)!={'action','id'} or data['action']!='ack':
                raise ValueError('Invalid reminder action')
            self.care.acknowledge(data['id'],now)

    def _care_tick(self, now):
        feedback=getattr(self.link,'feedback',None)
        device=(self.observer.snapshot(now) if self.observer is not None else
                feedback.snapshot(now,connected=self.status is not None) if feedback else dict(connected=False))
        self.care.tick(now,video_fresh=self.captured is not None and 0<=now-self.captured<1500,device=device)
        return device

    def _select(self, data, now):
        if self.mode!='idle' or self.stop_requested:
            raise ValueError('先停止当前动作，再选择目标')
        if data['action']=='clear_target':
            self.selection_required=False
            self.selected_box=None
            self.selection_state='single'
        else:
            if type(data['frameId']) is not int or type(data['index']) is not int:
                raise ValueError('Invalid target selection')
            frame=next((item for item in self.frame_history if item['id']==data['frameId']),None)
            if (frame is None or not 0<=now-frame['id']<400 or self.captured is None
                    or not 0<=now-self.captured<400 or not 0<=data['index']<len(frame['faces'])
                    or (frame['width'],frame['height'])!=(self.width,self.height)):
                raise ValueError('画面已变化，请在新画面重新选择')
            box,state=match_selected(frame['faces'][data['index']],self.faces,self.width,self.height)
            if box is None:
                raise ValueError('目标丢失或重叠，请重新选择')
            self.selected_box=box
            self.selection_required=True
            self.selection_state=state
        self.generation+=1
        self.message='目标已选择，尚未启动跟踪' if self.selection_required else '已恢复单人脸自动选择'

    def command(self, data, now):
        with self.lock:
            if not isinstance(data,dict) or type(data.get('action')) is not str:
                raise ValueError('Invalid action')
            action=data['action']
            fields={'action','client','generation'}
            if action=='manual': fields|={'pan','tilt'}
            if action=='auto': fields|={'panSign','tiltSign'}
            if action=='rock': fields|={'speed','duration'}
            if action=='patrol': fields|={'slots','dwell','pauseOnFace'}
            if action=='voice_arm': fields|={'slots','dwell','pauseOnFace'}
            if action=='voice_simulate': fields|={'id'}
            if action=='select_target': fields|={'frameId','index'}
            if action in ('preset_apply','preset_delete'): fields|={'slot'}
            if action=='preset_save': fields|={'slot','name','pan','tilt','panSign','tiltSign'}
            if (set(data)!=fields or not isinstance(data['client'],str) or not 1<=len(data['client'])<=64
                    or type(data['generation']) is not int):
                raise ValueError('Invalid action fields')
            if action=='stop':
                self.stop('已停止；再次运行需手动启动',global_stop=True,now=now)
                return
            if data['generation']!=self.generation:
                raise ValueError('控制状态已改变，请重新操作')
            if action=='heartbeat':
                if self.mode!='idle' and self.owner==data['client']:
                    self.heartbeat=now
                if self.voice_window.armed and self.voice_window.owner==data['client']:
                    self.voice_window.heartbeat=now
                return
            if action in ('voice_arm','voice_disarm','voice_simulate'):
                if action=='voice_disarm':
                    if self.voice_window.owner not in (None,data['client']): raise ValueError('另一页面已允许扩展语音')
                    owned=self.voice_window.owner==self.owner and self.mode!='idle'
                    self.voice_window.disarm('扩展语音已关闭')
                    if owned:self.stop('扩展语音已关闭',now=now)
                elif action=='voice_simulate':
                    if self.hardware: raise ValueError('硬件模式禁止注入模拟口令')
                    if not self.voice_window.armed or self.voice_window.owner!=data['client']:
                        raise ValueError('先允许本页面的限时口令预演')
                    if type(data['id']) is not int or data['id'] not in (*COMMANDS,1,2): raise ValueError('Invalid simulated command')
                    item=self._voice_item(now)
                    self.simulated_voice.update(frames=item['data']['frames']+1,id=data['id'],last_ms=now&0xffffffff)
                    if data['id'] in (1,2):
                        self.simulated_voice['stops']+=1
                        self.stop('模拟STOP已停止软件会话',notify_device=False,now=now)
                    else:self._voice_tick(now)
                else:
                    if self.mode!='idle' and self.owner!=data['client']: raise ValueError('另一页面正在控制')
                    if self.stop_requested: raise ValueError('正在停止，暂不允许扩展语音')
                    if self.hardware and (self.status is None or self.status['trip']): raise ValueError('主控或保护未就绪')
                    if type(data['slots']) is not list or type(data['dwell']) is not int or type(data['pauseOnFace']) is not bool:
                        raise ValueError('Invalid voice route')
                    if data['slots']:
                        PatrolPlan(self.presets,data['slots'],data['dwell'],pan=self.pan,tilt=self.tilt,pause_on_face=data['pauseOnFace'],now=now,max_ms=self.patrol_limit_ms)
                    elif data['dwell'] not in PatrolPlan.DWELLS: raise ValueError('Invalid dwell')
                    item=self._voice_item(now)
                    if self.hardware and (not VoiceWindow.fresh(item,now) or item['data']['epoch']!=self.status['epoch']):
                        raise ValueError('尚无与控制会话一致的增强语音数据')
                    self.voice_window.arm(item,now,client=data['client'],
                        route={key:data[key] for key in ('slots','dwell','pauseOnFace')},revision=self.presets.revision,
                        source='device' if self.hardware else 'simulation')
                self.generation+=1
                return
            if action in ('select_target','clear_target'):
                self._select(data,now)
                return
            if action in ('preset_save','preset_delete'):
                if self.mode!='idle' or self.stop_requested:
                    raise ValueError('先停止当前动作，再保存或删除预置')
                preset={key:data[key] for key in ('name','pan','tilt','panSign','tiltSign')} if action=='preset_save' else None
                self.presets.change(data['slot'],preset)
                self.generation+=1
                self.message='预置已保存，未发送动作' if preset else '预置已删除'
                return
            if action=='preset_apply':
                preset=self.presets.get(data['slot'])
                self.command(dict(action='manual',client=data['client'],generation=self.generation,
                                  pan=preset['pan'],tilt=preset['tilt']),now)
                self.pan_sign,self.tilt_sign=preset['panSign'],preset['tiltSign']
                self.message='预置目标已设置：'+preset['name']
                return
            if action not in ('manual','center','auto','rock','arm_soothe','patrol','patrol_pause','patrol_resume'):
                raise ValueError('Unsupported action')
            if self.mode!='idle' and self.owner!=data['client']:
                raise ValueError('另一页面正在控制，请先停止')
            if self.stop_requested:
                raise ValueError('正在停止，请稍后重试')
            if self.hardware and (self.status is None or self.status['trip']):
                raise ValueError('尚无可用控制固件或保护未就绪')
            if action in ('patrol','patrol_pause','patrol_resume'):
                self._patrol_command(data,now)
                return
            if action in ('rock','arm_soothe'):
                if self.mode!='idle': raise ValueError('先停止当前动作，再启动摇摆')
                if action=='rock' and (type(data['speed']) is not int or data['speed'] not in (10,15,20)
                        or type(data['duration']) is not int or data['duration'] not in (2000,3000,5000)):
                    raise ValueError('请选择10/15/20%和2/3/5秒')
                if action=='arm_soothe' and (not self.emotion_ready or self.emotion_captured is None
                        or not 0<=now-self.emotion_captured<1500 or self.captured is None or not 0<=now-self.captured<1500):
                    raise ValueError('需要本机模型和新鲜相机识别结果')
                self.started=self.heartbeat=now
                self.last_write=now-100
                self.owner=data['client']
                self.lease=secrets.randbelow(0xffffffff)+1
                self.pending=self.sent=False
                self.generation+=1
                self.rock_source='vision' if action=='arm_soothe' else 'manual'
                self.rock_speed=20 if action=='arm_soothe' else data['speed']
                self.rock_duration=3000 if action=='arm_soothe' else data['duration']
                self.duration=60000 if action=='arm_soothe' else self.rock_duration
                self.mode='observe' if action=='arm_soothe' else 'rock'
                self.care.interrupt('已切换到新的控制流程',now)
                if action=='rock': self.care.begin(now,source='manual',hardware=self.hardware)
                self.arm_epoch=self.status['epoch'] if self.status else None
                self.emotion.invalidate('armed' if action=='arm_soothe' else 'manual')
                self.message='等待持续3秒 Cry，最多等待60秒，仅触发一次' if action=='arm_soothe' else '限时电机摇摆'
                return
            if action=='manual':
                if (type(data['pan']) is not int or type(data['tilt']) is not int
                        or not self.manual_pan_range[0] <= data['pan'] <= self.manual_pan_range[1]
                        or not self.manual_tilt_range[0] <= data['tilt'] <= self.manual_tilt_range[1]):
                    raise ValueError('角度超出限位')
                pan,tilt=data['pan'],data['tilt']
            else:
                angles=(self.status['pan'],self.status['tilt']) if self.mode=='idle' and self.status else (self.pan,self.tilt)
                pan,tilt=(90,90) if action=='center' else angles
            if action=='auto' and (type(data['panSign']) is not int or type(data['tiltSign']) is not int
                                    or data['panSign'] not in (-1,1) or data['tiltSign'] not in (-1,1)):
                raise ValueError('Invalid axis direction')
            if action=='auto' and self.selection_required and (self.selected_box is None
                    or self.captured is None or not 0<=now-self.captured<400):
                raise ValueError('所选目标已失效，请重新选择')
            # Switching modes needs a stop; it cannot renew an already active deadline.
            next_mode='auto' if action=='auto' else 'manual'
            if self.mode!='idle' and next_mode!=self.mode:
                raise ValueError('切换模式前请先停止')
            if self.mode=='auto':
                raise ValueError('跟踪已运行，重新设置前请先停止')
            if self.mode=='idle':
                self.duration=10000
                self.rock_source=None
                self.started=now
                self.owner=data['client']
                self.lease=secrets.randbelow(0xffffffff)+1
                self.pending=False
                self.sent=False
                self.last_write=now-100
                self.generation+=1
                if self.status is not None:
                    self.pan,self.tilt=self.status['pan'],self.status['tilt']
                self.tracker.reset(self.pan,self.tilt)
            self.pan,self.tilt=pan,tilt
            self.care.interrupt('已切换到云台控制，结束上次只读观察',now)
            self.mode=next_mode
            self.heartbeat=now
            if action=='auto':
                self.pan_sign,self.tilt_sign=data['panSign'],data['tiltSign']
                if not self.selection_required:
                    self._seed_single_face(now)
                self.metrics.begin(now,hardware=self.hardware,selected=self.selection_required)
            else:
                self.selected_box=None
                if self.selection_required:self.selection_state='reselect'
            self.message=('等待所选目标连续三帧' if self.selection_required else '等待连续三帧单人脸') if next_mode=='auto' else '角度目标已设置'

    def _seed_single_face(self, now):
        # At most two observations: cached detections can never produce a target.
        for frame in list(self.frame_history)[-2:]:
            if not (0<=now-frame['id']<400 and (frame['width'],frame['height'])==(self.width,self.height)):
                continue
            if all(valid_face(box,self.width,self.height) for box in frame['faces']):
                self.tracker.update(frame['faces'],self.width,self.height,frame['id'],now)
            else:
                self.tracker.lose('invalid')

    def stop(self, reason, *, notify_device=True, global_stop=False, now=None, completed=False):
        now=milliseconds() if now is None else now
        if self.voice_window.armed:
            self.voice_window.disarm('停止、接管或控制结束，扩展语音已关闭')
        if self.mode=='auto': self.metrics.end(now,reason)
        if self.mode=='patrol' and self.patrol is not None:
            self.patrol.end(now,reason,completed=completed)
        if self.mode=='rock':
            accepted=bool(self.hardware and self.sent and not self.pending and self.status
                          and not self.status['trip'] and self.status['lease']==self.lease)
            self.care.end(now,completed=completed,accepted=accepted)
        elif not completed:
            self.care.interrupt('停止或连接切换，结束本次只读观察',now)
        if not notify_device:
            self.stop_requested=False
        elif global_stop or self.mode!='idle' or self.pending:
            self.stop_requested=self.hardware
            self.stop_lease=self.lease or 1
            self.stop_op=0 if global_stop else 3
        self.mode='idle'
        self.lease=0
        self.pending=False
        self.start_epoch=None
        self.sent=False
        self.owner=None
        self.message=reason
        self.generation+=1
        self.tracker.reset(self.pan,self.tilt)
        self.selected_box=None
        if self.selection_required:self.selection_state='reselect'

    def emotion_frame(self,detections,captured,now,width,height):
        with self.lock:
            self.care.observe(detections,captured,now,width,height)
            if self.emotion_captured is not None and captured<=self.emotion_captured: return
            # Inference can finish after the user arms; pre-arm frames cannot start a new confirmation.
            if self.mode=='observe' and captured<self.started: return
            if (type(width) is not int or type(height) is not int or width<=0 or height<=0
                    or not 0<=now-captured<1500
                    or any(not item.valid() or item.xyxy[2]>width or item.xyxy[3]>height for item in detections)):
                self.emotion_preview=None
                self.emotion.invalidate('stale_or_invalid')
                if self.mode in ('observe','rock') and self.rock_source=='vision': self.stop('识别过期或无效，自动安抚已取消',now=now)
                return
            self.emotion_captured=captured
            self.emotion_ready=True
            self.emotion_error=None
            # Display candidates never feed the temporal state or authorize motion.
            self.emotion_preview=dict(frameId=captured,width=width,height=height,
                cameraGeneration=self.camera['generation'],
                detections=[dict(label=item.label,confidence=item.confidence,xyxy=list(item.xyxy))
                    for item in sorted(detections,key=lambda item:item.confidence,reverse=True)[:20]])
            observation=self.emotion.update(detections,captured)
            if self.mode=='observe' and observation.state=='Cry':
                self.mode='rock'
                self.started=now
                self.duration=self.rock_duration
                self.last_write=now-100
                self.message='持续 Cry 已确认，启动一次限时摇摆'
                self.care.begin(now,source='vision',hardware=self.hardware)
            elif self.mode=='rock' and self.rock_source=='vision' and observation.state!='Cry':
                self.stop('Cry 不再持续或目标不明确，自动摇摆已停止',now=now)

    def frame(self, faces, width, height, captured, now, jpeg=None):
        with self.lock:
            if self.captured is not None and captured <= self.captured:
                return
            previous_capture = self.captured
            if self.selected_box is not None:
                if (self.captured is None or captured-self.captured>=400
                        or (width,height)!=(self.width,self.height) or not 0<=now-captured<400):
                    self.selected_box=None;self.selection_state='stale'
                else:
                    self.selected_box,self.selection_state=match_selected(self.selected_box,faces,width,height)
            self.faces=[tuple(box) for box in faces]; self.width=width; self.height=height; self.captured=captured; self.jpeg=jpeg
            self.frame_history.append(dict(id=captured,faces=list(self.faces),width=width,height=height,jpeg=jpeg))
            if self.mode=='patrol' and self.patrol.pause_on_face and captured>=self.started:
                if (not 0<=now-captured<400 or type(width) is not int or type(height) is not int
                        or width<=0 or height<=0 or any(not valid_face(box,width,height) for box in faces)):
                    self.stop('巡航人脸监测画面无效，已退出',now=now)
                    return
                if self.patrol.face_seen(bool(faces),captured,now):
                    self.pan,self.tilt=self.patrol.target
                    self.generation+=1
                    self.message=self.patrol.reason
            if self.mode=='auto':
                if captured<=self.started:
                    return
                if self.selection_required and self.selected_box is None:
                    self.metrics.observe([],width,height,captured,now,state=self.selection_state,pan=self.pan,tilt=self.tilt)
                    self.stop('所选目标丢失、交叉或过期，跟踪已停止',now=now)
                    return
                targets=[self.selected_box] if self.selection_required else faces
                max_gap = self.auto_max_frame_gap_ms if not self.selection_required else 400
                last_good_capture = self.tracker.last_capture
                bridge_misses = (self.auto_frame_grace_ms and not self.selection_required and self.sent
                                 and self.tracker.state in ('tracking', 'centered')
                                 and last_good_capture is not None)
                if (bridge_misses and previous_capture != last_good_capture
                        and now-last_good_capture >= max_gap):
                    self.stop('人脸持续丢失，跟踪已停止',now=now)
                    return
                if (bridge_misses and not targets and 0 <= now-last_good_capture < max_gap
                        and (width,height) == self.tracker.dimensions and 0 <= now-captured < 400):
                    self.metrics.observe([],width,height,captured,now,state='waiting',pan=self.pan,tilt=self.tilt)
                    self.message='本帧未检测到人脸，暂停角度更新'
                    return
                angles=self.tracker.update(targets,width,height,captured,now,pan_sign=self.pan_sign,
                                           tilt_sign=self.tilt_sign,max_gap_ms=max_gap)
                if angles is not None:
                    self.pan,self.tilt=angles
                    self.message='人脸居中' if self.tracker.state=='centered' else ('正在跟踪所选目标' if self.selection_required else '正在跟踪单人脸')
                self.metrics.observe(targets,width,height,captured,now,state=self.tracker.state,pan=self.pan,tilt=self.tilt)
                if angles is None and (self.pending or (self.tracker.state not in ('waiting','acquiring') and self.sent)):
                    reason = {'lost':'人脸目标丢失', 'changed':'目标位置或画面尺寸突变',
                              'multiple':'检测到多张人脸', 'stale':'画面间隔过长',
                              'invalid':'人脸数据无效'}.get(self.tracker.state, '无法确认跟踪目标')
                    self.stop(reason+'，跟踪已停止',now=now)

    def tick(self, now):
        with self.lock:
            try:
                if self.observer is not None:
                    self.observer.poll(now)
                if self.link:
                    self.status=self.link.poll(now)
                self._care_tick(now)
                self._voice_tick(now)
                if self.stop_requested:
                    if self.status is not None:
                        self.link.write(lease=self.stop_lease,op=self.stop_op)
                    self.stop_requested=False
                    return
                if self.mode=='idle': return
                if now-self.heartbeat>=600:
                    self.stop('控制页面失联，已停止',now=now); return
                if now-self.started>=self.duration:
                    self.stop('本次控制结束，请手动重新启动',now=now,completed=True); return
                if self.hardware and self.status is None:
                    self.stop('主控回执失联，停止发送动作',now=now); return
                if self.mode in ('observe','rock') and self.rock_source=='vision':
                    if (self.emotion_captured is None or not 0<=now-self.emotion_captured<1500
                            or self.captured is None or not 0<=now-self.captured<1500):
                        self.stop('识别画面失联，自动安抚已取消',now=now); return
                    if not self.sent and self.hardware and self.status['epoch']!=self.arm_epoch:
                        self.stop('主控状态已改变，自动安抚已取消',notify_device=False,now=now); return
                if self.mode=='observe': return
                if self.pending and self.status['lease']!=self.lease:
                    if self.status['epoch']!=self.start_epoch:
                        self.stop('主控已停止或语音接管，不自动恢复',notify_device=False,now=now); return
                    if now-self.last_write>=400:
                        self.stop('启动未获主控确认，或被语音/保护取消',now=now); return
                    return
                if self.pending:
                    self.pending=False
                if self.hardware and self.sent and self.status['lease']!=self.lease:
                    # This is an observed handover, not a new global STOP request.
                    # Sending STOP here would also cancel the new voice-owned action.
                    self.stop('主控已停止或语音接管，不自动恢复',notify_device=False,now=now); return
                if self.mode=='auto':
                    grace = 0 if self.selection_required else self.auto_frame_grace_ms
                    capture = self.tracker.last_capture if grace and self.sent else self.captured
                    if (grace and self.sent and capture is not None and capture != self.captured
                            and now-capture >= self.auto_max_frame_gap_ms):
                        self.stop('人脸持续丢失，跟踪已停止',now=now); return
                    if capture is None or not 0 <= now-capture < 400:
                        if self.sent and (capture is None or not 0 <= now-capture < 400+grace):
                            self.stop('画面已过期，跟踪停止',now=now)
                        return
                    if capture != self.captured: return
                    if self.tracker.state not in ('centered','tracking'): return
                if self.mode=='patrol':
                    if self.patrol.pause_on_face and (self.captured is None or not 0<=now-self.captured<400):
                        self.stop('巡航人脸监测画面失联，已退出',now=now); return
                    angles=self.patrol.advance(now)
                    if angles is None:
                        self.stop('巡航一轮已完成；再次运行需手动启动',now=now,completed=True); return
                    self.pan,self.tilt=angles
                target=(self.pan,self.tilt)
                changed_manual=self.mode=='manual' and self.sent and target!=self.last_manual_target
                if now-self.last_write<(40 if changed_manual else 100): return
                if self.hardware:
                    starting=self.status['lease']==0
                    if starting: self.start_epoch=self.status['epoch']
                    if self.mode=='rock':
                        self.link.write(lease=self.lease,op=4 if starting else 5,speed=self.rock_speed,period=self.rock_period,
                            duration=max(1,self.duration-(now-self.started)) if starting else 0)
                    else:
                        self.link.write(lease=self.lease,op=1 if starting else 2,pan=self.pan,tilt=self.tilt,
                            duration=max(1,self.duration-(now-self.started)) if starting else 0)
                    self.pending=starting
                if self.mode=='manual':
                    self.last_manual_target=target
                self.last_write=now
                self.sent=True
            except (OSError,ValueError) as error:
                self.error=type(error).__name__
                self.stop('控制链路异常，已停止；不自动重连',now=now)
                if self.link:
                    self.link.close()
                    self.link=None
                self.status=None

    def snapshot(self, now):
        with self.lock:
            fresh=self.captured is not None and 0<=now-self.captured<400
            emotion_fresh=self.emotion_captured is not None and 0<=now-self.emotion_captured<1500
            preview=self.emotion_preview
            preview=(dict(preview,frameAgeMs=now-preview['frameId'],
                detections=[dict(item,xyxy=list(item['xyxy'])) for item in preview['detections']])
                if preview and emotion_fresh and self.emotion_ready and not self.emotion_error
                and preview['cameraGeneration']==self.camera['generation'] else None)
            device=self._care_tick(now)
            control_state=('preview' if not self.hardware else 'offline' if self.status is None else
                'stopping' if self.stop_requested else 'observing' if self.mode=='observe' else
                'idle' if self.mode=='idle' else 'pending' if self.pending or not self.sent else 'accepted')
            return dict(token=self.token, hardware=self.hardware, mode=self.mode, faceDetector=self.face_detector_name,
                pan=self.pan,tilt=self.tilt, remainingMs=max(0,self.duration-(now-self.started)) if self.mode!='idle' else 0,
                generation=self.generation, message=self.message, tracker=self.tracker.state,
                faces=self.faces if fresh else [], width=self.width,height=self.height,frameFresh=fresh,
                status=self.status,error=self.error,camera=dict(self.camera,
                    fps={key:rate.value(now) for key,rate in self.camera_rates.items()}),physicalAngleMeasured=False,
                videoFresh=self.video is not None and 0<=now-self.video['id']<400,
                device=device,controlState=control_state,
                care=self.care.snapshot(now),
                trackingMetrics=self.metrics.snapshot(now),
                patrol=self.patrol.snapshot(now) if self.patrol else None,
                patrolLimitMs=self.patrol_limit_ms,
                voiceControl=dict(self.voice_window.snapshot(now),
                    associatedDataFresh=VoiceWindow.fresh(self._voice_item(now),now)),
                frameId=self.captured,frameAgeMs=now-self.captured if self.captured is not None else None,
                selection=dict(mode='selected' if self.selection_required else 'single',
                    state=self.selection_state if fresh else 'stale',box=self.selected_box if fresh else None),
                presets=self.presets.snapshot(),axes=dict(panSign=self.pan_sign,tiltSign=self.tilt_sign),
                emotion=dict(ready=self.emotion_ready,error=self.emotion_error,
                    fresh=emotion_fresh,preview=preview,frameId=self.emotion_captured,
                    observation=self.emotion.expire(now).to_dict()),
                motor=dict(source=self.rock_source,speed=self.rock_speed,period=self.rock_period,
                    active=self.mode=='rock',targetPercent=(self.rock_speed*(-1 if (now-self.started)//self.rock_period%2 else 1)) if self.mode=='rock' else 0,
                    physicalMotionMeasured=False))


class ConsoleServer:
    def __init__(self, session, port=0, camera=None, mobile=None):
        self.session=session
        class Handler(BaseHTTPRequestHandler):
            def send(self,status,kind,body,extra=None):
                self.send_response(status)
                for k,v in {'Content-Type':kind,'Content-Length':str(len(body)),'Cache-Control':'no-store',
                    'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY',
                    'Content-Security-Policy':"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self' http://127.0.0.1:5000; frame-ancestors 'none'"}.items():
                    self.send_header(k,v)
                for k,v in (extra or {}).items(): self.send_header(k,str(v))
                try:
                    self.end_headers(); self.wfile.write(body)
                except (ConnectionError,TimeoutError):
                    pass
            def local(self): return self.headers.get('Host')==f'127.0.0.1:{self.server.server_port}'
            def do_GET(self):
                if not self.local(): return self.send(403,'text/plain',b'Invalid host')
                route=self.path.split('?',1)[0]
                files={'/':('gimbal_console.html','text/html; charset=utf-8'),
                       '/console.js':('gimbal_console.js','text/javascript; charset=utf-8'),
                       '/console.css':('gimbal_console.css','text/css; charset=utf-8'),
                       '/mobile':('harmony_pair.html','text/html; charset=utf-8'),
                       '/mobile.js':('harmony_pair.js','text/javascript; charset=utf-8'),
                       '/mobile.css':('harmony_pair.css','text/css; charset=utf-8')}
                if route in files:
                    filename,kind=files[route]; return self.send(200,kind,Path(__file__).with_name(filename).read_bytes())
                if route=='/api/mobile-pairing':
                    if (mobile is None or self.headers.get('Sec-Fetch-Site')=='cross-site'
                            or self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}')):
                        return self.send(403,'text/plain',b'Local mobile gateway required')
                    result=mobile.pairing.display(milliseconds())
                    result.update(url=mobile.url,hardware=session.hardware)
                    return self.send(200,'application/json',json.dumps(result).encode())
                if route=='/api/state': return self.send(200,'application/json',json.dumps(session.snapshot(milliseconds()),ensure_ascii=False).encode())
                if route=='/api/tracking-report':
                    with session.lock: report=session.metrics.report(milliseconds())
                    return self.send(200,'application/json',json.dumps(report,ensure_ascii=False,allow_nan=False).encode())
                if route=='/video.jpg':
                    # One current JPEG, never a client-specific frame backlog or a second camera connection.
                    query=parse_qs(urlsplit(self.path).query)
                    with session.video_condition:
                        if session.video and query.get('after')==[str(session.video['id'])]:
                            session.video_condition.wait_for(lambda:session.video is None
                                or query.get('after')!=[str(session.video['id'])],timeout=.1)
                        frame=session.video
                        now=milliseconds()
                        fresh=frame is not None and 0<=now-frame['id']<2000
                        image=frame['jpeg'] if fresh and query.get('after')!=[str(frame['id'])] else None
                        headers={'X-Frame-Id':frame['id'],'X-Frame-Age-Ms':now-frame['id'],
                            'X-Frame-Width':frame['width'],'X-Frame-Height':frame['height'],
                            'X-Camera-Generation':session.camera['generation']} if image else None
                    return self.send(200 if image else 204,'image/jpeg',image or b'',headers)
                if route=='/frame.jpg':
                    with session.lock:
                        query=parse_qs(urlsplit(self.path).query)
                        selected=query.get('frame',[str(session.captured)])
                        frame=next((item for item in session.frame_history if selected==[str(item['id'])]),None)
                        image=frame['jpeg'] if frame and session.captured is not None and 0<=milliseconds()-frame['id']<400 else None
                        metadata=dict(frameId=frame['id'],frameAgeMs=milliseconds()-frame['id'],
                            faces=frame['faces'],width=frame['width'],height=frame['height'],
                            camera=dict(generation=session.camera['generation']),
                            selection=dict(box=session.selected_box)) if image else None
                    return self.send(200 if image else 404,'image/jpeg',image or b'',
                        {'X-Frame-Meta':json.dumps(metadata,separators=(',',':'))} if metadata else None)
                return self.send(404,'text/plain',b'Not found')
            def do_POST(self):
                token=self.headers.get('X-Control-Token','')
                if (not self.local() or self.path not in ('/api/control','/api/camera','/api/care','/api/controller')
                        or self.headers.get('Origin')!=f'http://127.0.0.1:{self.server.server_port}'
                        or not token.isascii() or not secrets.compare_digest(token,session.token)):
                    return self.send(403,'text/plain',b'Local control session required')
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 1<=size<=512 or self.headers.get('Content-Type')!='application/json' or self.headers.get('Transfer-Encoding'):
                        raise ValueError('Invalid body')
                    self.connection.settimeout(1)
                    data=json.loads(self.rfile.read(size),object_pairs_hook=unique_object)
                    if self.path=='/api/camera':
                        if camera is None: raise ValueError('Camera connection unavailable')
                        camera.configure(data)
                    elif self.path=='/api/controller':
                        session.controller_command(data,milliseconds())
                    elif self.path=='/api/care':
                        session.care_command(data,milliseconds())
                    else:
                        session.command(data,milliseconds())
                except (ValueError,OSError,UnicodeError,RecursionError) as error:
                    return self.send(400,'text/plain; charset=utf-8',str(error).encode())
                return self.send(200,'application/json',json.dumps(session.snapshot(milliseconds()),ensure_ascii=False).encode())
            def log_message(self,*args): pass
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    @property
    def url(self): return f'http://127.0.0.1:{self.server.server_port}/'
    def start(self): self.thread.start()
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--source',help='Local camera IP or /stream URL, /capture for compatibility; omitted means no camera access')
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--console-alias-port',type=int,help='Second loopback view of the same session and camera')
    parser.add_argument('--emotion-weights',type=Path,help='Pinned local Cry/Happy/Neutral/Sleep model; no download')
    parser.add_argument('--face-weights',type=Path,help='Opt in to pinned local YuNet face detection; no download')
    parser.add_argument('--preferences-file',type=Path,help='Local preset file; never restores a running action')
    parser.add_argument('--mobile-bind',help='Opt-in private LAN IPv4 for the paired TLS native app')
    parser.add_argument('--mobile-port',type=int,default=8443)
    parser.add_argument('--mobile-cert',type=Path)
    parser.add_argument('--mobile-key',type=Path)
    parser.add_argument('--enable-hardware',action='store_true')
    parser.add_argument('--servo-tilt-only',action='store_true',help='Verified GB01 A7 manual profile; no interlock claims')
    parser.add_argument('--servo-bench',action='store_true',help='GB01 original UI; axes must be explicitly selected and confirmed')
    parser.add_argument('--servo-firmware-root',type=Path)
    parser.add_argument('--serial-port')
    parser.add_argument('--serial-number')
    parser.add_argument('--usb-location')
    parser.add_argument('--confirm-physical-interlock',action='store_true')
    parser.add_argument('--unoccupied-bench-confirmed',action='store_true')
    parser.add_argument('--telemetry-port')
    parser.add_argument('--telemetry-serial-number')
    parser.add_argument('--telemetry-usb-location')
    parser.add_argument('--confirm-native-controller',action='store_true')
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--confirm-telemetry-isolated',action='store_true')
    scope.add_argument('--confirm-voice-dht-wiring',action='store_true',
                       help='Voice and H743 separately USB powered, DHT11 on 3V3, actuators disconnected')
    args=parser.parse_args(argv)
    if args.console_alias_port is not None and (not 1<=args.console_alias_port<=65535
            or args.console_alias_port==args.port):
        parser.error('Alias must use a different valid port')
    if any((args.mobile_bind,args.mobile_cert,args.mobile_key)) and not all((args.mobile_bind,args.mobile_cert,args.mobile_key)):
        parser.error('Mobile access requires explicit bind address, certificate and key')
    hardware_args=(args.serial_port,args.serial_number,args.usb_location,args.confirm_physical_interlock,args.unoccupied_bench_confirmed)
    if args.enable_hardware and not all(hardware_args):
        parser.error('Hardware requires explicit endpoint, verified physical interlock and unoccupied bench confirmations')
    if not args.enable_hardware and any(hardware_args):
        parser.error('Hardware arguments require explicit --enable-hardware')
    telemetry_args=(args.telemetry_port,args.telemetry_serial_number,args.telemetry_usb_location,
                    args.confirm_native_controller,args.confirm_telemetry_isolated or args.confirm_voice_dht_wiring)
    servo_bench = args.servo_tilt_only or args.servo_bench
    if args.servo_tilt_only and args.servo_bench:
        parser.error('Select one GB01 profile')
    if bool(args.servo_firmware_root) != servo_bench:
        parser.error('GB01 tilt profile requires the verified firmware workspace')
    if servo_bench and (args.enable_hardware or any(hardware_args) or any(telemetry_args)):
        parser.error('GB01 cannot be combined with GC03 or a second serial owner')
    if any(telemetry_args) and (not all(telemetry_args) or args.enable_hardware):
        parser.error('Read-only telemetry requires exact endpoint and fresh native USB wiring confirmation; no motion link')
    camera_options={}
    if args.face_weights:
        from .face_yunet import YuNetFaceDetector
        try:
            face_detector=YuNetFaceDetector(args.face_weights)
        except (OSError,ValueError) as error:
            parser.error(f'YuNet could not start: {error}')
        camera_options['detector_factory']=lambda: face_detector
    if servo_bench:
        from .servo_bench_profile import open_verified_link, TiltBenchSession, verified_codec
        from .gimbal_presets import DEFAULT_LIMITS
        connection_error=None
        try:
            link=open_verified_link(args.servo_firmware_root)
        except (OSError,ValueError,RuntimeError) as error:
            link=None
            connection_error='主控未连接：'+str(error)
        try:
            codec = link.codec if link is not None else verified_codec(args.servo_firmware_root)
        except (OSError, ValueError):
            codec = None
        limits = (getattr(codec, "manual_pan_range", DEFAULT_LIMITS[0]),
                  getattr(codec, "manual_tilt_range", DEFAULT_LIMITS[1]))
        session=TiltBenchSession(link,presets=PresetStore(args.preferences_file, limits=limits),allow_gimbal=args.servo_bench,
                                 firmware_root=args.servo_firmware_root)
        if args.servo_bench:
            session.servo_axes='gimbal'
            session.message='GB01双轴实机已接入；确认水平A6/俯仰A7后手动操作'
        if connection_error:
            session.message=connection_error
    else:
        link=ControlLink(args.serial_port,args.serial_number,args.usb_location) if args.enable_hardware else None
        session=GimbalSession(link,presets=PresetStore(args.preferences_file))
    from .gimbal_camera import GimbalCamera
    if args.face_weights:
        session.face_detector_name='yunet'
    emotion_factory=None
    if args.emotion_weights:
        from .emotion_control import EmotionModel
        emotion_factory=lambda: EmotionModel(args.emotion_weights)
    camera=GimbalCamera(session,emotion_factory=emotion_factory,**camera_options)
    mobile=None
    if args.mobile_bind:
        from .harmony_bridge import MobileServer
        mobile=MobileServer(session,camera,args.mobile_bind,args.mobile_port,args.mobile_cert,args.mobile_key)
    server=ConsoleServer(session,args.port,camera=camera,mobile=mobile)
    alias=ConsoleServer(session,args.console_alias_port,camera=camera,mobile=mobile) if args.console_alias_port else None
    stop=threading.Event()
    observer=None
    try:
        if args.telemetry_port:
            from .controller_observer import ControllerObserver
            observer=ControllerObserver(args.telemetry_port,args.telemetry_serial_number,args.telemetry_usb_location,
                                        native_board_confirmed=True,isolated_usb_confirmed=args.confirm_telemetry_isolated,
                                        voice_and_dht11_harness_confirmed=args.confirm_voice_dht_wiring,
                                        actuators_disconnected_confirmed=args.confirm_voice_dht_wiring,
                                        separate_usb_power_confirmed=args.confirm_voice_dht_wiring)
            session.observer=observer
        camera.start()
        if args.source:
            camera.configure(dict(action='connect',source=args.source,generation=session.generation))
        server.start()
        if alias: alias.start()
        if mobile:
            mobile.start()
            print(f'Mobile: {mobile.url}  Pairing code (5 minutes, one phone): {mobile.pairing.code}',flush=True)
        print(server.url+' '+('GB01 SERVO HARDWARE; NO STARTUP ACTION' if servo_bench else
              'HARDWARE CONTROL' if args.enable_hardware else
              'READ-ONLY CONTROLLER TELEMETRY' if observer is not None else 'PREVIEW ONLY, NO DEVICE OUTPUT'),flush=True)
        while not stop.wait(.02): session.tick(milliseconds())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        with session.lock: session.stop('Console closed')
        session.tick(milliseconds())
        camera.close()
        if mobile and mobile.thread.is_alive(): mobile.close()
        if server.thread.is_alive(): server.close()
        if alias and alias.thread.is_alive(): alias.close()
        if session.link: session.link.close()
        if observer: observer.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
