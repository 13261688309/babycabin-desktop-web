"""Verified GB01 servos inside the original console, not a GC03 interlock bypass."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from .gimbal_console import ControlLink, GimbalSession, milliseconds
from .gimbal import FaceTracker

PIN = '1c2d2ee6e3ff9e5a91011c961369be4898fba060d1c76e2ce4b88accf7ed0faa'
RESPONSE_PIN = 'e11dbb9f0881eb7db810269e26b1724f6a4a7d9125fa8d5927ffb4f38bb85dff'
FASTER_RESPONSE_PIN = '3c0e09d423d28b837d0d6314027aaee58965023e0dde9d885c548bd6a28fbd6d'
TRACKING_30S_PIN = 'fa0bf305aa0a21e631493317b16305e726e2219a040bf4d244997639c27dff46'
TILT_SPAN_PIN = '40c4ebdee50b48c524134c341d7c6f4f7b8582b190b749bdb7cc4f06cd902310'
TRAVEL_CALIBRATION_PIN = '62a5c0257ce1f76da53a38e92f10e2ebcff2193840baf1b0bf97bec5d9f2ec46'
DOWNWARD_CALIBRATION_PIN = '6414a0ad887ba3c9b60e2d598c13ff9d3c94bbdd4ce618e3571879b0a5206f90'
EXPANDED_CALIBRATION_PIN = '043d302e8d1fb513ce4f5c1b3bf2dfe720a0f9e0a9bc39c21e2d9ffed7aa149e'
DHT11_PIN = 'eaa956abc0d5399897d21fbeea5ea1d196fe14451b49d41db6375e4a56a4aee6'
FW_BUILDS = {
    DHT11_PIN: ('gb01-dht11-01', 'gb01-dht11-dfu-01'),
    EXPANDED_CALIBRATION_PIN: ('gb01-expanded-calibration-01', 'gb01-expanded-calibration-dfu-01'),
    PIN: ('servo-idle-low-01', 'servo-idle-low-dfu-01'),
    RESPONSE_PIN: ('servo-response-03', 'servo-response-dfu-01'),
    FASTER_RESPONSE_PIN: ('servo-response-04', 'servo-response-dfu-02'),
    TRACKING_30S_PIN: ('gb01-tracking-30s-01', 'gb01-tracking-30s-dfu-01'),
    TILT_SPAN_PIN: ('gb01-tilt-span-01', 'gb01-tilt-span-dfu-01'),
    TRAVEL_CALIBRATION_PIN: ('gb01-travel-calibration-01', 'gb01-travel-calibration-dfu-01'),
    DOWNWARD_CALIBRATION_PIN: ('gb01-downward-calibration-02', 'gb01-downward-calibration-dfu-02'),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_codec(root):
    root = Path(root).resolve()
    active = json.loads((root/'handoff/next-firmware.json').read_text(encoding='utf-8'))
    pin = active['manifestSha256']
    if pin not in FW_BUILDS:
        raise ValueError('Verified current GB01 flash required')
    build, trial = FW_BUILDS[pin]
    base = root/'build/verification'/build
    if hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest() != pin:
        raise ValueError('GB01 manifest mismatch')
    manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    codec_path = base/'console/companion/gimbal.py'
    if hashlib.sha256(codec_path.read_bytes()).hexdigest() != manifest['consoleFiles']['companion/gimbal.py']:
        raise ValueError('GB01 protocol mismatch')
    evidence = root/'build/bench'/trial
    proof = json.loads((evidence/'readback-verification.json').read_text(encoding='utf-8'))
    flash = evidence/'after-flash.bin'
    if (not active['fullFlashReadbackVerified'] or not proof['passed'] or flash.stat().st_size != 0x200000
            or proof['manifestSha256'] != pin
            or hashlib.sha256(flash.read_bytes()).hexdigest() != proof['completeReadbackSha256']):
        raise ValueError('Verified current GB01 flash required')
    if pin != PIN:
        idle = json.loads((evidence/'runtime-verification.json').read_text(encoding='utf-8'))
        if not (idle['passed'] and idle['firmwareManifestSha256'] == pin
                and idle['applicationBytesWritten'] == 0 and not idle['actuatorPowerApplied']):
            raise ValueError('Verified response firmware idle output required')
    codec = load_module('verified_gb01_codec',codec_path)
    codec.tracking_session_limit_ms = 30000 if pin in (
        TRACKING_30S_PIN, TILT_SPAN_PIN, TRAVEL_CALIBRATION_PIN, DOWNWARD_CALIBRATION_PIN, EXPANDED_CALIBRATION_PIN, DHT11_PIN
    ) else 10000
    if pin in (TRAVEL_CALIBRATION_PIN, DOWNWARD_CALIBRATION_PIN, EXPANDED_CALIBRATION_PIN, DHT11_PIN):
        codec.manual_pan_range = tuple(manifest['manualAxes']['pan'])
        codec.manual_tilt_range = tuple(manifest['manualAxes']['tilt'])
    return codec


def open_verified_link(root):
    from serial.tools import list_ports
    codec = verified_codec(root)
    ports = [p for p in list_ports.comports() if
             (p.vid,p.pid,p.serial_number) == (0x0483,0x5740,'336435763533')]
    if len(ports) != 1:
        raise RuntimeError('H743 endpoint missing or ambiguous; no automatic retry')
    p = ports[0]
    link = ControlLink(p.device,p.serial_number,p.location,codec=codec)
    try:
        raw = bytearray()
        until = milliseconds()+1800
        while milliseconds() < until:
            raw.extend(link.serial.read(4096))
            time.sleep(.02)
        checker = load_module('idle_v2_checker',Path(root)/'build/bench/servo-idle-low-dfu-01/verify_runtime.py')
        rows = [json.loads(line[len(b'servo_regs '):],object_pairs_hook=checker.unique)
                for line in bytes(raw).split(b'\n')[:-1] if line.startswith(b'servo_regs ')]
        if not rows or not all(checker.idle_checks(row)['passed'] for row in rows):
            raise RuntimeError('Fresh idle-low registers required')
        link.buffer.extend(raw)
        s = link.poll(milliseconds())
        if s is None or any(s[key] for key in ('lease','pwm','fault','motor','rock','remaining')):
            raise RuntimeError('MCU must be idle; no startup action sent')
        return link
    except BaseException:
        link.close()
        raise


class TiltBenchSession(GimbalSession):
    auto_frame_grace_ms = 250
    auto_max_frame_gap_ms = 500
    def __init__(self, link=None, *, allow_gimbal=False, firmware_root=None, **kwargs):
        super().__init__(link,**kwargs)
        if allow_gimbal:
            self.tracker = FaceTracker(fixed_camera=True)
        self.firmware_root = firmware_root
        if firmware_root is not None:
            self.hardware = True
        self.tilt_ready = False
        self.allow_gimbal = allow_gimbal
        self.servo_axes = 'tilt'
        self.message = 'GB01俯仰实机已接入；确认接线后手动应用角度'

    def controller_command(self, data, now):
        with self.lock:
            if (not isinstance(data,dict) or set(data) != {'action','generation'}
                    or data['action'] not in ('connect','disconnect')
                    or type(data['generation']) is not int or data['generation'] != self.generation):
                raise ValueError('Invalid controller connection request')
            if self.firmware_root is None or self.mode != 'idle' or self.stop_requested:
                raise ValueError('先停止当前动作')
            if self.status and (self.status['lease'] or self.status['pwm']):
                raise ValueError('主控仍在运行，不能切换连接')
            self.tilt_ready = False
            if self.link:
                self.link.close()
            self.link = None
            self.status = None
            self.generation += 1
            if data['action'] == 'disconnect':
                self.message = '主控已断开；相机保持连接，不自动重连'
                return
            try:
                self.link = open_verified_link(self.firmware_root)
                self.status = self.link.poll(milliseconds())
                self.error = None
                self.message = '主控已连接；未发送动作，请确认接线范围'
            except (OSError,ValueError,RuntimeError) as error:
                self.error = type(error).__name__
                self.message = '主控连接失败：'+str(error)
                raise ValueError(self.message) from error

    def command(self, data, now):
        if not isinstance(data,dict):
            raise ValueError('Invalid command')
        action = data.get('action')
        allowed = ('manual','center','stop','heartbeat','tilt_ready','servo_setup','preset_save',
                   'preset_delete','preset_apply','select_target','clear_target')
        if self.servo_axes == 'gimbal':
            allowed += ('auto','patrol','patrol_pause','patrol_resume')
        if action not in allowed:
            raise ValueError('当前固件或接线范围不支持此动作；不会改为模拟执行')
        with self.lock:
            if action == 'servo_setup':
                if (not self.allow_gimbal or set(data) != {'action','client','generation','axes'}
                        or data['axes'] not in ('tilt','gimbal') or type(data['generation']) is not int
                        or data['generation'] != self.generation or not isinstance(data['client'],str)
                        or not 1 <= len(data['client']) <= 64):
                    raise ValueError('Invalid servo setup')
                if self.mode != 'idle' or self.stop_requested or self.tilt_ready:
                    raise ValueError('先停止并取消接线确认；断舵机电源后才能切换接线范围')
                if self.hardware and (self.status is None or self.status['lease'] or self.status['pwm']):
                    raise ValueError('需要主控空闲回执')
                self.servo_axes = data['axes']
                self.generation += 1
                self.message = '接线范围已选择；尚未启动任何动作'
                return
            if action == 'tilt_ready':
                if (set(data) != {'action','client','generation','value'} or type(data['value']) is not bool
                        or type(data['generation']) is not int or data['generation'] != self.generation
                        or not isinstance(data['client'],str) or not 1 <= len(data['client']) <= 64):
                    raise ValueError('Invalid readiness request')
                if self.mode != 'idle' or self.stop_requested:
                    raise ValueError('先停止，再确认接线')
                self.tilt_ready = data['value']
                self.generation += 1
                return
            if action in ('manual','center','preset_apply','auto','patrol','patrol_resume') and not self.tilt_ready:
                raise ValueError('先确认当前舵机接线范围和独立5V供电')
            if self.servo_axes == 'tilt' and action in ('manual','preset_save'):
                if (type(data.get('pan')) is not int or data['pan'] != 90
                        or type(data.get('tilt')) is not int or not 85 <= data['tilt'] <= 95):
                    raise ValueError('本轮水平固定90；俯仰只开放85至95')
            idle = self.mode == 'idle'
            if idle and action == 'auto':
                self.tracker.limits = (self.manual_pan_range, self.manual_tilt_range)
            super().command(data,now)
            if idle and self.mode == 'manual' and self.servo_axes == 'tilt':
                self.duration = 6000
            if idle and action == 'auto' and self.mode == 'auto':
                self.duration = self.tracking_session_limit_ms

    @property
    def patrol_limit_ms(self):
        return self.tracking_session_limit_ms

    @property
    def tracking_session_limit_ms(self):
        return getattr(getattr(self.link, 'codec', None), 'tracking_session_limit_ms', 10000)

    @property
    def manual_pan_range(self):
        return getattr(getattr(self.link, 'codec', None), 'manual_pan_range', (60, 120))

    @property
    def manual_tilt_range(self):
        return getattr(getattr(self.link, 'codec', None), 'manual_tilt_range', (75, 105))

    def tick(self, now):
        super().tick(now)
        with self.lock:
            if self.hardware and self.status is None:
                self.tilt_ready = False

    def snapshot(self, now):
        with self.lock:
            s = super().snapshot(now)
            tilt = self.servo_axes == 'tilt'
            outputs = getattr(self.link, 'servo_outputs', None)
            if outputs and not 0 <= now - outputs['receivedMs'] < 2500:
                outputs = None
            s.update(profile='GB01-idle-low',servoBench=True,tiltOnly=tilt,tiltReady=self.tilt_ready,
                     servoOutputs=outputs, lastActiveServoOutputs=getattr(self.link, 'last_active_servo_outputs', None),
                     trackingCameraMount='fixed' if not tilt else None,
                     servoAxes=self.servo_axes,canSelectServoAxes=self.allow_gimbal,
                     panRange=self.manual_pan_range, tiltRange=[85,95] if tilt else self.manual_tilt_range,
                     sessionLimitMs=6000 if tilt else 10000,
                     trackingSessionLimitMs=None if tilt else self.tracking_session_limit_ms,
                     externalInterlockPresent=False,physicalAngleMeasured=False,
                     hardwareFeatures=dict(manual=True,presets=True,tracking=not tilt,patrol=not tilt,
                         voice=False,motor=False,camera='computer',emotion='computer'))
            return s
