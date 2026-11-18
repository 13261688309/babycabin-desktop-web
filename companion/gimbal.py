"""Gimbal and bounded motor control using a distinct opt-in GC03 protocol."""
import math
import re
import struct
import zlib

PACKET = struct.Struct('<4sIIIIHHBBHI')
LIMITS = ((60, 120), (75, 105))
FIELDS = {'v', 'epoch', 'lease', 'seq', 'now_ms', 'accepted', 'rejected', 'pan', 'tilt', 'remaining', 'enabled', 'trip', 'rock', 'motor'}


def encode_control(state, *, sequence, lease, op, pan=90, tilt=90, duration=0):
    values = (sequence, lease, op, pan, tilt, duration, state.get('epoch'), state.get('now_ms'))
    if any(type(v) is not int for v in values):
        raise ValueError('Integer control fields required')
    if (not 0 <= sequence <= 0xffffffff or not 1 <= lease <= 0xffffffff or op not in (0, 1, 2, 3)
            or not 0 <= state['epoch'] <= 0xffffffff or not 0 <= state['now_ms'] <= 0xffffffff
            or not 60 <= pan <= 120 or not 75 <= tilt <= 105
            or (op == 1 and not 1 <= duration <= 10000) or (op != 1 and duration != 0)):
        raise ValueError('Control bounds exceeded')
    data = PACKET.pack(b'GC03', state['epoch'], sequence, state['now_ms'], lease,
                       pan, tilt, op, 0, duration, 0)
    return data[:-4] + struct.pack('<I', zlib.crc32(data[:-4]))


def encode_rock_control(state, *, sequence, lease, op, speed=20, period=1000, duration=0):
    # Reuse the checked header; bytes 20..23 carry motor percent and half-cycle ms.
    if (type(speed) is not int or not 1<=speed<=20 or type(period) is not int or not 500<=period<=2000
            or type(op) is not int or op not in (4,5) or type(duration) is not int
            or (op==4 and not 1<=duration<=5000) or (op==5 and duration!=0)):
        raise ValueError('Rocking bounds exceeded')
    data=bytearray(encode_control(state,sequence=sequence,lease=lease,op=1 if op==4 else 2,duration=duration))
    struct.pack_into('<HHB',data,20,speed,period,op)
    struct.pack_into('<I',data,28,zlib.crc32(data[:28]))
    return bytes(data)


def parse_control(line):
    if not isinstance(line, bytes) or not line.startswith(b'gimbal_link ') or len(line) > 256:
        raise ValueError('Not a gimbal capability')
    result = {}
    for token in line.decode('ascii').strip().split(' ')[1:]:
        key, sep, value = token.partition('=')
        pattern='-?[0-9]{1,3}' if key=='motor' else '[0-9]{1,10}'
        if not sep or key in result or not re.fullmatch(pattern, value) or int(value) > 0xffffffff:
            raise ValueError('Invalid control status')
        result[key] = int(value)
    if (set(result) != FIELDS or result['v'] != 3 or result['enabled'] not in (0, 1)
            or result['rock'] not in (0,1) or not -100<=result['motor']<=100
            or not 60 <= result['pan'] <= 120 or not 75 <= result['tilt'] <= 105
            or result['remaining'] > 10000):
        raise ValueError('Unsupported control status')
    return result


def overlap(a, b):
    x=max(a[0],b[0]); y=max(a[1],b[1]); r=min(a[2],b[2]); d=min(a[3],b[3])
    intersection=max(0,r-x)*max(0,d-y)
    return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def valid_face(box, width, height):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def match_selected(box, faces, width, height):
    """Spatial continuity only; ambiguous crossings fail closed, never identify a person."""
    if not valid_face(box, width, height) or any(not valid_face(face, width, height) for face in faces):
        return None, 'invalid'
    candidates = [face for face in faces if overlap(box, face) >= .25]
    if len(candidates) != 1:
        return None, 'ambiguous' if candidates else 'lost'
    selected = candidates[0]
    if sum(overlap(selected, face) > .10 for face in faces) != 1:
        return None, 'ambiguous'
    return tuple(selected), 'selected'


class FaceTracker:
    """Single target, three-frame acquisition, bounded proportional steering. No identity recognition."""
    def __init__(self, *, fixed_camera=False):
        self.fixed_camera = fixed_camera
        self.reset()

    def reset(self, pan=90, tilt=90):
        self.pan, self.tilt = float(pan), float(tilt)
        self.previous = None
        self.last_capture = None
        self.dimensions = None
        self.confirmations = 0
        self.filtered = None
        self.state = 'waiting'
        self.box = None

    def lose(self, reason):
        self.previous = self.filtered = self.box = None
        self.confirmations = 0
        self.state = reason
        return None

    def update(self, faces, width, height, captured_ms, now_ms, *, pan_sign=1, tilt_sign=1, max_gap_ms=400):
        if (type(captured_ms) is not int or not 0 <= now_ms-captured_ms < 400
                or type(width) is not int or type(height) is not int or width <= 0 or height <= 0):
            return self.lose('stale')
        if self.last_capture is not None and captured_ms <= self.last_capture:
            return None
        gap = None if self.last_capture is None else captured_ms-self.last_capture
        self.last_capture = captured_ms
        changed_size = self.dimensions != (width, height)
        self.dimensions = (width, height)
        if self.previous is not None and changed_size:
            return self.lose('changed')
        if self.previous is not None and gap is not None and gap >= max_gap_ms:
            return self.lose('stale')
        if len(faces) != 1:
            return self.lose('multiple' if faces else 'lost')
        box = faces[0]
        if (len(box) != 4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height
                or pan_sign not in (-1,1) or tilt_sign not in (-1,1)):
            return self.lose('invalid')
        if self.previous is not None and overlap(self.previous,box) < .25:
            return self.lose('changed')
        self.previous = self.box = tuple(box)
        self.confirmations += 1
        if self.confirmations < 3:
            self.state = 'acquiring'
            return None
        error = ((box[0]+box[2])/width-1, (box[1]+box[3])/height-1)
        elapsed = (gap or 200) / 1000
        alpha = -math.expm1(-elapsed / .12)
        deadband = .06

        def filter_axis(raw, previous):
            if abs(raw) <= deadband:
                return 0.
            if previous is None:
                return raw
            # A previous error must not steer past center or against the new frame.
            if previous * raw < 0:
                previous = 0.
            return previous + alpha * (raw - previous)

        prior = self.filtered if self.filtered is not None else (None, None)
        self.filtered = tuple(filter_axis(raw, previous) for raw,previous in zip(error,prior))

        def step(e, angle, limit, sign):
            if self.fixed_camera:
                # A stationary camera measures target position, not pointing error.
                center = (limit[0] + limit[1]) / 2
                target = center + e * sign * (limit[1] - limit[0]) / 2
                max_delta = min(3., 24. * elapsed)
                delta = max(-max_delta, min(max_delta, target - angle))
                return max(limit[0], min(limit[1], angle + delta))
            rate = min(24., 72. * max(0., abs(e)-deadband))
            # Keep larger corrections from breaking inter-frame box overlap.
            delta = math.copysign(min(3., rate * elapsed), e) * sign
            return max(limit[0],min(limit[1],angle+delta))
        self.pan=step(self.filtered[0],self.pan,LIMITS[0],pan_sign)
        self.tilt=step(self.filtered[1],self.tilt,LIMITS[1],tilt_sign)
        self.state = 'centered' if all(abs(e) <= deadband for e in error) else 'tracking'
        return round(self.pan), round(self.tilt)


class FaceDetector:
    def __init__(self):
        import cv2
        from pathlib import Path
        self.cv2 = cv2
        # OpenCV's Windows filename loader cannot reliably open Unicode paths.
        xml = Path(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml').read_text(encoding='utf-8')
        storage = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        self.detector = cv2.CascadeClassifier()
        try:
            self.detector.read(storage.getFirstTopLevelNode())
        finally:
            storage.release()
        if self.detector.empty():
            raise RuntimeError('OpenCV face cascade is unavailable')

    def detect(self, frame):
        cv2=self.cv2
        h,w=frame.shape[:2]
        scale=min(1.0,640/max(h,w))
        small=cv2.resize(frame,(round(w*scale),round(h*scale))) if scale<1 else frame
        gray=cv2.equalizeHist(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY))
        faces=self.detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,minSize=(32,32))
        return [tuple(float(v)/scale for v in (x,y,x+width,y+height)) for x,y,width,height in faces]
