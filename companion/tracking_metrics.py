"""Read-only image-space metrics for one explicit tracking run; no device access."""
import copy
import math

from .gimbal import valid_face


class TrackingMetrics:
    FRESH_MS = 400
    CENTER_PERCENT = 5

    def __init__(self):
        self.run = None
        self.sequence = 0
        self.last_capture = None
        self.previous_locked = None
        self.size = None
        self.sum_error = 0.0

    def begin(self, now, *, hardware, selected):
        self.sequence += 1
        self.last_capture = self.previous_locked = self.size = None
        self.sum_error = 0.0
        self.run = dict(id=self.sequence, active=True, hardwareConfigured=hardware,
            selection='selected' if selected else 'single', startedMs=now, endedMs=None,
            reason='等待连续三帧锁定', observedFrames=0, lockedFrames=0, centeredFrames=0,
            validTrackingMs=0, acquisitionMs=None, initialErrorPercent=None,
            meanErrorPercent=None, lastSample=None, physicalMotionMeasured=False,
            faceIdentityRecognized=False)

    def end(self, now, reason):
        if self.run and self.run['active']:
            self.run.update(active=False, endedMs=now, reason=reason)

    def observe(self, faces, width, height, captured, now, *, state, pan, tilt):
        if (not self.run or not self.run['active'] or type(captured) is not int
                or captured < self.run['startedMs'] or not 0 <= now-captured < self.FRESH_MS
                or self.last_capture is not None and captured <= self.last_capture):
            return
        self.last_capture = captured
        self.run['observedFrames'] += 1
        self.run['reason'] = state
        size = (width, height)
        dimensions_valid = type(width) is int and type(height) is int and width > 0 and height > 0
        if (not dimensions_valid or self.size is not None and size != self.size
                or state not in ('tracking', 'centered') or len(faces) != 1
                or not valid_face(faces[0], width, height)):
            self.previous_locked = None
            self.run['lastSample'] = None
            self.size = size
            return
        self.size = size
        box = faces[0]
        dx = ((box[0]+box[2])/2/width-.5)*100
        dy = ((box[1]+box[3])/2/height-.5)*100
        error = math.hypot(dx, dy)
        centered = abs(dx) <= self.CENTER_PERCENT+1e-9 and abs(dy) <= self.CENTER_PERCENT+1e-9
        if self.previous_locked is not None:
            gap = captured-self.previous_locked
            if 0 < gap < self.FRESH_MS:
                self.run['validTrackingMs'] += gap
        self.previous_locked = captured
        self.run['lockedFrames'] += 1
        self.run['centeredFrames'] += int(centered)
        self.sum_error += error
        if self.run['acquisitionMs'] is None:
            self.run['acquisitionMs'] = now-self.run['startedMs']
            self.run['initialErrorPercent'] = round(error, 3)
        self.run['meanErrorPercent'] = round(self.sum_error/self.run['lockedFrames'], 3)
        limits = []
        if pan in (60, 120) and abs(dx) > self.CENTER_PERCENT: limits.append('pan')
        if tilt in (75, 105) and abs(dy) > self.CENTER_PERCENT: limits.append('tilt')
        self.run['lastSample'] = dict(frameId=captured, box=list(box), dxPercent=round(dx, 3),
            dyPercent=round(dy, 3), errorPercent=round(error, 3), centered=centered, targetAngleLimits=limits)

    def snapshot(self, now):
        result = copy.deepcopy(self.run)
        if result is None: return None
        sample = result['lastSample']
        result['sampleFresh'] = bool(result['active'] and sample is not None
                                     and 0 <= now-sample['frameId'] < self.FRESH_MS)
        result['elapsedMs'] = max(0, (now if result['active'] else result['endedMs'])-result['startedMs'])
        frames = result['lockedFrames']
        result['centeredFramePercent'] = round(result['centeredFrames']/frames*100, 1) if frames else None
        return result

    def report(self, now):
        return dict(schemaVersion=1, kind='single-run-image-space-tracking-metrics',
            run=self.snapshot(now), definition=dict(
                displacement='Target center minus frame center, percent of full image width/height',
                centered='Both raw displacement axes within +/-5 percent; not filtered controller state',
                validTime='Sum of consecutive locked capture intervals strictly below 400ms; polling adds no time',
                meanError='Frame-weighted mean of hypot(horizontalPercent, verticalPercent)',
                scope='Image-space observation only, not mechanical motion, identity, detection accuracy or causal proof'),
            containsImages=False, containsCameraAddress=False, containsControlToken=False,
            mechanicalMotionQualified=False)
