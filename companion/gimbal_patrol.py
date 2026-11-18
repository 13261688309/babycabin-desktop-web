"""One-pass bounded target-angle route. No device access or mechanical position claims."""
import copy
import math

from .gimbal_presets import DEFAULT_LIMITS


class PatrolPlan:
    MAX_MS = 10000
    RATE_DEG_PER_SECOND = 10
    DWELLS = (500, 1000, 1500)

    def __init__(self, store, slots, dwell, *, pan, tilt, pause_on_face, now, max_ms=MAX_MS):
        self.max_ms = max_ms
        if (type(slots) is not list or not 2 <= len(slots) <= 3
                or any(type(slot) is not str or slot not in ('1', '2', '3') for slot in slots)
                or len(set(slots)) != len(slots)):
            raise ValueError('请选择2至3个不同的已保存预置位')
        if type(dwell) is not int or dwell not in self.DWELLS or type(pause_on_face) is not bool:
            raise ValueError('停留时间请选择0.5、1或1.5秒')
        limits = getattr(store, "limits", DEFAULT_LIMITS)
        if any(type(value) is not int or not low <= value <= high
               for value, (low, high) in zip((pan, tilt), limits)):
            raise ValueError('起始目标角度无效')
        self.points = []
        previous = (pan, tilt)
        total = 0
        for slot in slots:
            value = store.get(slot)
            target = (value['pan'], value['tilt'])
            travel = math.ceil(max(abs(target[i]-previous[i]) for i in (0, 1))*100)
            self.points.append(dict(slot=slot, name=value['name'], pan=target[0], tilt=target[1],
                origin=previous, fromMs=total, travelMs=travel, untilMs=total+travel+dwell))
            total += travel+dwell
            previous = target
        if total > self.max_ms:
            raise ValueError(f'目标路线预计{total/1000:.1f}秒，超过{self.max_ms/1000:g}秒；请缩短角差或停留时间')
        self.planned_ms, self.dwell = total, dwell
        self.started = now
        self.paused_at = None
        self.paused_ms = 0
        self.ended_elapsed = None
        self.active = True
        self.state = 'running'
        self.phase = 'moving'
        self.index = 0
        self.pause_reason = None
        self.reason = '巡航目标角度进行中'
        self.target = (pan, tilt)
        self.pause_on_face = pause_on_face
        self.face_frames = 0
        self.face_capture = None

    def elapsed(self, now):
        if self.ended_elapsed is not None: return self.ended_elapsed
        at = self.paused_at if self.paused_at is not None else now
        return max(0, at-self.started-self.paused_ms)

    def complete(self, now):
        return self.elapsed(now) >= self.planned_ms

    def advance(self, now):
        if not self.active or now-self.started>=self.max_ms or self.complete(now): return None
        if self.paused_at is not None: return self.target
        elapsed = self.elapsed(now)
        for index, point in enumerate(self.points):
            if elapsed < point['untilMs']:
                self.index = index
                relative = elapsed-point['fromMs']
                self.phase = 'moving' if relative < point['travelMs'] else 'dwell'
                portion = min(1, relative/point['travelMs']) if point['travelMs'] else 1
                self.target = tuple(round(point['origin'][i]+(point[axis]-point['origin'][i])*portion)
                                    for i, axis in enumerate(('pan', 'tilt')))
                return self.target
        return None

    def pause(self, now, reason='manual'):
        if not self.active or self.paused_at is not None or self.advance(now) is None:
            raise ValueError('巡航当前不能暂停')
        self.paused_at = now
        self.state = 'paused'
        self.pause_reason = reason
        self.reason = '发现人脸，已暂停目标进度，需人工继续' if reason == 'face' else '已暂停目标进度；10秒截止时间仍继续'

    def resume(self, now):
        if not self.active or self.paused_at is None or now-self.started>=self.max_ms:
            raise ValueError('巡航当前未暂停')
        self.paused_ms += now-self.paused_at
        self.paused_at = None
        self.state = 'running'
        self.pause_reason = None
        self.face_capture = None
        self.face_frames = 0
        self.reason = '已继续本轮，不延长原10秒截止时间'

    def face_seen(self, found, captured, now):
        if not self.pause_on_face or not self.active or captured < self.started or self.complete(now):
            return False
        if self.face_capture is not None and captured <= self.face_capture:
            return False
        continuous = self.face_capture is not None and captured-self.face_capture < 400
        self.face_capture = captured
        self.face_frames = self.face_frames+1 if found and continuous else int(found)
        if self.paused_at is None and self.face_frames >= 3:
            self.pause(now, 'face')
            return True
        return False

    def end(self, now, reason, *, completed=False):
        if not self.active: return
        self.ended_elapsed = self.elapsed(now)
        self.active = False
        self.state = 'completed' if completed and self.complete(now) else 'stopped'
        self.reason = reason

    def snapshot(self, now):
        return dict(active=self.active, state=self.state, phase=self.phase,
            paused=self.paused_at is not None, pauseReason=self.pause_reason, reason=self.reason,
            index=self.index, points=copy.deepcopy(self.points), dwellMs=self.dwell,
            plannedMs=self.planned_ms, routeElapsedMs=self.elapsed(now),
            routeRemainingMs=max(0, self.planned_ms-self.elapsed(now)),
            pauseOnFace=self.pause_on_face, target=list(self.target),
            deadlineRemainingMs=max(0, self.max_ms-(now-self.started)) if self.active else 0,
            targetRateDegPerSecond=self.RATE_DEG_PER_SECOND, physicalArrivalMeasured=False)
