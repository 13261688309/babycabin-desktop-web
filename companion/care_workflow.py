"""Session-local care workflow. Reminders and follow-up never issue commands."""
import copy
from collections import deque

from .alert_inbox import AlertInbox, utc_now
from .state import TemporalState, intersection_over_union


def confirmation():
    return TemporalState(confidence=.80, confirm_ms=3000, min_frames=5, max_gap_ms=1500)


class CareWorkflow:
    FOLLOWUP_MS = 20000

    def __init__(self):
        self.inbox = AlertInbox(capacity=50)
        self.visual = confirmation()
        self.captured = None
        self.target = None
        self.size = None
        self.minimum_capture = 0
        self.run = None
        self.post = None
        self.events = deque(maxlen=50)
        self.next_event = 1
        self.known_alerts = {}

    def event(self, kind, text, now):
        self.events.append(dict(id=self.next_event, kind=kind, text=text,
                                atMs=now, utc=utc_now()))
        self.next_event += 1

    def _finish(self, state, reason, now):
        self.run.update(state=state, reason=reason, endedMs=now)
        if state in ('insufficient', 'unavailable', 'interrupted'):
            self.run['after'] = 'Unknown'
        self.event('followup', reason, now)
        self.post = None

    def interrupt(self, reason, now):
        if self.run and self.run['state'] in ('executing', 'following'):
            self._finish('interrupted', reason, now)

    def invalidate(self, reason, now):
        self.minimum_capture = now
        self.visual.invalidate(reason)
        self.captured = self.target = self.size = None
        if self.run and self.run['state'] == 'executing':
            self._continuous = False
        if self.run and self.run['state'] == 'following':
            self._finish('insufficient', '相机或模型输入已重置，观察结果不足', now)

    def begin(self, now, *, source, hardware):
        self.interrupt('新动作替代了上次观察', now)
        fresh = self.captured is not None and 0 <= now-self.captured < 1500
        before = self.visual.expire(now).state if fresh else 'Unknown'
        self.run = dict(state='executing', source=source, hardware=hardware,
                        before=before, after='Unknown', samples=0, confirmedSamples=0,
                        requestedMs=now, startedMs=None, endedMs=None,
                        reason='等待本次限时摇摆结束', controllerAccepted=False,
                        physicalMotionMeasured=False)
        self._target = self.target if fresh else None
        self._target_capture = self.captured
        self._continuous = self._target is not None
        self.event('request', '请求一次限时摇摆' if hardware else '预演一次限时摇摆（无硬件输出）', now)

    def end(self, now, *, completed, accepted):
        if not self.run or self.run['state'] != 'executing':
            return
        self.run['controllerAccepted'] = accepted
        if not completed:
            self._finish('interrupted', '本次摇摆已中断，未形成完整观察流程', now)
            return
        if self.run['hardware'] and not accepted:
            self._finish('unavailable', '未确认主控接纳本次会话，不能形成动作后观察', now)
            return
        if (not self._continuous or self.captured is None
                or not 0 <= now-self.captured < 1500):
            self._finish('unavailable', '缺少连续单目标画面，不能形成动作后观察', now)
            return
        self.post = confirmation()
        self.run.update(state='following', startedMs=now, reason='只读观察20秒，不再自动动作')
        self.event('followup', '限时控制已结束，开始20秒只读观察（非机械停机证明）', now)

    def observe(self, detections, captured, now, width, height):
        if captured < self.minimum_capture or self.captured is not None and captured <= self.captured:
            return
        same_size = self.size is None or self.size == (width, height)
        if not same_size:
            self.visual.invalidate('dimensions_changed')
        self.size = (width, height)
        valid = (0 <= now-captured < 1500 and all(item.valid()
                 and item.xyxy[2] <= width and item.xyxy[3] <= height for item in detections))
        self.captured = captured
        if not valid:
            self.visual.invalidate('stale_or_invalid')
            self.target = None
        else:
            self.visual.update(detections, captured)
            accepted = [item for item in detections if item.confidence >= .80]
            self.target = accepted[0] if len(accepted) == 1 else None
        if not self.run or self.run['state'] not in ('executing', 'following'):
            return
        # Spatial continuity is not identity recognition. Never compare across a gap or target jump.
        continuous = (same_size and self._continuous and self.target is not None and self._target is not None
                      and self._target_capture is not None
                      and 0 < captured-self._target_capture <= 1500
                      and intersection_over_union(self._target, self.target) >= .30)
        self._continuous = bool(continuous)
        self._target, self._target_capture = self.target, captured
        if self.run['state'] == 'following':
            if not continuous:
                self._finish('insufficient', '观察目标丢失、跳变或不明确，结果不足；未重新启动动作', now)
            elif captured >= self.run['startedMs']:
                # Only new captures made after the end of the action count; polling adds no samples.
                observation = self.post.update(detections, captured)
                self.run['samples'] += 1
                if observation.state != 'Unknown':
                    self.run['confirmedSamples'] += 1
                self.run['after'] = observation.state

    def tick(self, now, *, video_fresh, device):
        fresh = (video_fresh and self.captured is not None
                 and 0 <= now-self.captured < 1500)
        observation = self.visual.expire(now).to_dict() if fresh else dict(state='Unknown', reason='stale')
        self.inbox.observe(dict(live=fresh, record=dict(sourceKind='network',
            frameAgeMs=now-self.captured if fresh else None,
            video=dict(connectionState='streaming' if fresh else 'unavailable'),
            observation=observation, device=dict(device, online=device.get('connected') is True))), now, utc_now())
        for item in self.inbox.items:
            previous = self.known_alerts.get(item['id'])
            if previous is None:
                self.event('reminder', item['title'], now)
            elif previous != 'recovered' and item['condition'] == 'recovered':
                self.event('recovery', item['title']+'：条件已恢复', now)
            self.known_alerts[item['id']] = item['condition']
        self.known_alerts = {item['id']: self.known_alerts[item['id']] for item in self.inbox.items}
        if not self.run or self.run['state'] != 'following':
            return
        if not fresh:
            self._finish('insufficient', '观察画面中断或过期，结果不足；未重新启动动作', now)
        elif now-self.run['startedMs'] >= self.FOLLOWUP_MS:
            after = self.post.expire(now).state
            self.run['after'] = after
            if after == 'Unknown' or self.run['confirmedSamples'] < 5:
                self._finish('insufficient', '观察结束，有效连续识别不足', now)
            elif after == 'Cry':
                self._finish('cry_persists', '观察结束，视觉标签仍为Cry，请人工查看；不会再次自动摇摆', now)
            elif self.run['before'] == 'Cry':
                self._finish('label_changed', '观察结束，视觉标签已变化；不等于情绪改善或安抚有效', now)
            else:
                self._finish('observed', '观察结束，已取得连续视觉标签；不判定安抚效果', now)

    def acknowledge(self, alert_id, now):
        if type(alert_id) is not str or not 1 <= len(alert_id) <= 32:
            raise ValueError('Invalid reminder id')
        item = next((item for item in self.inbox.items if item['id'] == alert_id), None)
        if item is None:
            raise ValueError('提醒不存在或已移出本场列表')
        already = item['acknowledged']
        self.inbox.acknowledge(alert_id, utc_now())
        if not already:
            self.event('ack', item['title']+'：已查看（不代表条件恢复）', now)

    def snapshot(self, now):
        run = copy.deepcopy(self.run)
        if run:
            run['remainingMs'] = max(0, self.FOLLOWUP_MS-(now-run['startedMs'])) if run['state'] == 'following' else 0
        return dict(alertInbox=self.inbox.snapshot(), followup=run,
                    events=copy.deepcopy(list(self.events)), sessionOnly=True,
                    automaticActionOnAcknowledgement=False, automaticRetry=False)
