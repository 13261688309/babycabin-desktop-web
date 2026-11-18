import unittest

from tools.check_gimbal_frame_age import check_idle, observe, summarize


def state(age=180, frame=42, generation=1):
    return dict(profile='GB01-idle-low', mode='idle', controlState='idle',
                tiltReady=False, status=dict(pwm=0, lease=0, fault=0, motor=0, rock=0),
                camera=dict(state='streaming', generation=generation),
                frameId=frame, frameAgeMs=age, frameFresh=age < 400)


class FrameAgeCheckTests(unittest.TestCase):
    def test_stale_boundary_missing_and_repeated_frames(self):
        missing = state()
        missing['camera']['state'] = 'error'
        result = summarize([state(110), state(400), state(270, 43), missing])
        self.assertEqual(result['observedDistinctFrames'], 2)
        self.assertEqual(result['staleSamples'], 1)
        self.assertEqual(result['missingSamples'], 1)
        self.assertEqual(result['p95AgeMs'], 400)

    def test_analysis_gaps_and_first_seen_age_ignore_duplicate_frames(self):
        samples = [state(80, 100), state(180, 100), state(40, 250), state(60, 670)]
        for sample, count in zip(samples, (3, 3, 5, 8)):
            sample['camera']['frames'] = count
        result = summarize(samples)
        self.assertEqual(result['capturedFrameCounterIncrease'], 5)
        self.assertEqual(result['observedAnalysisGapP95Ms'], 420)
        self.assertEqual(result['observedAnalysisGapMaxMs'], 420)
        self.assertEqual(result['gapsAtLeastFreshnessLimit'], 1)
        self.assertEqual(result['firstObservedAnalysisFrameAgeP95Ms'], 80)

    def test_face_counts_include_only_distinct_fresh_frames(self):
        samples = [dict(state(age, frame), faceCount=count) for age, frame, count in
                   [(80, 100, 1), (180, 100, 1), (400, 200, 0), (30, 300, 3), (40, 400, 0)]]
        self.assertEqual(summarize(samples)['freshFaceFrames'],
                         dict(noFace=1, oneFace=1, multipleFaces=1))

    def test_transport_transitions_include_fallback_and_recovery(self):
        samples = []
        for index, (transport, status, reason) in enumerate([
                ('mjpeg', 'streaming', None), ('mjpeg', 'streaming', None),
                ('snapshot', 'error', 'TimeoutError'), ('snapshot', 'streaming', 'TimeoutError'),
                ('mjpeg', 'streaming', None)]):
            sample = state(frame=100 + index * 100)
            sample['sampleMs'] = index * 100
            sample['camera'].update(transport=transport, state=status, fallbackReason=reason)
            samples.append(sample)
        result = summarize(samples)
        self.assertEqual(result['transportStateSamples'], {'mjpeg/streaming': 3, 'snapshot/error': 1,
                                                          'snapshot/streaming': 1})
        self.assertEqual([row['atMs'] for row in result['transportTransitions']], [0, 200, 300, 400])
        self.assertEqual(result['transportTransitions'][1]['reason'], 'TimeoutError')

    def test_requires_disarmed_idle_controller_with_fresh_status(self):
        check_idle(state())
        for change in (dict(tiltReady=True), dict(mode='auto'), dict(status=None),
                       dict(status={**state()['status'], 'lease': 1}),
                       dict(status={**state()['status'], 'fault': 1})):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_idle({**state(), **change})

    def test_aborts_on_camera_generation_change(self):
        samples = iter([state(), state(generation=2)])
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
        with self.assertRaisesRegex(ValueError, 'generation changed'):
            observe(1, lambda: next(samples), clock=lambda: now[0], sleep=sleep)

    def test_camera_only_observation_does_not_require_controller(self):
        offline = {**state(), 'controlState': 'offline', 'status': None}
        check_idle(offline)
        with self.assertRaises(ValueError):
            check_idle({**offline, 'tiltReady': True})

    def test_no_stream_has_no_latency_statistics(self):
        for metric in ('maxAgeMs', 'observedAnalysisGapP95Ms', 'capturedFrameCounterIncrease'):
            self.assertIsNone(summarize([])[metric])
        missing = state()
        missing['camera']['state'] = 'disconnected'
        with self.assertRaisesRegex(ValueError, 'Connect the camera'):
            observe(1, lambda: missing)
