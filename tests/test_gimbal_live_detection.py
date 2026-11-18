from types import SimpleNamespace
import unittest

from tools.check_gimbal_live_detection import Preview, paired_observation


def state(generation=1, armed=False):
    return dict(profile='GB01-idle-low', mode='idle', controlState='offline',
                tiltReady=armed, status=None, camera=dict(generation=generation, state='streaming'))


class LiveDetectionTests(unittest.TestCase):
    def run_samples(self, samples, scene='front', states=None, new_ms=5):
        now = [0.]
        remaining = iter(samples)
        requested = []
        inputs = dict(haar=[], yunet=[])

        def get_preview(after):
            requested.append(after)
            sample = next(remaining, None)
            now[0] += .01 if sample else 1
            return sample

        def decode(data):
            return SimpleNamespace(shape=(240, 320, 3), data=data)

        def detector(name, duration):
            def detect(frame):
                inputs[name].append(frame.data)
                now[0] += duration/1000
                if name == 'yunet' or frame.data != b'profile':
                    return [(90., 60., 150., 120.)]
                return []
            return SimpleNamespace(detect=detect)

        result = paired_observation(.5, scene, (lambda: next(states)) if states else state,
            get_preview, decode, dict(haar=detector('haar', 3), yunet=detector('yunet', new_ms)),
            clock=lambda: now[0], sleep=lambda _: None)
        return result, requested, inputs

    def test_same_fresh_frames_with_separate_counts_and_simulated_lock(self):
        data = [Preview(1, 1000+i*100, age, name) for i, (age, name) in enumerate(
            [(75, b'front'), (80, b'profile'), (85, b'profile'), (400, b'old')])]
        result, requested, inputs = self.run_samples(data)
        self.assertEqual(requested, [0, 1000, 1100, 1200, 1300])
        self.assertEqual(inputs['haar'], inputs['yunet'])
        self.assertEqual(inputs['haar'], [b'front', b'profile', b'profile'])
        self.assertEqual(result['missing']['staleFrames'], 1)
        self.assertFalse(result['enoughPairs'])
        self.assertEqual(result['results']['haar']['exactlyOneBoxWithinDeadlineRate'], .3333)
        self.assertEqual(result['results']['yunet']['exactlyOneBoxWithinDeadlineRate'], 1.)
        self.assertEqual(result['results']['yunet']['simulatedLockFrames'], 1)

    def test_empty_scene_boxes_are_unexpected_and_late_box_is_counted_separately(self):
        sample = Preview(1, 1000, 375, b'profile')
        result, _, _ = self.run_samples([sample], scene='empty', new_ms=20)
        self.assertEqual(result['results']['haar']['unexpectedBoxRate'], 0.)
        self.assertEqual(result['results']['yunet']['unexpectedBoxRate'], 1.)
        self.assertEqual(result['results']['yunet']['pastDeadlineFrames'], 1)
        self.assertEqual(result['results']['yunet']['oneBoxWithinDeadlineFrames'], 0)

    def test_discards_phase_when_camera_or_control_state_changes(self):
        with self.assertRaisesRegex(ValueError, 'Camera generation changed'):
            self.run_samples([], states=iter([state(), state(generation=2)]))
        with self.assertRaisesRegex(ValueError, 'wiring enable unchecked'):
            self.run_samples([], states=iter([state(), state(armed=True)]))
        with self.assertRaisesRegex(ValueError, 'Preview generation changed'):
            self.run_samples([Preview(2, 1000, 50, b'front')])
