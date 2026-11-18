import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from companion.gimbal import match_selected
from companion.gimbal_presets import PresetStore, validate_preset
from companion.gimbal_console import GimbalSession, ConsoleServer, milliseconds


LEFT = (50, 160, 170, 280)
RIGHT = (430, 180, 550, 300)
PRESET = dict(name='观察位', pan=105, tilt=95, panSign=-1, tiltSign=1)


class FeatureTests(unittest.TestCase):
    def command(self, session, action, now=1000, **values):
        session.command(dict(action=action, client='tab', generation=session.generation, **values), now)

    def selected(self):
        session = GimbalSession()
        session.frame([LEFT, RIGHT], 640, 480, 1000, 1000, b'first')
        self.command(session, 'select_target', frameId=1000, index=1)
        return session

    def test_presets_persist_and_loading_never_starts_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'presets.json'
            session = GimbalSession(presets=PresetStore(path))
            self.command(session, 'preset_save', slot='1', **PRESET)
            session.tick(1000)
            self.assertEqual(session.mode, 'idle')
            restored = GimbalSession(presets=PresetStore(path))
            self.assertEqual(restored.presets.get('1'), PRESET)
            self.assertEqual((restored.mode, restored.pan, restored.tilt), ('idle', 90, 90))
            self.command(restored, 'preset_apply', slot='1')
            self.assertEqual((restored.pan, restored.tilt, restored.pan_sign), (105, 95, -1))
            self.command(restored, 'preset_apply', now=1400, slot='1')
            self.assertEqual(restored.started, 1000)
            with self.assertRaises(ValueError): self.command(restored, 'preset_delete', slot='1')
            self.command(restored, 'stop')
            self.command(restored, 'preset_delete', slot='1')
            self.assertEqual(PresetStore(path).slots, {})

    def test_preset_validation_and_corrupt_file_is_preserved(self):
        for change in ({'pan': 121}, {'tilt': 74}, {'pan': True}, {'panSign': 0},
                       {'name': ''}, {'name': 'a' * 17}, {'name': 'a\nb'}, {'extra': 1}):
            with self.assertRaises(ValueError): validate_preset({**PRESET, **change})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            for content in ('not json', '{"version":1,"version":1,"slots":{}}',
                            '{"version":1,"slots":{"4":{}}}', ' ' * 8193):
                path.write_text(content, encoding='utf-8')
                store = PresetStore(path)
                self.assertIsNotNone(store.error)
                with self.assertRaises(ValueError): store.change('1', PRESET)
                self.assertEqual(path.read_text(encoding='utf-8'), content)

    def test_failed_persistence_is_atomic_and_snapshot_is_detached(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            store = PresetStore(path)
            store.change('1', PRESET)
            before = path.read_bytes()
            with patch('companion.gimbal_presets.os.replace', side_effect=OSError):
                with self.assertRaises(ValueError): store.change('1', {**PRESET, 'pan': 110})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(store.get('1')['pan'], 105)
            self.assertEqual(list(path.parent.glob('.gimbal-preset-*')), [])
            snapshot = store.snapshot()
            snapshot['slots']['1']['pan'] = 120
            self.assertEqual(store.get('1')['pan'], 105)

    def test_preset_reuses_mode_generation_and_ownership_guards(self):
        session = GimbalSession()
        generation = session.generation
        self.command(session, 'preset_save', slot='1', **PRESET)
        with self.assertRaises(ValueError):
            session.command(dict(action='preset_apply', client='tab', generation=generation, slot='1'), 1000)
        self.command(session, 'auto', panSign=1, tiltSign=1)
        with self.assertRaises(ValueError): self.command(session, 'preset_apply', slot='1')
        self.command(session, 'stop')
        self.command(session, 'preset_apply', slot='1')
        with self.assertRaises(ValueError):
            session.command(dict(action='preset_apply', client='another', generation=session.generation, slot='1'), 1200)
        self.command(session, 'heartbeat', now=10900)
        session.tick(11000)
        self.assertEqual(session.mode, 'idle')

    def test_hardware_preset_save_has_no_write_and_apply_requires_fresh_capability(self):
        class Link:
            def poll(self, now): return None
            def write(self, **values): raise AssertionError('Unexpected device write')
        session = GimbalSession(Link())
        self.command(session, 'preset_save', slot='1', **PRESET)
        session.tick(1000)
        with self.assertRaises(ValueError): self.command(session, 'preset_apply', slot='1')
        self.assertEqual(session.mode, 'idle')

    def test_selected_target_tracks_with_other_face_and_reordered_detections(self):
        session = self.selected()
        self.assertEqual(session.mode, 'idle')
        self.command(session, 'auto', panSign=1, tiltSign=1)
        for at in (1100, 1300, 1500):
            self.command(session, 'heartbeat', now=at)
            session.frame([RIGHT, LEFT], 640, 480, at, at)
            session.tick(at)
        self.assertEqual(session.mode, 'auto')
        self.assertGreater(session.pan, 90)
        self.assertEqual(session.selected_box, RIGHT)
        self.assertEqual(session.snapshot(1500)['selection']['mode'], 'selected')

    def test_selected_loss_stops_even_before_first_command_and_cannot_switch(self):
        session = self.selected()
        self.command(session, 'auto', panSign=1, tiltSign=1)
        session.frame([LEFT], 640, 480, 1100, 1100)
        self.assertEqual(session.mode, 'idle')
        for at in (1200, 1300, 1400): session.frame([RIGHT, LEFT], 640, 480, at, at)
        self.assertIsNone(session.selected_box)
        with self.assertRaises(ValueError): self.command(session, 'auto', now=1400, panSign=1, tiltSign=1)
        self.command(session, 'clear_target', now=1400)
        self.assertFalse(session.selection_required)

    def test_crossings_duplicates_invalid_boxes_and_gap_fail_closed(self):
        for faces in ([RIGHT, RIGHT], [RIGHT, (450, 180, 570, 300)],
                      [(0, 0, float('nan'), 10)], [LEFT]):
            self.assertIsNone(match_selected(RIGHT, faces, 640, 480)[0])
        for at, faces, width in ((1100, [RIGHT, (450, 180, 570, 300)], 640),
                                 (1400, [RIGHT, LEFT], 640), (1100, [RIGHT, LEFT], 800)):
            session = self.selected()
            self.command(session, 'auto', panSign=1, tiltSign=1)
            session.frame(faces, width, 480, at, at)
            self.assertEqual(session.mode, 'idle')

    def test_selection_exact_frame_index_staleness_and_mode_validation(self):
        session = self.selected()
        for values in (dict(frameId=999, index=1), dict(frameId=1000, index=2),
                       dict(frameId=True, index=1), dict(frameId=1000, index=False)):
            with self.assertRaises(ValueError): self.command(session, 'select_target', **values)
        with self.assertRaises(ValueError): self.command(session, 'select_target', now=1400, frameId=1000, index=1)
        self.command(session, 'manual', pan=100, tilt=90)
        with self.assertRaises(ValueError): self.command(session, 'select_target', frameId=1000, index=1)
        with self.assertRaises(ValueError): self.command(session, 'clear_target')

    def test_old_frame_selection_uses_unique_overlap_not_current_index(self):
        session = GimbalSession()
        session.frame([LEFT, RIGHT], 640, 480, 1000, 1000)
        session.frame([RIGHT, LEFT], 640, 480, 1100, 1100)
        self.command(session, 'select_target', now=1100, frameId=1000, index=1)
        self.assertEqual(session.selected_box, RIGHT)

    def test_frame_image_matches_requested_id_and_cache_is_bounded(self):
        session = GimbalSession()
        now = milliseconds()
        session.frame([LEFT], 640, 480, now - 30, now - 30, b'jpeg-a')
        session.frame([RIGHT], 640, 480, now - 10, now - 10, b'jpeg-b')
        server = ConsoleServer(session)
        server.start()
        try:
            def get(frame):
                connection = http.client.HTTPConnection('127.0.0.1', server.server.server_port, timeout=2)
                connection.request('GET', '/frame.jpg?frame=' + str(frame))
                response = connection.getresponse()
                result = response.status, response.read()
                connection.close()
                return result
            self.assertEqual(get(now - 30), (200, b'jpeg-a'))
            self.assertEqual(get(now - 10), (200, b'jpeg-b'))
            self.assertEqual(get(now - 1000)[0], 404)
            session.captured = None
            self.assertEqual(get(now - 10)[0], 404)
        finally:
            server.close()
        for at in range(now + 1, now + 100): session.frame([], 640, 480, at, at)
        self.assertEqual(len(session.frame_history), 32)


if __name__ == '__main__':
    unittest.main()
