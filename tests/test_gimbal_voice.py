import copy
import unittest

from companion.gimbal_console import GimbalSession, ControlLink
from companion.voice_control import VoiceWindow, parse_voice_control, COMMANDS
from companion.native_usb import NativeStatusDecoder
from test_gimbal import FakeLink
from test_gimbal_patrol import store


def item(now=1000, **changes):
    data=dict(v=1,frames=10,type=0,id=0,last_ms=500,now_ms=now,
              stops=0,applied=0,alarms=0,errors=0,dropped=0,epoch=3)
    data.update(changes)
    return dict(data=data,receivedMs=now)


class WindowTests(unittest.TestCase):
    def armed(self, at=1000):
        w=VoiceWindow()
        w.arm(item(at),at,client='owner',route={},revision=0,source='device')
        return w

    def poll(self,w,at=1100,**changes):
        return w.poll(item(at,**changes),at,revision=0,epoch=3)

    def test_parser_strict_schema(self):
        data=item()['data']
        wire=('voice_cmd '+' '.join(f'{k}={v}' for k,v in data.items())+'\r\n').encode()
        self.assertEqual(parse_voice_control(wire),data)
        for bad in [wire+b' v=1',wire.replace(b'v=1',b'v=2'),wire.replace(b'type=0',b'type=256'),
                    wire.replace(b'id=0',b'id=-1'),wire.replace(b'frames=10',b'frames=4294967296'),
                    wire.replace(b' id=0',b''),wire+b'\xff',b'voice_rx v=2']:
            with self.subTest(bad=bad),self.assertRaises((ValueError,UnicodeError)):
                parse_voice_control(bad)

    def test_baseline_no_replay_and_exact_one_new_frame(self):
        w=self.armed(); self.assertIsNone(self.poll(w))
        for at in (1200,1300): self.assertIsNone(self.poll(w,at))
        result=self.poll(w,1400,frames=11,id=0xA1,last_ms=1399)
        self.assertEqual(result,dict(action='preset_apply',slot='1'))
        self.assertIsNone(self.poll(w,1500,frames=11,id=0xA1,last_ms=1399))
        self.assertTrue(w.armed)

    def test_gap_reset_old_frame_drop_error_stop_all_close(self):
        for changes in [dict(frames=12,id=0xA1,last_ms=1100),dict(frames=9),
                dict(frames=11,id=0xA1,last_ms=600),*[ {key:1} for key in ('stops','applied','alarms','errors','dropped')]]:
            with self.subTest(changes=changes):
                w=self.armed(); self.assertIsNone(self.poll(w,**changes)); self.assertFalse(w.armed)

    def test_idle_heartbeat_window_and_no_extension(self):
        w=self.armed()
        with self.assertRaises(ValueError): w.arm(item(),1100,client='x',route={},revision=0,source='device')
        self.assertEqual(w.started,1000)
        for at in range(1100,31000,100):
            w.heartbeat=at; self.assertIsNone(self.poll(w,at))
        w.heartbeat=31000; self.assertIsNone(self.poll(w,31000)); self.assertFalse(w.armed)
        w=self.armed(); self.assertIsNone(self.poll(w,1600)); self.assertFalse(w.armed)

    def test_metadata_conflicts_and_stale_local_snapshot(self):
        for at,changes in [(1000,dict(id=0xA1)),(1100,dict(id=0xA1)),(1500,{})]:
            with self.subTest(at=at,changes=changes):
                w=self.armed(); self.poll(w,at,**changes); self.assertFalse(w.armed)
        w=self.armed(); w.poll(item(),1400,revision=0,epoch=3); self.assertFalse(w.armed)
        w=self.armed(); w.poll(item(1100),1100,revision=1,epoch=3); self.assertFalse(w.armed)

    def test_only_type_zero_extensions_legacy_preemption(self):
        for kind,code,armed in [(255,0xA1,True),(0,0x5B,True),(0,0x60,True),(0,4,False),(0,0x26,False)]:
            with self.subTest(kind=kind,code=code):
                w=self.armed(); self.assertIsNone(self.poll(w,frames=11,type=kind,id=code,last_ms=1100))
                self.assertEqual(w.armed,armed)

    def test_epoch_transition_only_for_own_ack_and_wrap(self):
        for owned in (False,True):
            w=self.armed(); w.poll(item(1100,epoch=4),1100,revision=0,epoch=4,own_active_lease=owned)
            self.assertEqual(w.armed,owned)
        w=self.armed(); self.poll(w,epoch=4); self.assertFalse(w.armed)
        at=0xfffffff0; w=self.armed(at)
        wire=item(at+100,now_ms=84,frames=11,id=0xA8,last_ms=80)
        self.assertEqual(w.poll(wire,at+100,revision=0,epoch=3)['action'],'center')


class SessionTests(unittest.TestCase):
    def command(self,s,action,at=1000,client='owner',**extra):
        s.command(dict(action=action,client=client,generation=s.generation,**extra),at)

    def arm(self,s,at=1000,**extra):
        values=dict(slots=['1','2'],dwell=500,pauseOnFace=False); values.update(extra)
        self.command(s,'voice_arm',at,**values)

    def session(self,hardware=False):
        link=FakeLink() if hardware else None
        s=GimbalSession(link,presets=store())
        if link: link.voice_snapshot=item(); s.tick(1000)
        return s

    def inject(self,s,code,at=1100):
        self.command(s,'voice_simulate',at,id=code)

    def test_preview_presets_center_once_and_stop_closes(self):
        s=self.session(); self.arm(s); self.inject(s,0xA1)
        self.assertEqual((s.mode,s.pan,s.tilt),('manual',95,90)); start=s.started
        for at in (1200,1300): self.command(s,'heartbeat',at); s.tick(at)
        self.assertEqual(s.started,start)
        self.inject(s,0xA8,1400); self.assertEqual(s.pan,90); self.assertEqual(s.started,start)
        self.inject(s,1,1500); self.assertEqual(s.mode,'idle'); self.assertFalse(s.voice_window.armed)
        with self.assertRaises(ValueError): self.inject(s,0xA1,1600)
        self.assertIsNone(s.link)

    def test_preview_patrol_pause_resume_original_lease_deadline(self):
        s=self.session(); self.arm(s); self.inject(s,0xA5); s.tick(1100)
        lease=s.lease; start=s.started
        self.inject(s,0xA6,1200); self.assertTrue(s.patrol.snapshot(1200)['paused'])
        self.command(s,'heartbeat',1400); s.tick(1400)
        self.inject(s,0xA7,1500); self.assertFalse(s.patrol.snapshot(1500)['paused'])
        self.assertEqual((s.lease,s.started),(lease,start))

    def test_auto_requires_fresh_image_and_failure_stops_owned_motion(self):
        s=self.session(); self.arm(s); self.inject(s,0xA1)
        self.inject(s,0xA4,1200)
        self.assertEqual(s.mode,'idle'); self.assertFalse(s.voice_window.armed)
        self.assertIn('画面',s.voice_window.last_command['result'])
        self.arm(s,1300); s.frame([(260,180,380,300)],640,480,1350,1350)
        self.inject(s,0xA4,1400); self.assertEqual(s.mode,'auto')

    def test_owner_schema_generation_and_hardware_injection_block(self):
        s=self.session(); self.arm(s)
        for action,fields in [('voice_simulate',dict(id=0xA1)),('voice_disarm',{})]:
            with self.assertRaises(ValueError): self.command(s,action,1100,client='other',**fields)
        with self.assertRaises(ValueError): s.command(dict(action='voice_disarm',client='owner',generation=-1),1100)
        with self.assertRaises(ValueError): self.inject(s,True)
        with self.assertRaises(ValueError): self.command(s,'voice_arm',slots=[],dwell=True,pauseOnFace=False)
        s=self.session(True); self.arm(s)
        with self.assertRaisesRegex(ValueError,'禁止'): self.inject(s,0xA1)
        self.assertEqual(s.link.writes,[])

    def test_sticky_legacy_status_never_enables_extension(self):
        s=self.session(True); s.link.voice_snapshot=None
        s.status['voice']=0xA1
        with self.assertRaises(ValueError): self.arm(s)
        s.tick(1100); self.assertEqual(s.link.writes,[]); self.assertEqual(s.mode,'idle')

    def test_hardware_frame_action_own_ack_then_native_stop_no_global_stop(self):
        s=self.session(True); self.arm(s)
        s.link.voice_snapshot=item(1100,frames=11,id=0xA1,last_ms=1090)
        s.tick(1100); self.assertEqual(s.link.writes[-1]['op'],1); lease=s.lease
        s.link.status.update(lease=lease,epoch=4,enabled=1)
        s.link.voice_snapshot=item(1200,epoch=4,frames=11,id=0xA1,last_ms=1090)
        self.command(s,'heartbeat',1200); s.tick(1200); self.assertTrue(s.voice_window.armed)
        writes=len(s.link.writes)
        s.link.status.update(lease=0,epoch=5,enabled=0)
        s.link.voice_snapshot=item(1300,epoch=5,frames=12,id=1,last_ms=1290,stops=1,applied=1)
        s.tick(1300); self.assertEqual(s.mode,'idle'); self.assertFalse(s.voice_window.armed)
        self.assertEqual(len(s.link.writes),writes)

    def test_data_loss_closes_window_and_cancels_owned_lease(self):
        s=self.session(True); self.arm(s)
        s.link.voice_snapshot=item(1100,frames=11,id=0xA1,last_ms=1090); s.tick(1100)
        s.link.status['lease']=s.lease
        self.command(s,'heartbeat',1400); s.tick(1400)
        self.command(s,'heartbeat',1500); s.tick(1500)
        self.assertFalse(s.voice_window.armed); self.assertEqual(s.mode,'idle')
        self.assertEqual(s.link.writes[-1]['op'],3)

    def test_other_source_takeover_not_cancelled_and_no_resume(self):
        s=self.session(True); self.arm(s)
        s.link.voice_snapshot=item(1100,frames=11,id=0xA1,last_ms=1090); s.tick(1100)
        s.link.status.update(lease=999,epoch=4)
        s.link.voice_snapshot=item(1200,frames=12,id=4,last_ms=1190,epoch=4)
        count=len(s.link.writes); s.tick(1200)
        self.assertEqual(s.mode,'idle'); self.assertEqual(len(s.link.writes),count)
        self.assertFalse(s.voice_window.armed)

    def test_preset_revision_and_heartbeat_loss_close_even_idle(self):
        s=self.session(); self.arm(s); s.presets.change('3',None); s.tick(1100)
        self.assertFalse(s.voice_window.armed)
        self.arm(s,1200); s.tick(1800); self.assertFalse(s.voice_window.armed)
        self.assertEqual(s.mode,'idle')


class LinkTests(unittest.TestCase):
    def test_native_decoder_separates_associated_topic_from_sticky_voice(self):
        wire=('voice_cmd '+' '.join(f'{k}={v}' for k,v in item()['data'].items())).encode()
        decoder=NativeStatusDecoder()
        decoded=decoder.feed(wire+b'\n')
        self.assertEqual(decoded[0]['type'],'voiceControl')
        self.assertEqual(decoded[0]['data'],item()['data'])
        self.assertEqual(decoder.feed(b'voice_cmd v=2\n'),[])

    def test_control_link_parses_and_rejects_corrupt_association(self):
        class Serial:
            def read(self,n): value=self.chunk; self.chunk=b''; return value
        link=ControlLink.__new__(ControlLink)
        from companion.control_feedback import ControlFeedback
        link.serial=Serial(); link.buffer=bytearray(); link.feedback=ControlFeedback()
        link.status=None; link.received=0; link.voice_snapshot=None
        wire=('voice_cmd '+' '.join(f'{k}={v}' for k,v in item()['data'].items())+'\n').encode()
        link.serial.chunk=wire[:30]; link.poll(1000); self.assertIsNone(link.voice_snapshot)
        link.serial.chunk=wire[30:]; link.poll(1100); self.assertEqual(link.voice_snapshot['data']['frames'],10)
        link.serial.chunk=b'voice_cmd v=1\n'; link.poll(1200); self.assertIsNone(link.voice_snapshot)


if __name__=='__main__': unittest.main()
