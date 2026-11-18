import copy
import unittest

from companion.gimbal_console import GimbalSession
from companion.gimbal_camera import GimbalCamera
from companion.gimbal_patrol import PatrolPlan
from companion.gimbal_presets import PresetStore
from test_gimbal import FakeLink


def store():
    result=PresetStore()
    for slot,pan,tilt in [('1',95,90),('2',100,90),('3',90,95)]:
        result.change(slot,dict(name='观察位'+slot,pan=pan,tilt=tilt,panSign=1,tiltSign=1))
    return result


class PlanTests(unittest.TestCase):
    def plan(self, **changes):
        data=dict(pan=90,tilt=90,pause_on_face=False,now=1000)
        data.update(changes)
        return PatrolPlan(store(),['1','2'],500,**data)

    def test_single_pass_ramp_dwell_and_complete(self):
        plan=self.plan()
        self.assertEqual(plan.planned_ms,2000)
        self.assertEqual(plan.advance(1000),(90,90))
        self.assertEqual(plan.advance(1400),(94,90))
        self.assertEqual(plan.advance(1500),(95,90)); self.assertEqual(plan.phase,'dwell')
        self.assertEqual(plan.advance(2100),(96,90)); self.assertEqual(plan.index,1)
        self.assertEqual(plan.advance(2500),(100,90)); self.assertEqual(plan.phase,'dwell')
        self.assertIsNone(plan.advance(3000))
        plan.end(3000,'完成',completed=True)
        self.assertEqual(plan.state,'completed')
        self.assertIsNone(plan.advance(4000))

    def test_pause_holds_progress_and_resume_does_not_change_deadline(self):
        plan=self.plan(); plan.pause(1400)
        self.assertEqual(plan.advance(2000),(94,90))
        self.assertEqual(plan.elapsed(2000),400)
        plan.resume(2200)
        self.assertEqual(plan.advance(2300),(95,90))
        self.assertEqual(plan.started,1000)
        self.assertEqual(plan.snapshot(2300)['deadlineRemainingMs'],8700)

    def test_absolute_deadline_applies_while_paused(self):
        plan=self.plan(); plan.pause(1200)
        self.assertIsNone(plan.advance(11000))
        with self.assertRaises(ValueError): plan.resume(11000)
        plan.end(11000,'到期',completed=True)
        self.assertEqual(plan.state,'stopped')

    def test_validation_and_time_budget(self):
        source=store()
        for slots,dwell,face in [([],500,False),(['1'],500,False),(['1','1'],500,False),
                (['1',2],500,False),(['1','4'],500,False),(['1','2'],True,False),
                (['1','2'],499,False),(['1','2'],500,1),('12',500,False)]:
            with self.subTest(slots=slots,dwell=dwell,face=face),self.assertRaises(ValueError):
                PatrolPlan(source,slots,dwell,pan=90,tilt=90,pause_on_face=face,now=1000)
        source.change('1',dict(name='左端',pan=60,tilt=90,panSign=1,tiltSign=1))
        source.change('2',dict(name='右端',pan=120,tilt=90,panSign=1,tiltSign=1))
        with self.assertRaisesRegex(ValueError,'超过10秒'):
            PatrolPlan(source,['1','2'],1000,pan=90,tilt=90,pause_on_face=False,now=1000)

    def test_zero_distance_and_immutable_preset_snapshot(self):
        source=store()
        plan=PatrolPlan(source,['1','2'],500,pan=95,tilt=90,pause_on_face=False,now=1000)
        self.assertEqual(plan.advance(1000),(95,90)); self.assertEqual(plan.phase,'dwell')
        source.change('2',dict(name='改动',pan=120,tilt=105,panSign=-1,tiltSign=-1))
        self.assertEqual(plan.points[1]['pan'],100)
        snap=plan.snapshot(1000); snap['points'].clear()
        self.assertEqual(len(plan.points),2)

    def test_face_pause_requires_three_new_continuous_captures(self):
        plan=self.plan(pause_on_face=True)
        self.assertFalse(plan.face_seen(True,900,1000))
        self.assertFalse(plan.face_seen(True,1100,1100))
        self.assertFalse(plan.face_seen(True,1100,1200))
        self.assertFalse(plan.face_seen(True,1300,1300))
        self.assertTrue(plan.face_seen(True,1500,1500))
        self.assertEqual(plan.pause_reason,'face')
        self.assertFalse(plan.face_seen(False,1700,1700))
        self.assertTrue(plan.snapshot(1700)['paused'])
        plan.resume(1800); self.assertEqual(plan.face_frames,0)

    def test_face_gap_reset_and_end_boundary_is_not_exception(self):
        plan=self.plan(pause_on_face=True)
        for at in (1100,1500,1900): self.assertFalse(plan.face_seen(True,at,at))
        self.assertFalse(plan.snapshot(1900)['paused'])
        self.assertFalse(plan.face_seen(True,3000,3000))


class SessionTests(unittest.TestCase):
    def session(self,link=None):
        result=GimbalSession(link,presets=store())
        if link: result.tick(900)
        return result

    def command(self,session,action,at=1000,client='owner',**extra):
        session.command(dict(action=action,client=client,generation=session.generation,**extra),at)

    def start(self,s,at=1000,watch=False):
        self.command(s,'patrol',at,slots=['1','2'],dwell=500,pauseOnFace=watch)

    def test_preview_completes_once_without_camera_or_serial(self):
        s=self.session(); self.start(s)
        for at in range(1000,3200,100):
            self.command(s,'heartbeat',at); s.tick(at)
        self.assertEqual(s.mode,'idle'); self.assertEqual(s.patrol.state,'completed')
        self.assertFalse(s.hardware); self.assertIsNone(s.link)
        for at in (3300,4000): s.tick(at)
        self.assertEqual(s.mode,'idle')

    def test_pause_resume_use_same_lease_and_original_deadline(self):
        link=FakeLink(); s=self.session(link); self.start(s); s.tick(1000)
        lease=s.lease; link.status['lease']=lease; s.tick(1100)
        self.command(s,'patrol_pause',1200); s.tick(1200)
        target=(s.pan,s.tilt)
        self.command(s,'heartbeat',1400); s.tick(1400)
        self.assertEqual((s.pan,s.tilt),target)
        self.command(s,'patrol_resume',1500); s.tick(1500)
        self.assertEqual(s.lease,lease); self.assertEqual(s.started,1000)
        self.assertEqual([item['op'] for item in link.writes].count(1),1)
        self.assertTrue(all(item['lease']==lease for item in link.writes))
        self.assertEqual(link.writes[0]['duration'],10000)

    def test_paused_route_does_not_extend_absolute_ten_seconds(self):
        s=self.session(); self.start(s); self.command(s,'patrol_pause',1200)
        for at in range(1400,11001,200):
            self.command(s,'heartbeat',at); s.tick(at)
        self.assertEqual(s.mode,'idle'); self.assertEqual(s.patrol.state,'stopped')
        with self.assertRaises(ValueError): self.command(s,'patrol_resume',11001)

    def test_other_client_stale_generation_and_mode_switch_rejected(self):
        s=self.session(); self.start(s)
        for action,extra in [('patrol_pause',{}),('patrol_resume',{}),('manual',dict(pan=90,tilt=90)),
                              ('center',{}),('auto',dict(panSign=1,tiltSign=1)),('rock',dict(speed=10,duration=2000))]:
            with self.subTest(action=action),self.assertRaises(ValueError):
                self.command(s,action,1100,client='other',**extra)
        old=s.generation; self.command(s,'patrol_pause',1100)
        with self.assertRaises(ValueError):
            s.command(dict(action='patrol_resume',client='owner',generation=old),1200)
        with self.assertRaises(ValueError): self.command(s,'center',1200)
        with self.assertRaises(ValueError): self.command(s,'patrol',1200,slots=['1','2'],dwell=500,pauseOnFace=False)

    def test_stop_is_global_and_resume_cannot_restart(self):
        link=FakeLink(); s=self.session(link); self.start(s); s.tick(1000)
        self.command(s,'stop',1100,client='other'); s.tick(1100)
        self.assertEqual(s.mode,'idle'); self.assertEqual(link.writes[-1]['op'],0)
        with self.assertRaises(ValueError): self.command(s,'patrol_resume',1200)

    def test_voice_handover_and_link_loss_cancel_without_new_start(self):
        for variant in ('voice','link'):
            link=FakeLink(); s=self.session(link); self.start(s); s.tick(1000)
            link.status['lease']=s.lease; s.tick(1100)
            if variant=='voice': link.status['lease']=0; link.status['epoch']+=1
            else: link.status=None
            self.command(s,'heartbeat',1200); s.tick(1200)
            self.assertEqual(s.mode,'idle'); self.assertEqual(s.patrol.state,'stopped')
            self.assertEqual([item['op'] for item in link.writes].count(1),1)

    def test_busy_controller_and_protection_reject_before_motion(self):
        for field,value in [('lease',99),('trip',1)]:
            link=FakeLink(); s=self.session(link); link.status[field]=value
            with self.assertRaises(ValueError): self.start(s)
            self.assertEqual(link.writes,[]); self.assertEqual(s.mode,'idle')

    def test_fields_validation_rejects_without_changing_idle(self):
        s=self.session()
        base=dict(action='patrol',client='owner',generation=s.generation,slots=['1','2'],dwell=500,pauseOnFace=False)
        for changes in ({'slots':['1','1']},{'dwell':True},{'pauseOnFace':'yes'},{'loop':True}):
            with self.assertRaises(ValueError): s.command({**base,**changes},1000)
            self.assertEqual(s.mode,'idle'); self.assertEqual(s.generation,0)

    def test_face_monitor_requires_camera_and_no_face_at_start(self):
        s=self.session()
        with self.assertRaises(ValueError): self.start(s,watch=True)
        s.frame([(25,25,75,75)],100,100,1000,1000)
        with self.assertRaises(ValueError): self.start(s,watch=True)
        s.frame([],100,100,1200,1200)
        self.start(s,at=1200,watch=True)
        self.assertEqual(s.mode,'patrol')

    def test_face_pause_never_auto_resumes_and_resume_needs_clear_fresh_view(self):
        s=self.session(); s.frame([],100,100,1000,1000); self.start(s,watch=True)
        for at in (1200,1400,1600):
            self.command(s,'heartbeat',at); s.frame([(25,25,75,75)],100,100,at,at); s.tick(at)
        self.assertTrue(s.patrol.snapshot(1600)['paused'])
        with self.assertRaises(ValueError): self.command(s,'patrol_resume',1700)
        s.frame([],100,100,1800,1800)
        self.assertTrue(s.patrol.snapshot(1800)['paused'])
        self.command(s,'patrol_resume',1850)
        self.assertFalse(s.patrol.snapshot(1850)['paused'])

    def test_face_monitor_stale_and_invalid_fail_closed(self):
        for variant in ('stale','invalid'):
            s=self.session(); s.frame([],100,100,1000,1000); self.start(s,watch=True)
            if variant=='invalid': s.frame([(0,0,float('nan'),30)],100,100,1200,1200)
            else: self.command(s,'heartbeat',1400); s.tick(1400)
            self.assertEqual(s.mode,'idle')
            s.frame([],100,100,1500,1500); self.assertEqual(s.mode,'idle')

    def test_camera_failure_does_not_resume_route_after_recovery(self):
        s=self.session(); camera=GimbalCamera(s)
        capture=object(); camera.capture=capture
        s.frame([],100,100,1000,1000); self.start(s,watch=True)
        camera._failed(capture,s.camera['generation'],OSError('test'))
        self.assertEqual(s.mode,'idle'); self.assertFalse(s.patrol.active)
        s.frame([],100,100,1500,1500); self.assertEqual(s.mode,'idle')

    def test_page_loss_paused_route_and_preset_change_are_safe(self):
        s=self.session(); self.start(s); self.command(s,'patrol_pause',1100)
        before=copy.deepcopy(s.presets.snapshot())
        with self.assertRaises(ValueError): self.command(s,'preset_delete',1200,slot='1')
        self.assertEqual(before,s.presets.snapshot())
        s.tick(1700); self.assertEqual(s.mode,'idle')
        self.assertEqual(s.patrol.state,'stopped')


if __name__=='__main__': unittest.main()
