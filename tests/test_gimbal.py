import copy
import http.client
import json
import struct
import unittest
import zlib
from companion.gimbal import FaceTracker, FaceDetector, encode_control, parse_control
from companion.gimbal_console import GimbalSession, ConsoleServer


def capability(**changes):
    result=dict(v=3,epoch=3,lease=0,seq=0,now_ms=1000,accepted=0,rejected=0,pan=90,tilt=90,
                remaining=0,enabled=0,trip=0,rock=0,motor=0)
    result.update(changes)
    return result


class FakeLink:
    def __init__(self):
        self.status=capability()
        self.writes=[]
        self.closed=False
        self.fail=False
    def poll(self,now): return self.status
    def write(self,**values):
        if self.fail: raise OSError('write failed')
        self.writes.append(values)
    def close(self): self.closed=True


class GimbalTests(unittest.TestCase):
    def command(self,s,action,now=1000,**extra):
        s.command(dict(action=action,client='test-tab',generation=s.generation,**extra),now)

    def test_codec_exact_32_byte_crc_and_bounds(self):
        status=capability()
        data=encode_control(status,sequence=0xffffffff,lease=1234,op=1,pan=110,tilt=95,duration=1000)
        self.assertEqual(len(data),32)
        self.assertEqual(data[:4],b'GC03')
        self.assertEqual(struct.unpack_from('<I',data,28)[0],zlib.crc32(data[:28]))
        self.assertEqual(data[20:28],bytes((110,0,95,0,1,0,232,3)))
        for changes in ({'pan':121},{'tilt':74},{'pan':True},{'duration':10001},{'lease':0},{'op':4}):
            args=dict(sequence=1,lease=1,op=1,duration=1000);args.update(changes)
            with self.assertRaises(ValueError): encode_control(status,**args)

    def test_status_schema(self):
        state=capability()
        line=('gimbal_link '+' '.join(f'{k}={v}' for k,v in state.items())).encode()
        self.assertEqual(parse_control(line),state)
        for bad in (line+b' v=3',line.replace(b'v=3',b'v=2'),line.replace(b'pan=90',b'pan=190'),b'vision_link v=1'):
            with self.assertRaises(ValueError): parse_control(bad)

    def test_tracker_acquisition_deadband_and_rate(self):
        t=FaceTracker(); box=(430,180,550,300)
        self.assertIsNone(t.update([box],640,480,1000,1000))
        self.assertIsNone(t.update([box],640,480,1200,1200))
        self.assertEqual(t.update([box],640,480,1400,1400),(93,90))
        self.assertIsNone(t.update([box],640,480,1400,1500))
        self.assertEqual(t.pan,93)
        for at in range(1600,8000,200): t.update([box],640,480,at,at)
        self.assertEqual(t.pan,120)
        t.reset()
        for at in (1000,1200,1400): result=t.update([(260,180,380,300)],640,480,at,at)
        self.assertEqual(result,(90,90));self.assertEqual(t.state,'centered')

    def test_tracker_reversal_and_vertical(self):
        t=FaceTracker()
        for at in (1000,1200,1400): result=t.update([(430,300,550,430)],640,480,at,at,pan_sign=-1,tilt_sign=-1)
        self.assertEqual(result,(87,87))
        for at in range(1600,9000,200): t.update([(430,300,550,430)],640,480,at,at,pan_sign=-1,tilt_sign=-1)
        self.assertEqual((t.pan,t.tilt),(60,75))

    def test_tracker_ambiguous_stale_jump_reacquisition(self):
        t=FaceTracker();box=(10,10,110,110)
        for at in (1000,1200,1400): t.update([box],640,480,at,at)
        self.assertIsNone(t.update([box,box],640,480,1600,1600));self.assertEqual(t.state,'multiple')
        for at in (1800,2000,2200): t.update([box],640,480,at,at)
        self.assertIsNone(t.update([(400,10,500,110)],640,480,2400,2400));self.assertEqual(t.state,'changed')
        self.assertIsNone(t.update([box],640,480,2600,3000));self.assertEqual(t.state,'stale')
        self.assertIsNone(t.update([(0,0,float('nan'),110)],640,480,3200,3200));self.assertEqual(t.state,'invalid')

    def test_tracker_corrects_small_error_and_tracks_at_low_frame_rate(self):
        t=FaceTracker()
        box=(285,200,405,320)
        for at in range(1000,2400,200):
            result=t.update([box],640,480,at,at)
        self.assertGreater(result[0],90)
        self.assertGreater(result[1],90)
        t.reset()
        for at in (1000,1350,1700,2050):
            result=t.update([(460,300,580,420)],640,480,at,at)
        self.assertEqual(result,(96,96))

    def test_tracker_brakes_on_current_center_without_extra_target(self):
        tracker=FaceTracker()
        def update(x,y,at):
            return tracker.update([(x-150,y-140,x+150,y+140)],640,480,at,at)
        for at in (1000,1100,1200):
            update(430,320,at)
        previous=(tracker.pan,tracker.tilt)
        for at in (1300,1400):
            self.assertEqual(update(320,240,at),(92,92))
            self.assertEqual((tracker.pan,tracker.tilt),previous)
            self.assertEqual(tracker.state,'centered')

    def test_tracker_does_not_steer_against_a_reversed_axis(self):
        tracker=FaceTracker()
        def update(x,y,at):
            return tracker.update([(x-150,y-140,x+150,y+140)],640,480,at,at)
        for at in (1000,1100,1200):
            update(430,295,at)
        previous=(tracker.pan,tracker.tilt)
        update(280,275,1300)
        self.assertLess(tracker.pan,previous[0])
        self.assertGreater(tracker.tilt,previous[1])
        self.assertEqual(tracker.state,'tracking')

    def test_detector_loads_and_blank_not_a_face(self):
        import numpy as np
        self.assertEqual(FaceDetector().detect(np.zeros((480,640,3),dtype=np.uint8)),[])

    def test_preview_manual_center_limits_timeout(self):
        s=GimbalSession()
        self.command(s,'manual',pan=100,tilt=95);s.tick(1000)
        self.assertEqual((s.pan,s.tilt),(100,95));self.assertFalse(s.hardware)
        self.command(s,'center',now=1400);s.tick(1400)
        self.assertEqual((s.pan,s.tilt),(90,90));self.assertEqual(s.started,1000)
        with self.assertRaises(ValueError): self.command(s,'manual',pan=180,tilt=90)
        self.command(s,'heartbeat',now=10900);s.tick(11000)
        self.assertEqual(s.mode,'idle')

    def test_heartbeat_and_multiple_tabs(self):
        s=GimbalSession();self.command(s,'manual',pan=100,tilt=95);s.tick(1000)
        s.command(dict(action='heartbeat',client='other',generation=s.generation),1500)
        s.tick(1600);self.assertEqual(s.mode,'idle')
        with self.assertRaises(ValueError): s.command(dict(action='center',client='old',generation=0),1700)

    def test_new_firmware_and_confirmation_required_for_hardware(self):
        link=FakeLink();s=GimbalSession(link)
        with self.assertRaises(ValueError): self.command(s,'center')
        s.tick(900);self.command(s,'center');s.tick(1000)
        self.assertEqual(link.writes[0]['op'],1)
        s.tick(1200);self.assertEqual(len(link.writes),1)
        s.tick(1400);self.assertEqual(s.mode,'idle')
        s.tick(1420);self.assertEqual(link.writes[-1]['op'],3)

    def test_explicit_stop_works_even_when_pc_is_idle(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900)
        self.command(s,'stop');s.tick(1000)
        self.assertEqual(link.writes[-1]['op'],0)

    def test_automatic_cancel_retains_own_lease_when_voice_arrives(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900)
        self.command(s,'center');s.tick(1000)
        token=s.lease;link.status['lease']=token
        s.tick(1100);s.tick(1600)
        link.status['lease']=0;link.status['epoch']+=1
        s.tick(1620)
        self.assertEqual(link.writes[-1],dict(lease=token,op=3))

    def test_hardware_voice_or_protection_stop_never_restarts(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900)
        self.command(s,'center');s.tick(1000)
        token=s.lease;link.status['lease']=token
        s.tick(1100);self.assertEqual(link.writes[-1]['op'],2)
        link.status['lease']=0;link.status['epoch']+=1
        s.tick(1200);self.assertEqual(s.mode,'idle')
        s.tick(1220)
        for at in (1300,1400,1500):s.tick(at)
        self.assertEqual([v['op'] for v in link.writes],[1,2])
        self.assertFalse(s.stop_requested)

    def test_manual_sends_latest_changed_target_early_with_same_watchdog_and_stop(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900)
        self.command(s,'manual',pan=95,tilt=92);s.tick(1000)
        self.assertEqual([v['op'] for v in link.writes],[1])
        self.command(s,'manual',now=1005,pan=100,tilt=94)
        s.tick(1010)
        self.assertEqual(len(link.writes),1)
        link.status['lease']=s.lease
        s.tick(1020)
        self.command(s,'manual',now=1021,pan=105,tilt=96)
        s.tick(1039)
        self.assertEqual(len(link.writes),1)
        s.tick(1040)
        self.assertEqual((link.writes[-1]['op'],link.writes[-1]['pan'],link.writes[-1]['tilt']),(2,105,96))
        s.tick(1080)
        s.tick(1139)
        self.assertEqual(len(link.writes),2)
        s.tick(1140)
        self.assertEqual(len(link.writes),3)
        self.assertEqual(s.started,1000)
        self.command(s,'manual',now=1150,pan=110,tilt=98)
        self.command(s,'stop',now=1151)
        s.tick(1152);s.tick(1240)
        self.assertEqual([v['op'] for v in link.writes],[1,2,2,0])

    def test_voice_takeover_before_start_ack_is_not_cancelled(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900)
        self.command(s,'center');s.tick(1000)
        link.status['epoch']+=2
        link.status['remaining']=3000
        link.status['pan']=100
        s.tick(1100);s.tick(1600)
        self.assertEqual(s.mode,'idle')
        self.assertEqual([v['op'] for v in link.writes],[1])
        self.assertFalse(s.stop_requested)

    def test_auto_starts_from_latest_mcu_target_after_voice(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900)
        link.status['pan']=105;link.status['tilt']=95
        self.command(s,'auto',panSign=1,tiltSign=1)
        self.assertEqual((s.pan,s.tilt),(105,95))
        for at in (1001,1200,1400):
            self.command(s,'heartbeat',now=at)
            s.frame([(260,180,380,300)],640,480,at,at);s.tick(at)
        self.assertEqual((link.writes[0]['pan'],link.writes[0]['tilt']),(105,95))

    def test_auto_tracks_then_stops_on_loss_and_does_not_reacquire(self):
        s=GimbalSession();self.command(s,'auto',panSign=1,tiltSign=1)
        for at in (1001,1200,1400):
            self.command(s,'heartbeat',now=at)
            s.frame([(430,180,550,300)],640,480,at,at);s.tick(at)
        self.assertEqual(s.pan,93);self.assertTrue(s.sent)
        s.frame([],640,480,1600,1600);self.assertEqual(s.mode,'idle')
        for at in (1800,2000,2200):s.frame([(430,180,550,300)],640,480,at,at);s.tick(at)
        self.assertEqual(s.mode,'idle');self.assertEqual(s.pan,93)

    def test_auto_network_stall_and_repeated_frame(self):
        s=GimbalSession();self.command(s,'auto',panSign=1,tiltSign=1)
        for at in (1001,1200,1400):
            self.command(s,'heartbeat',now=at);s.frame([(430,180,550,300)],640,480,at,at);s.tick(at)
        s.frame([(430,180,550,300)],640,480,1400,1500);self.assertEqual(s.mode,'auto')
        self.command(s,'heartbeat',now=1800);s.tick(1800);self.assertEqual(s.mode,'idle')

    def test_auto_seeds_only_two_observations_and_needs_a_post_click_frame(self):
        box=(430,280,550,400)
        for interval in (100,200,350):
            with self.subTest(interval=interval):
                link=FakeLink();s=GimbalSession(link);s.tick(900)
                for at in (990-interval,990):
                    s.frame([box],640,480,at,at)
                self.command(s,'auto',panSign=1,tiltSign=1)
                self.assertEqual(s.tracker.confirmations,2)
                s.frame([box],640,480,1000,1001)
                s.tick(1001)
                self.assertEqual(link.writes,[])
                s.frame([box],640,480,990+interval,990+interval)
                s.tick(990+interval)
                self.assertEqual(link.writes[0]['op'],1)
                self.assertGreater(link.writes[0]['pan'],90)
                self.assertGreater(link.writes[0]['tilt'],90)

    def test_auto_seed_cannot_bridge_uncertain_stale_or_changed_scene(self):
        box=(430,280,550,400)
        for empty_at_click, first, next_width, post in (
                (True,800,640,1100),(False,599,640,1100),
                (False,800,800,1100),(False,800,640,1390)):
            with self.subTest(first=first,width=next_width,post=post,empty=empty_at_click):
                link=FakeLink();s=GimbalSession(link);s.tick(900)
                s.frame([box],640,480,first,first)
                s.frame([] if empty_at_click else [box],640,480,990,990)
                self.command(s,'auto',panSign=1,tiltSign=1)
                s.frame([box],next_width,480,post,post)
                s.tick(post)
                self.assertEqual(link.writes,[])
                self.assertFalse(s.sent)

    def test_active_auto_stops_if_next_frame_arrives_after_a_gap(self):
        s=GimbalSession();box=(430,180,550,300)
        self.command(s,'auto',panSign=1,tiltSign=1)
        for at in (1001,1200,1400):
            s.frame([box],640,480,at,at)
        s.tick(1400)
        self.assertTrue(s.sent)
        s.frame([box],640,480,1800,1800)
        self.assertEqual(s.mode,'idle')

    def test_write_fault_no_retry_and_explicit_stop(self):
        link=FakeLink();s=GimbalSession(link);s.tick(900);link.fail=True
        self.command(s,'center');s.tick(1000)
        self.assertEqual(s.mode,'idle');self.assertTrue(link.closed)
        s.tick(1100);self.assertEqual(len(link.writes),0)

    def test_http_origin_token_exact_payload_and_no_device_preview(self):
        s=GimbalSession();server=ConsoleServer(s);server.start()
        try:
            port=server.server.server_port
            def post(data,headers=None):
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
                base={'Origin':server.url[:-1],'Content-Type':'application/json','X-Control-Token':s.token}
                base.update(headers or {})
                c.request('POST','/api/control',json.dumps(data),base)
                response=c.getresponse();body=response.read();status=response.status;c.close();return status,body
            data=dict(action='center',client='test',generation=0)
            self.assertEqual(post(data,{'X-Control-Token':'bad'})[0],403)
            self.assertEqual(post(data,{'Origin':'http://example.com'})[0],403)
            self.assertEqual(post(data)[0],200)
            self.assertEqual(post({**data,'extra':1})[0],400)
            self.assertEqual(post(data)[0],400)
            self.assertIsNone(s.link)
        finally:server.close()


if __name__=='__main__':unittest.main()
