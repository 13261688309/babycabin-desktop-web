"""Fail-closed, explicitly armed mapping of associated voice-frame snapshots."""
import copy
import re


FIELDS=frozenset(('v','frames','type','id','last_ms','now_ms','stops','applied','alarms','errors','dropped','epoch'))
COMMANDS={0xA1:('preset_apply','1','观察位一'),0xA2:('preset_apply','2','观察位二'),
          0xA3:('preset_apply','3','观察位三'),0xA4:('auto',None,'开始跟踪'),
          0xA5:('patrol',None,'开始巡航'),0xA6:('patrol_pause',None,'暂停巡航'),
          0xA7:('patrol_resume',None,'继续巡航'),0xA8:('center',None,'云台回中')}


def parse_voice_control(line):
    if not isinstance(line,bytes) or not line.startswith(b'voice_cmd ') or len(line)>384:
        raise ValueError('Invalid voice control snapshot')
    result={}
    for token in line.decode('ascii').strip().split(' ')[1:]:
        key,sep,value=token.partition('=')
        if not sep or key in result or not re.fullmatch('[0-9]{1,10}',value):
            raise ValueError('Invalid voice control token')
        result[key]=int(value)
    if (set(result)!=FIELDS or any(not 0<=value<=0xffffffff for value in result.values())
            or result['v']!=1 or result['type']>255 or result['id']>255):
        raise ValueError('Unsupported voice control snapshot')
    return result


class VoiceWindow:
    WINDOW_MS=30000
    FRESH_MS=400

    def __init__(self):
        self.armed=False
        self.owner=None
        self.started=self.heartbeat=0
        self.previous=None
        self.source=None
        self.route=None
        self.preset_revision=None
        self.message='扩展语音未启用'
        self.last_command=None

    @staticmethod
    def fresh(item,now):
        return (isinstance(item,dict) and type(item.get('receivedMs')) is int
                and 0<=now-item['receivedMs']<VoiceWindow.FRESH_MS)

    def arm(self,item,now,*,client,route,revision,source):
        if self.armed: raise ValueError('扩展语音已启用，不能重复延长窗口')
        if not self.fresh(item,now): raise ValueError('需要新鲜增强语音数据，旧版voice_rx不能触发动作')
        self.armed=True
        self.owner=client
        self.started=self.heartbeat=now
        self.previous=copy.deepcopy(item['data'])
        self.source=source
        self.route=copy.deepcopy(route)
        self.preset_revision=revision
        self.last_command=None
        self.message='允许30秒扩展口令；STOP仍由主控原流程处理'

    def disarm(self,reason):
        self.armed=False
        self.owner=None
        self.message=reason

    def poll(self,item,now,*,revision,epoch,own_active_lease=False):
        if not self.armed: return None
        if now-self.started>=self.WINDOW_MS or now-self.heartbeat>=600:
            self.disarm('扩展语音窗口到期或页面失联'); return None
        if not self.fresh(item,now) or revision!=self.preset_revision:
            self.disarm('增强语音数据失联或预置已变化'); return None
        data=item['data']; old=self.previous
        if data['epoch']!=epoch:
            self.disarm('语音数据与控制会话不一致'); return None
        if data['epoch']!=old['epoch'] and not own_active_lease:
            self.disarm('主控已停止、复位或其他来源接管'); return None
        elapsed=(data['now_ms']-old['now_ms'])&0xffffffff
        if elapsed==0:
            if data!=old: self.disarm('同一时刻语音数据冲突')
            return None
        if not 0<elapsed<500 or data['frames']<old['frames']:
            self.disarm('语音时钟或计数不连续'); return None
        self.previous=copy.deepcopy(data)
        for field in ('stops','applied','alarms','errors','dropped'):
            if data[field]!=old[field]:
                self.disarm('STOP、报警、接收错误或丢帧，扩展语音已关闭'); return None
        delta=data['frames']-old['frames']
        if delta==0:
            if any(data[key]!=old[key] for key in ('type','id','last_ms')):
                self.disarm('接收序号未变但口令数据冲突')
            return None
        if delta!=1 or not 0<=((data['now_ms']-data['last_ms'])&0xffffffff)<400:
            self.disarm('不能唯一关联一份新口令，扩展语音已关闭'); return None
        if data['type']!=0: return None
        code=data['id']
        if code in tuple(range(1,10))+(0x26,):
            self.disarm('原有语音口令已接管'); return None
        if code not in COMMANDS: return None
        action,slot,phrase=COMMANDS[code]
        self.last_command=dict(id=code,phrase=phrase,action=action,source=self.source,
                               frames=data['frames'],result='待校验')
        self.message='收到新扩展口令：'+phrase
        return dict(action=action,slot=slot)

    def snapshot(self,now):
        return dict(armed=self.armed,remainingMs=max(0,self.WINDOW_MS-(now-self.started)) if self.armed else 0,
                    message=self.message,source=self.source,lastCommand=copy.deepcopy(self.last_command),
                    route=copy.deepcopy(self.route),
                    requiresAssociatedSnapshot=True,usesStickyStatusVoice=False,
                    commands=[dict(id=code,phrase=value[2]) for code,value in COMMANDS.items()])
