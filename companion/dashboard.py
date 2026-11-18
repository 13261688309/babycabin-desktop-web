"""Localhost dashboard: device observation and local reminder acknowledgement."""

from __future__ import annotations

import copy
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time

from .alert_inbox import AlertInbox, utc_now
from .monitor_rules import MonitorRules
from .state import Observation
from .telemetry import unique_object
from .voice_observation import VoiceCounterTracker
from .vision_protocol import observation_fresh


HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>婴儿陪护舱观察看板</title><link rel="stylesheet" href="/style.css"></head>
<body><header><div><p class="eyebrow">本地观察终端</p><h1>婴儿陪护舱</h1></div>
<div class="header-status"><span id="liveDot" class="dot"></span><span id="liveText">等待数据</span><b id="linkMode">仅观察</b></div></header>
<main><section class="summary" aria-label="状态摘要">
<article class="state-panel"><p>视觉状态</p><strong id="visionState">Unknown</strong><span id="visionReason">等待第一帧</span></article>
<article><p>候选类别</p><strong id="candidate">--</strong><span id="confidence">置信度 --</span></article>
<article><p>环境温度</p><strong id="temperature">--</strong><span id="dhtHealth">DHT11 未连接</span></article>
<article><p>环境湿度</p><strong id="humidity">--</strong><span id="deviceState">主控未连接</span></article>
</section>
<section class="reminders" aria-label="待处理提醒">
<div class="section-head"><h2>待处理提醒 <span id="reminderCount">0</span></h2>
<div class="reminder-options"><label><input id="reminderHistory" type="checkbox">显示已查看</label>
<label><input id="reminderSound" type="checkbox">提示音</label></div></div>
<p id="reminderConnection" role="status" hidden></p><p id="reminderError" role="alert" hidden></p>
<ol id="reminderList"><li class="reminder-empty">暂无提醒</li></ol>
<p class="reminder-foot">“已查看”仅确认本场提醒，不解除故障、不启动机构。<span id="reminderDropped"></span></p>
</section>
<details class="rule-settings"><summary>自定义监测规则 <span id="rulesSummary">尚未启用</span></summary>
<form id="rulesForm"><div id="rulesFields"></div><div class="rule-save"><button type="submit" id="rulesSave">应用规则</button><span id="rulesMessage" role="status"></span></div>
<p>阈值由你设置，示例值不是医疗标准。恢复需连续5秒满足条件；仅本次运行有效，不控制执行机构。</p></form></details>
<section class="monitor-grid"><div class="live-view"><div class="section-head"><h2>实时图传</h2><span id="previewState">等待画面</span></div>
<div class="viewport"><img id="preview" alt="本次相机实时画面" hidden><p id="previewEmpty">尚无新鲜画面</p></div></div>
<aside class="events-panel"><div class="section-head"><h2>本场事件</h2><a href="/api/report.json" download="observation-session.json" title="导出本场元数据" aria-label="导出本场元数据"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4"/></svg></a></div>
<ol id="events"><li>等待状态变化</li></ol></aside></section>
<section class="mcu-band" aria-label="主控软件状态">
<div><p>最新语音命令</p><strong id="voiceCommand">--</strong><span id="voiceAge">无有效状态</span></div>
<div><p>执行使能报告</p><strong id="interlockEnable">--</strong><span>软件报告，不代表硬件已断电</span></div>
<div><p>主控模式</p><strong id="mcuMode">--</strong><span id="wiringScope">未接入主控</span></div>
<div><p>STOP 接收 / 软件处理累计</p><strong id="stopCounters">-- / --</strong><span id="stopCounterState">当前固件未报告计数</span></div>
</section>
<section class="vision-relay" aria-label="动作软件报告"><strong>动作指令 <span id="motionAction">--</span></strong><span id="motionDetail">尚无动作报告；不代表实物动作已完成</span></section>
<section class="vision-relay"><strong>主控视觉回传 <span id="mcuVision">--</span></strong><span id="visionRelayDetail">尚无识别上报链路</span></section>
<section class="band"><div><h2>识别质量</h2><p id="quality">等待模型输出</p></div>
<dl><div><dt>连续帧</dt><dd id="frames">0</dd></div><div><dt>推理耗时</dt><dd id="inference">--</dd></div><div><dt>数据年龄</dt><dd id="age">--</dd></div></dl></section>
<section class="grid"><div class="table-panel"><h2>传感器状态</h2><table><thead><tr><th>来源</th><th>状态</th></tr></thead>
<tbody id="sensors"><tr><td>DHT11</td><td>等待主控</td></tr></tbody></table></div>
<div class="alert-panel"><h2>最新告警</h2><strong id="alertName">无可用告警</strong><p id="alertDetail">接入主控串口后显示设备告警。</p>
<h2>运行信息</h2><dl class="runtime"><div><dt>主控接口</dt><dd id="transport">--</dd></div><div><dt>图传连接</dt><dd id="videoLink">--</dd></div><div><dt>模型</dt><dd id="model">--</dd></div><div><dt>来源</dt><dd id="source">--</dd></div><div><dt>序号</dt><dd id="sequence">--</dd></div></dl></div></section>
<p class="notice">研究原型，不代替成人照护或医疗判断。断流、冲突、低置信度均显示 Unknown；看板不提供执行机构控制。</p>
</main><script src="/reminders.js"></script><script src="/monitor_rules.js"></script><script src="/app.js"></script></body></html>""".encode("utf-8")

CSS = """*{box-sizing:border-box}body{margin:0;background:#f5f7f6;color:#17201d;font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif;letter-spacing:0}header{min-height:74px;padding:0 28px;background:#fff;border-bottom:1px solid #d8dfdc;display:flex;align-items:center;justify-content:space-between}h1{font-size:22px;margin:1px 0 0}.eyebrow{font-size:12px;color:#5f6f68;margin:0}.header-status{display:flex;align-items:center;gap:9px}.header-status b{font-size:12px;color:#8a3c12;background:#fff1e8;padding:4px 8px;border-radius:4px}.dot{width:9px;height:9px;border-radius:50%;background:#9aa5a0}.dot.live{background:#087e5b;box-shadow:0 0 0 3px #d8f0e8}main{max-width:1120px;margin:0 auto;padding:24px}.summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.summary article,.table-panel,.alert-panel{background:#fff;border:1px solid #d8dfdc;border-radius:6px;padding:17px}.summary p{color:#66736e;margin:0 0 10px}.summary strong{font-size:25px;display:block;overflow-wrap:anywhere}.summary span{display:block;color:#66736e;margin-top:5px}.state-panel{border-left:4px solid #0b8060!important}.state-panel.warning{border-left-color:#be4c2a!important}.band{margin-top:14px;padding:16px 18px;background:#173a30;color:#fff;display:flex;justify-content:space-between;align-items:center}.band h2,.grid h2{font-size:15px;margin:0 0 5px}.band p{margin:0;color:#c8dad4}.band dl{display:flex;gap:30px;margin:0}.band dl div,.runtime div{display:flex;flex-direction:column}.band dt{font-size:11px;color:#b8cec6}.band dd{font-size:16px;margin:2px 0 0}.grid{display:grid;grid-template-columns:1.35fr 1fr;gap:14px;margin-top:14px}table{width:100%;border-collapse:collapse;margin-top:12px}th,td{text-align:left;padding:9px 7px;border-bottom:1px solid #e4e9e7}th{font-size:12px;color:#68756f}.health-ok{color:#087e5b}.health-bad{color:#b23b24}.alert-panel>strong{display:block;font-size:18px;margin-top:14px}.alert-panel>p{color:#66736e}.alert-panel h2:nth-of-type(2){border-top:1px solid #e4e9e7;padding-top:15px;margin-top:18px}.runtime{display:grid;gap:7px;margin:10px 0 0}.runtime div{flex-direction:row;justify-content:space-between;gap:10px}.runtime dt{color:#66736e}.runtime dd{margin:0;max-width:65%;overflow-wrap:anywhere;text-align:right}.notice{color:#68756f;border-top:1px solid #d8dfdc;margin:20px 0 0;padding-top:14px}@media(max-width:760px){header{padding:15px 18px}.summary{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.band{align-items:flex-start;gap:18px;flex-direction:column}.band dl{width:100%;justify-content:space-between;gap:10px}main{padding:16px}}@media(max-width:430px){.summary{grid-template-columns:1fr}.header-status #liveText{display:none}}""".encode("utf-8")

CSS += b""".mcu-band{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;padding:18px 0;border-bottom:1px solid #d8dfdc}.mcu-band p{margin:0 0 5px;color:#66736e}.mcu-band strong{display:block;font-size:19px;overflow-wrap:anywhere}.mcu-band span{display:block;margin-top:5px;color:#66736e;font-size:12px;overflow-wrap:anywhere}@media(max-width:760px){.mcu-band{grid-template-columns:1fr;gap:15px}}"""
CSS += b"""@media(min-width:761px){.mcu-band{grid-template-columns:repeat(4,minmax(0,1fr))}}"""
CSS += b""".vision-relay{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:14px 0}.vision-relay strong{font-size:14px}.vision-relay>span{font-size:12px;color:#66736e;overflow-wrap:anywhere}.vision-relay strong span{margin-left:10px;color:#087e5b}"""
CSS += b""".monitor-grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(240px,1fr);gap:20px;margin-top:20px}.section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:36px}.section-head h2{font-size:15px;margin:0}.section-head span{font-size:12px;color:#66736e}.section-head a{display:grid;place-items:center;width:32px;height:32px;color:#087e5b;border:1px solid #d8dfdc;border-radius:4px}.viewport{position:relative;aspect-ratio:4/3;background:#e7ece9;display:grid;place-items:center;overflow:hidden}.viewport img{width:100%;height:100%;object-fit:contain}.viewport img[hidden]{display:none}.viewport p{position:absolute;color:#5f6f68;margin:12px;text-align:center}.events-panel{min-width:0}.events-panel ol{list-style:none;padding:0;margin:0;max-height:360px;overflow:auto}.events-panel li{padding:11px 0;border-bottom:1px solid #d8dfdc;overflow-wrap:anywhere}.events-panel time{display:block;font-size:11px;color:#66736e;margin-bottom:3px}@media(max-width:760px){.monitor-grid{grid-template-columns:minmax(0,1fr)}.events-panel ol{max-height:180px}}"""
CSS += b""".reminders{margin-top:20px;border-top:1px solid #d8dfdc;border-bottom:1px solid #d8dfdc;padding:12px 0}.reminder-options{display:flex;gap:14px;flex-wrap:wrap}.reminder-options label{display:flex;align-items:center;gap:5px;font-size:12px}.reminders ol{list-style:none;margin:6px 0;padding:0;max-height:310px;overflow:auto}.reminders li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;padding:10px 0;border-top:1px solid #e4e9e7}.reminders li strong{font-size:15px}.reminders li p{margin:3px 0;color:#52635a;overflow-wrap:anywhere}.reminders li small{color:#66736e}.reminders button{border:1px solid #087e5b;border-radius:4px;background:#fff;color:#086548;font:inherit;padding:7px 12px;cursor:pointer;white-space:nowrap}.reminders button:disabled{opacity:.6;cursor:default}.reminder-foot{font-size:12px;color:#66736e;margin:6px 0}.reminder-tag{display:inline-block;margin-left:8px;font-size:12px;color:#a32c28}.reminders [data-condition=recovered] .reminder-tag{color:#087e5b}#reminderError,#reminderConnection{color:#a32c28;font-size:13px}#reminderCount{color:#a32c28;font-weight:700}.reminder-empty{color:#66736e}@media(max-width:430px){.reminders .section-head{align-items:flex-start;flex-direction:column;gap:7px}.reminders li{gap:8px}.reminder-options{width:100%;justify-content:space-between}}"""

CSS += b""".rule-settings{padding:12px 0;border-bottom:1px solid #d8dfdc}.rule-settings summary{cursor:pointer;font-weight:600}.rule-settings summary span{margin-left:12px;font-size:12px;color:#66736e;font-weight:400}.rule-settings fieldset{border:0;border-top:1px solid #d8dfdc;padding:12px 0;margin:12px 0;display:flex;align-items:center;gap:14px;flex-wrap:wrap}.rule-settings legend{font-weight:600}.rule-settings label{display:flex;align-items:center;gap:6px;font-size:12px}.rule-settings input[type=number]{width:70px;border:1px solid #a3b2a9;border-radius:4px;padding:6px;font:inherit}.rule-settings output{font-size:12px;color:#52635a;min-width:100px}.rule-settings p{font-size:12px;color:#66736e}.rule-save{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.rule-save button{font:inherit;background:#087e5b;color:white;border:0;border-radius:4px;padding:8px 14px;cursor:pointer}.rule-save button:disabled{opacity:.5}.rule-save span{font-size:12px}"""

JS = """const $=id=>document.getElementById(id);
const labels={confirmed:'连续帧确认',confirming:'正在确认',ambiguous_class:'同一区域类别冲突',multiple_detections:'存在多个检测目标',low_confidence:'置信度不足',no_detection:'未检测到目标',stale:'数据已过期',stream_lost:'视频流中断',runtime_error:'运行异常',stopped:'观察已停止',interrupted:'观察被中断',end_of_stream:'文件已播放完毕',transport_lost:'看板连接中断',no_frame:'等待第一帧'};
labels.telemetry_only='仅主控遥测，未接入视觉';
const sensorNames={dht11:'DHT11 温湿度',mpu6050:'MPU6050 运动',microphone:'麦克风',wetProbe:'尿湿探头',battery:'电池',camera:'相机检测',ambientLight:'环境光',airQuality:'空气质量',presence:'在床/接近',irTemp:'红外测温',loadCell:'称重'};
const health={ok:'正常',stale:'过期',read_failed:'读取失败',disabled:'未启用'};
const connection={reconnecting:'主控重连中',awaiting_data:'等待重连后的有效遥测',reconnect_expired:'主控重连超时',cleanup_failed:'串口清理失败',closed:'主控连接已关闭'};
function value(v,suffix){return v===null||v===undefined?'--':`${v}${suffix}`}
function render(x){
 window.ReminderUI.render(x);
 window.MonitorRulesUI.render(x);
 const names={voice:'语音命令变化',stop:'语音状态变为 STOP',stop_rx:'STOP 接收计数增加',stop_applied:'STOP 软件处理计数增加',voice_error:'语音 UART 错误计数增加',voice_drop:'语音队列丢弃计数增加',video:'图传状态变化',vision:'视觉确认状态',alert:'主控告警'};
 const events=x.events||[];$('events').replaceChildren(...(events.length?events.slice().reverse().map(e=>{const li=document.createElement('li'),t=document.createElement('time'),s=document.createElement('span');t.textContent=e.utc||'';s.textContent=`${names[e.kind]||e.kind} · ${e.value}`;li.append(t,s);return li}):[Object.assign(document.createElement('li'),{textContent:'等待状态变化'})]));
 const r=x.record||{},d=r.device||{},report=d.report||{},data=report.data||{},alert=d.alert||{},ad=alert.data||{},sh=data.sensorHealth||{};
 $('transport').textContent={h743_native_usb_cdc_read_only:'H743 原生 USB（只读）',h743_native_usb_cdc_observation_uplink:'H743 原生 USB（识别上报）',uart_adapter_read_only:'UART 适配器（只读）'}[d.transport]||(r.device?'未标注':'未接入');
 $('linkMode').textContent=d.visionUplink?.enabled===true?'识别上报':'仅观察';
 const o=x.live===true?r.observation||{}:{state:'Unknown',reason:r.observation?.reason||'stale'};
 const collectorStopped=['stopped','interrupted','runtime_error','end_of_stream'].includes(r.observation?.reason);
 const reportFresh=!collectorStopped&&d.online===true&&report.fresh===true;
 const status=d.status||{},statusFresh=!collectorStopped&&d.online===true&&status.fresh===true,sd=statusFresh?status.data||{}:{};
 const voice=Number.isInteger(sd.voice)&&sd.voice>=0&&sd.voice<=255?sd.voice:null;
 const voiceName=voice===0?'未收到命令':voice===1||voice===2?'STOP 命令':'其他命令';
 $('voiceCommand').textContent=statusFresh&&voice!==null?`0x${voice.toString(16).toUpperCase().padStart(2,'0')} ${voiceName}`:'--';
 $('voiceAge').textContent=statusFresh?`最近状态 ${status.ageMs??0} ms前`:'无有效状态';
 const vr=d.voiceRx||{},vrFresh=!collectorStopped&&d.online===true&&vr.fresh===true,vrd=vr.data||{};
 $('stopCounters').textContent=vrFresh&&vrd.v===2?`${vrd.stops} / ${vrd.applied}`:'-- / --';
 $('stopCounterState').textContent=vrFresh&&vrd.v===2?`自本次主控启动累计 · UART错误 ${vrd.errors} · 队列丢弃 ${vrd.dropped}`:d.voiceRx?'计数过期或旧版格式':'当前固件未报告计数';
 const vl=d.visionLink||{},vld=vl.data||{},modelMatches=!r.model?.sha256||vld.model===r.model.sha256;
 const visualFresh=!collectorStopped&&d.online===true&&vl.fresh===true&&vl.observationFresh===true&&modelMatches;
 $('mcuVision').textContent=visualFresh?['Unknown','Cry','Happy','Neutral','Sleep'][vld.label]||'Unknown':d.visionLink?'Unknown':'--';
 $('visionRelayDetail').textContent=!modelMatches?'模型不一致':d.visionUplink?.fault?'上报写入已停止':d.visionLink?`主控接收 ${vld.accepted??0} · 拒收 ${vld.rejected??0} · 丢块 ${vld.drops??0} · 序号 ${vld.seq??'--'}`:'尚无识别上报链路';
 $('interlockEnable').textContent=statusFresh&&sd.il_en===0?'未使能':statusFresh&&sd.il_en===1?'已使能':'--';
 $('mcuMode').textContent=statusFresh?({idle:'待机',sleep:'睡眠',soothe:'安抚',alert:'告警',fault:'故障'}[sd.mode]||'--'):'--';
 const motion=d.motion||{},md=motion.data||{},motionFresh=statusFresh&&motion.fresh===true;
 const motionLeft=Math.max(0,(md.remaining_ms||0)-(motion.ageMs||0));
 const motionActive=motionFresh&&sd.il_en===1&&md.action>0&&motionLeft>0&&!['fault','alert'].includes(sd.mode);
 $('motionAction').textContent=motionActive?({1:md.motor_cmd<0?'电机反转指令':'电机正转指令',2:'云台调整指令',3:'安抚指令'}[md.action]||'--'):motionFresh?(sd.il_en===0?'未使能':md.action===0?'无动作指令':'指令到期或状态待确认'):'--';
 $('motionDetail').textContent=motionActive?`目标 PWM ${md.motor_cmd}% · 云台 ${md.pan_target}° / ${md.tilt_target}° · 剩余 ${(motionLeft/1000).toFixed(1)}s（软件报告）`:motionFresh?'软件报告，不代表转速、位置或物理断电已确认':d.motion?'动作报告过期':'当前固件未报告独立动作';
 if(statusFresh&&sd.mode==='idle'&&sd.il_en===1)$('mcuMode').textContent=motionActive?({1:'电机独立控制',2:'云台独立控制',3:'安抚'}[md.action]||'动作待确认'):'已使能，动作待确认';
 $('wiringScope').textContent={isolated_usb:'主控独立 USB',controlled_powered_voice_rx:'受控语音接收联调',usb_powered_dht11_3v3:'USB供电 DHT11（3.3V）',controlled_powered_voice_rx_and_dht11_3v3:'语音＋DHT11并行（受控供电）'}[d.wiringScope]||'未标注接线范围';
 const active=x.live===true,mcuLive=reportFresh||statusFresh;$('liveDot').classList.toggle('live',active||mcuLive);$('liveText').textContent=active?'观察运行中':mcuLive?'主控在线 · 视觉不可用':'数据已停止/过期';
 $('visionState').textContent=o.state||'Unknown';$('visionReason').textContent=labels[o.reason]||o.reason||'等待第一帧';document.querySelector('.state-panel').classList.toggle('warning',(o.state||'Unknown')==='Unknown');
 $('candidate').textContent=o.candidate||'--';$('confidence').textContent=o.confidence==null?'置信度 --':`置信度 ${(o.confidence*100).toFixed(1)}%`;
 $('temperature').textContent=value(reportFresh&&sh.dht11==='ok'?data.temp:null,' °C');$('humidity').textContent=value(reportFresh&&sh.dht11==='ok'?data.hum:null,' %RH');$('dhtHealth').textContent=`DHT11 ${!reportFresh&&sh.dht11==='ok'?'过期':health[sh.dht11]||'无数据'}`;$('deviceState').textContent=collectorStopped?'采集已结束':reportFresh?'主控遥测在线':connection[d.connectionState]||(d.online?'等待有效遥测':'主控未连接');
 $('quality').textContent=labels[o.reason]||o.reason||'等待模型输出';$('frames').textContent=o.confirmed_frames??0;$('inference').textContent=r.inferenceMs==null?'--':`${r.inferenceMs.toFixed(1)} ms`;$('age').textContent=x.ageMs==null?'--':`${x.ageMs} ms`;
 $('sensors').replaceChildren(...Object.entries(sensorNames).map(([key,name])=>{const status=!reportFresh&&sh[key]==='ok'?'stale':sh[key];const tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td');a.textContent=name;b.textContent=health[status]||'无数据';b.className=status==='ok'?'health-ok':status&&status!=='disabled'?'health-bad':'';tr.append(a,b);return tr}));
 const alertFresh=!collectorStopped&&d.online===true&&alert.fresh===true;
 $('alertName').textContent=alertFresh&&ad.alert?(ad.alert==='visual_cry'?'视觉 Cry 提醒':ad.alert):'无可用告警';$('alertDetail').textContent=alertFresh?`级别 ${ad.severity||'--'} · 建议 ${ad.suggestion==='check_live_view'?'查看实时画面':ad.suggestion||'--'}`:'接入主控串口后显示设备告警。';
 const video=r.video||{},videoNames={not_requested:'未接入',connecting:'正在连接',streaming:'实时接收',reconnecting:'正在重连',retry_exhausted:'重连已耗尽',end_of_stream:'文件已结束',closed:'已关闭'};
 $('videoLink').textContent=videoNames[video.connectionState]||'--';
 if(video.networkReader==='http-snapshot')$('videoLink').textContent+=' · 单张取图备用通道';
 if(Number.isInteger(video.reconnectAttempts)&&video.reconnectAttempts>0)$('videoLink').textContent+=` · ${video.reconnectAttempts}次重连`;
 const model=r.model||{};$('model').textContent=model.sha256?`${model.name||'model'} · ${model.sha256.slice(0,12)}`:'--';$('source').textContent={network:'网络视频',camera:'本地相机',file:'本地文件',serial_only:'仅主控'}[r.sourceKind]||r.sourceKind||'--';$('sequence').textContent=r.sequence??'--';
}
async function poll(){
 const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),2000);
 try{const response=await fetch('/api/state',{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error('dashboard unavailable');render(await response.json())}
 catch(error){render({live:false,record:{observation:{state:'Unknown',reason:'transport_lost'}}})}
 finally{clearTimeout(timeout);setTimeout(poll,1000)}
}poll();
let previewUrl=null,previewExpiry=null;
function clearPreview(){
 $('preview').hidden=true;$('preview').removeAttribute('src');$('previewEmpty').hidden=false;$('previewState').textContent='无新鲜画面';if(previewUrl){URL.revokeObjectURL(previewUrl);previewUrl=null}
}
async function previewPoll(){
 const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),2000),started=performance.now();
 try{const r=await fetch('/api/frame.jpg',{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error('no fresh frame');const blob=await r.blob();const remaining=Number(r.headers.get('X-Frame-Remaining-Ms'))-(performance.now()-started);if(!(remaining>0))throw new Error('expired frame');const next=URL.createObjectURL(blob);$('preview').src=next;$('preview').hidden=false;$('previewEmpty').hidden=true;$('previewState').textContent='实时 · 模型检测框';if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=next;clearTimeout(previewExpiry);previewExpiry=setTimeout(clearPreview,remaining)}
 catch(e){clearPreview()}
 finally{clearTimeout(timeout);setTimeout(previewPoll,500)}
}previewPoll();""".encode("utf-8")


class DashboardState:
    def __init__(self, stale_ms: int = 1500) -> None:
        if stale_ms < 0:
            raise ValueError("stale_ms must be non-negative")
        self.stale_ms = stale_ms
        self._lock = threading.Lock()
        self._record: dict | None = None
        self._published_ms: int | None = None
        self._frame: tuple[bytes, int] | None = None
        self._events = deque(maxlen=40)
        self._previous = {}
        self._voice_counter = VoiceCounterTracker()
        self._inbox = AlertInbox()
        self._rules = MonitorRules()
        self.ack_token = secrets.token_urlsafe(32)

    def publish(self, record: dict) -> None:
        with self._lock:
            device = record.get('device') or {}
            status = device.get('status') or {}
            delta = self._voice_counter.observe(record)
            if delta:
                for key, kind in (('stops', 'stop_rx'), ('applied', 'stop_applied'),
                                  ('errors', 'voice_error'), ('dropped', 'voice_drop')):
                    if delta[key]:
                        self._events.append({'utc': record.get('utc', ''), 'kind': kind,
                            'value': delta[key], 'fromMs': delta['fromMs'], 'toMs': delta['toMs']})
            values = {'video': (record.get('video') or {}).get('connectionState')}
            if device.get('online') is True and status.get('fresh') is True:
                values['voice'] = (status.get('data') or {}).get('voice')
            observation = record.get('observation') or {}
            if observation.get('reason') == 'confirmed':
                values['vision'] = observation.get('state')
            alert = device.get('alert') or {}
            if device.get('online') is True and alert.get('fresh') is True:
                data = alert.get('data') or {}
                values['alert'] = data.get('alert')
            for kind, value in values.items():
                if value is None:
                    continue
                previous = self._previous.get(kind)
                counter_stop = self._voice_counter.seen and kind == 'voice' and value in (1, 2)
                if not counter_stop and previous != value and (previous is not None or kind in ('vision', 'alert')):
                    self._events.append({'utc': record.get('utc', ''),
                        'kind': 'stop' if kind == 'voice' and value in (1, 2) else kind,
                        'value': value})
                self._previous[kind] = value
            if (values['video'] not in (None, 'streaming') or observation.get('reason') in
                    ('stale', 'stopped', 'interrupted', 'runtime_error', 'end_of_stream')):
                self._frame = None
            self._record = copy.deepcopy(record)
            self._published_ms = time.monotonic_ns() // 1_000_000
            self._snapshot_locked(self._published_ms)

    def publish_jpeg(self, jpeg: bytes, captured_ms: int) -> None:
        if not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 2 * 1024 * 1024:
            raise ValueError('Invalid bounded JPEG')
        if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
            raise ValueError('Invalid JPEG markers')
        if type(captured_ms) is not int:
            raise ValueError('Frame timestamp must be integer milliseconds')
        with self._lock:
            self._frame = (jpeg, captured_ms)

    def frame(self) -> bytes | None:
        return self.frame_payload()[0]

    def frame_payload(self) -> tuple[bytes | None, int]:
        now = time.monotonic_ns() // 1_000_000
        with self._lock:
            if self._frame is None or not 0 <= now - self._frame[1] <= self.stale_ms:
                return None, 0
            return self._frame[0], self.stale_ms - (now - self._frame[1])

    def snapshot(self) -> dict:
        now = time.monotonic_ns() // 1_000_000
        with self._lock:
            return self._snapshot_locked(now)

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            self._snapshot_locked(time.monotonic_ns() // 1_000_000)
            return self._inbox.acknowledge(alert_id, utc_now())

    def configure_rules(self, config: dict, revision: int) -> bool:
        with self._lock:
            if revision != self._rules.revision:
                return False
            self._rules.configure(config)
            self._snapshot_locked(time.monotonic_ns() // 1_000_000)
            return True

    def _snapshot_locked(self, now: int) -> dict:
        record = copy.deepcopy(self._record)
        published = self._published_ms
        events = copy.deepcopy(list(self._events))
        age = None if published is None else max(0, now - published)
        reason = ((record or {}).get("observation") or {}).get("reason")
        terminal = reason in {"stream_lost", "runtime_error", "stopped", "interrupted", "end_of_stream"}
        collector_stopped = terminal and reason != "stream_lost"
        live = age is not None and age <= self.stale_ms and not terminal
        frame_age = (record or {}).get('frameAgeMs')
        if type(frame_age) in (int, float) and age is not None:
            record['frameAgeMs'] = frame_age + age
            live = live and 0 <= record['frameAgeMs'] <= self.stale_ms
        if record is not None and not live:
            record["observation"] = Observation(reason=reason if terminal else "stale").to_dict()
            record["detections"] = []
        device = (record or {}).get("device")
        if isinstance(device, dict) and age is not None:
            stale_ms = device.get("staleMs", 6000)
            for kind in ("report", "alert", "status", "voiceRx", "visionLink", "motion"):
                item = device.get(kind)
                if isinstance(item, dict):
                    item["ageMs"] = item.get("ageMs", 0) + age
                    item["fresh"] = (not collector_stopped and device.get("online") is True and item.get("fresh") is True
                                     and item["ageMs"] <= stale_ms)
                    if kind == 'visionLink':
                        item['observationFresh'] = observation_fresh(item)
        result = {"record": record, "ageMs": age, "live": live, "events": events}
        self._inbox.observe(result, now, utc_now())
        rules = self._rules.observe(result, now)
        self._inbox.apply_rules(rules["results"], now, utc_now())
        result.update(alertInbox=self._inbox.snapshot(), ackToken=self.ack_token, monitorRules=rules)
        return result


class DashboardServer:
    def __init__(self, port: int, stale_ms: int = 1500) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("dashboard port must be in 0..65535")
        self.state = DashboardState(stale_ms)
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def _local_host(self) -> bool:
                return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

            def do_GET(self) -> None:
                if not self._local_host():
                    self._send(403, "text/plain", b"Invalid host")
                    return
                route = self.path.split("?", 1)[0]
                if route == "/":
                    self._send(200, "text/html; charset=utf-8", HTML)
                elif route == "/style.css":
                    self._send(200, "text/css; charset=utf-8", CSS)
                elif route == "/app.js":
                    self._send(200, "text/javascript; charset=utf-8", JS)
                elif route == "/reminders.js":
                    from pathlib import Path
                    self._send(200, "text/javascript; charset=utf-8",
                               Path(__file__).with_name("reminders.js").read_bytes())
                elif route == "/monitor_rules.js":
                    from pathlib import Path
                    self._send(200, "text/javascript; charset=utf-8",
                               Path(__file__).with_name("monitor_rules.js").read_bytes())
                elif route == "/api/state":
                    body = json.dumps(state.snapshot(), ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", body)
                elif route == "/api/frame.jpg":
                    frame, remaining = state.frame_payload()
                    self._send(200 if frame is not None else 404, "image/jpeg", frame or b'',
                               {'X-Frame-Remaining-Ms': str(remaining)})
                elif route == "/api/report.json":
                    snapshot = state.snapshot()
                    snapshot.pop("ackToken", None)
                    body = json.dumps({'kind': 'live_observation_metadata_not_physical_acceptance',
                        'physicalStopQualified': False, 'projectComplete': False,
                        **snapshot}, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
                    self._send(200, "application/json; charset=utf-8", body)
                elif route == "/favicon.ico":
                    self._send(204, "image/x-icon", b"")
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")

            def do_POST(self) -> None:
                # This route owns only local acknowledgement state, no serial/device handle.
                if self.path not in ("/api/alerts/ack", "/api/rules"):
                    self._send(501, "text/plain", b"No device control API")
                    return
                expected_origin = f"http://127.0.0.1:{self.server.server_port}"
                supplied_token = self.headers.get("X-Alert-Token", "")
                if (not self._local_host() or self.headers.get("Origin") != expected_origin
                        or not supplied_token.isascii()
                        or not secrets.compare_digest(supplied_token, state.ack_token)):
                    self._send(403, "text/plain", b"Local session required")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if (not 1 <= size <= (2048 if self.path == "/api/rules" else 256) or self.headers.get("Transfer-Encoding") is not None
                            or self.headers.get("Content-Type") != "application/json"):
                        raise ValueError("invalid request")
                    self.connection.settimeout(2)
                    data = json.loads(self.rfile.read(size), object_pairs_hook=unique_object)
                    if self.path == "/api/rules":
                        if not isinstance(data, dict) or set(data) != {"config", "revision"} or type(data["revision"]) is not int:
                            raise ValueError("invalid rule configuration")
                        if not state.configure_rules(data["config"], data["revision"]):
                            self._send(409, "text/plain", b"Rules changed in another tab; reload before editing")
                            return
                        self._send(200, "application/json", b'{"configured":true,"deviceCommandSent":false}')
                        return
                    if (not isinstance(data, dict) or set(data) != {"id"}
                            or not isinstance(data["id"], str) or len(data["id"]) > 20):
                        raise ValueError("invalid acknowledgement")
                except (OSError, ValueError, UnicodeError, RecursionError):
                    self._send(400, "text/plain", b"Invalid acknowledgement")
                    return
                if not state.acknowledge(data["id"]):
                    self._send(404, "text/plain", b"Reminder not found")
                    return
                self._send(200, "application/json", b'{"acknowledged":true,"deviceCommandSent":false}')

            def _send(self, status: int, content_type: str, body: bytes, headers=None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self' blob:; "
                    "connect-src 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="local-dashboard", daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/"

    def start(self) -> None:
        self._thread.start()

    def publish(self, record: dict) -> None:
        self.state.publish(record)

    def publish_frame(self, frame, detections, captured_ms: int, confidence: float) -> None:
        import cv2

        if time.monotonic_ns() // 1_000_000 - captured_ms > self.state.stale_ms:
            return
        height, width = frame.shape[:2]
        scale = min(1.0, 960 / max(height, width))
        preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        for detection in [d for d in detections if d.valid() and d.confidence >= confidence][:32]:
            x1, y1, x2, y2 = [round(v * scale) for v in detection.xyxy]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (60, 190, 100), 2)
            cv2.putText(preview, f'{detection.label} {detection.confidence:.0%}',
                        (max(0, x1), max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 190, 100), 1)
        ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.state.publish_jpeg(encoded.tobytes(), captured_ms)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
