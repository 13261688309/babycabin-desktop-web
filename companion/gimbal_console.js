(() => {
  const $=id=>document.getElementById(id), client=crypto.randomUUID();
  let current=null, editing=false, pending=false, cameraBusy=false, cameraError=null, imageBusy=false, imageUrl=null, online=false;
  let commandRevision=0;
  let displayFrame=null,choiceFrame=null,presetRevision=null,axesLoaded=false;
  let videoFrame=null,videoBusy=false,targetPicking=false,viewEpoch=0;
  const displaySamples=[];
  let careBusy=false,careRevision='';
  let trackingExportBusy=false;
  let patrolRevision=null;

  function cryWave(samples, sampleRate) {
    const wave = new DataView(new ArrayBuffer(44 + samples.length * 2));
    for (const [offset, text] of [[0, 'RIFF'], [8, 'WAVEfmt '], [36, 'data']]) {
      for (let i = 0; i < text.length; i++) wave.setUint8(offset + i, text.charCodeAt(i));
    }
    wave.setUint32(4, wave.byteLength - 8, true);
    wave.setUint32(16, 16, true);
    wave.setUint16(20, 1, true);
    wave.setUint16(22, 1, true);
    wave.setUint32(24, sampleRate, true);
    wave.setUint32(28, sampleRate * 2, true);
    wave.setUint16(32, 2, true);
    wave.setUint16(34, 16, true);
    wave.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, index) => {
      wave.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, sample)) * 32767), true);
    });
    return new Blob([wave], {type: 'audio/wav'});
  }

  async function captureCryAudio() {
    const context = new AudioContext();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const samples = new Float32Array(context.sampleRate * 5);
      let written = 0;
      source.connect(processor);
      processor.connect(context.destination);
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('麦克风没有在规定时间内提供音频')), 8500);
        processor.onaudioprocess = event => {
          const chunk = event.inputBuffer.getChannelData(0).subarray(0, samples.length - written);
          samples.set(chunk, written);
          written += chunk.length;
          if (written === samples.length) {
            clearTimeout(timeout);
            processor.onaudioprocess = null;
            resolve();
          }
        };
      });
      return cryWave(samples, context.sampleRate);
    } finally {
      stream?.getTracks().forEach(track => track.stop());
      await context.close();
    }
  }

  $('cryRecord').onclick = async () => {
    const button = $('cryRecord'), result = $('cryResult');
    button.disabled = true;
    try {
      const service = 'http://127.0.0.1:5000';
      const health = await fetch(service + '/api/health', {signal: AbortSignal.timeout(2500)});
      if (!health.ok || !(await health.json()).model_loaded) throw new Error('本机哭声模型未就绪');
      result.textContent = '正在录音（5 秒）…';
      const form = new FormData();
      form.append('audio', await captureCryAudio(), 'microphone.wav');
      result.textContent = '正在识别…';
      const response = await fetch(service + '/api/predict', {
        method: 'POST', body: form, signal: AbortSignal.timeout(25000)
      });
      const data = await response.json();
      if (!response.ok || typeof data.is_cry !== 'boolean' || !Number.isFinite(data.cry_probability)) {
        throw new Error(data.error || '未收到有效的识别结果');
      }
      result.textContent = `${data.is_cry ? '检测到疑似哭声' : '未检测到哭声'} · 模型分数 ${(data.cry_probability * 100).toFixed(1)}%`;
    } catch (error) {
      result.textContent = error.name === 'NotAllowedError' ? '请允许访问麦克风后重试'
        : '哭声识别失败：' + (error.message === 'Failed to fetch' ? '无法连接本机识别服务' : error.message);
    } finally {
      button.disabled = false;
    }
  };
  const recognitionHoldMs=5000;
  let lastRecognition=null,recognitionTimer;
  function showRecognition(generation,frameId,recognized,fallback){
    const now=performance.now();
    if(lastRecognition?.generation!==generation)lastRecognition=null;
    if(recognized&&Number.isFinite(frameId)&&lastRecognition?.frameId!==frameId){
      lastRecognition={generation,frameId,text:recognized,expires:now+recognitionHoldMs,missed:false};
    }else if(!recognized&&lastRecognition&&!lastRecognition.missed&&now<lastRecognition.expires){
      lastRecognition.missed=true;
      lastRecognition.expires=now+recognitionHoldMs;
    }
    const remaining=lastRecognition?lastRecognition.expires-now:0;
    const render=text=>{$('emotionState').textContent=$('recognitionSummary').textContent=text;};
    clearTimeout(recognitionTimer);
    if(remaining>0){
      render(!lastRecognition.missed&&recognized?recognized:`${lastRecognition.text} · 上次识别`);
      recognitionTimer=setTimeout(()=>render(fallback),Math.ceil(remaining));
    }else render(fallback);
  }
  let stateReceivedAt=0,candidateRevision='';
  const emotionOverlayMs=600;
  const emotionLabels={Cry:'哭泣表情',Happy:'笑脸',Neutral:'中性表情',Sleep:'闭眼状态'};
  const candidateText=item=>`${emotionLabels[item.label]||item.label} · 模型分数 ${(item.confidence*100).toFixed(1)}%`;
  function emotionPreview(){
    const emotion=current?.emotion,p=emotion?.preview;
    return online&&emotion.ready&&!emotion.error&&emotion.fresh&&p
      &&p.cameraGeneration===current.camera.generation&&p.frameAgeMs>=0
      &&p.frameAgeMs+performance.now()-stateReceivedAt<1500?p:null;
  }
  function recognitionView(){
    const emotion=current?.emotion;if(!emotion)return;
    const observation=emotion.observation,p=emotionPreview(),items=p?.detections||[];
    const reasons={no_detection:'未检测到目标',low_confidence:'低分候选，待确认',multiple_detections:'多目标待确认',ambiguous_class:'类别冲突，待确认',confirming:'等待连续确认'};
    const reason=reasons[observation.reason]||'待确认';
    const fresh=emotion.fresh&&(!emotion.preview||p);
    const immediate=observation.state==='Unknown'&&observation.reason==='confirming'
      &&Object.hasOwn(emotionLabels,observation.candidate)&&Number.isFinite(observation.confidence);
    const recognized=!online||emotion.error||!emotion.ready||!fresh?null:
      observation.state!=='Unknown'?`${emotionLabels[observation.state]||observation.state} · 连续确认${typeof observation.confidence==='number'?` · 模型分数 ${(observation.confidence*100).toFixed(1)}%`:''}`:
      immediate?`${candidateText({label:observation.candidate,confidence:observation.confidence})} · 即时识别`:null;
    const fallback=!online?'识别连接中断':emotion.error?'识别异常':!emotion.ready?'模型未就绪':!fresh||recognized?'等待新画面':
      items.length?`${candidateText(items[0])} · ${reason}`:`识别中 · ${reason}`;
    showRecognition(current.camera?.generation,emotion.frameId,recognized,fallback);
    const revision=JSON.stringify(items);
    if(revision!==candidateRevision){
      candidateRevision=revision;
      $('recognitionCandidates').replaceChildren(...items.slice(0,5).map(item=>{
        const tag=document.createElement('span');tag.textContent='候选：'+candidateText(item);return tag;
      }));
    }
  }
  const controls=['apply','center','auto','left','right','up','down','pan','tilt','reversePan','reverseTilt','rock','rockSpeed','rockDuration','armSoothe','targetChoice','clearTarget','presetSlot','presetName','presetApply','presetSave','presetDelete','patrolPoint1','patrolPoint2','patrolPoint3','patrolDwell','patrolFacePause','patrolStart','patrolPause','voiceEnable','voiceDisable','voiceChoice','voiceSimulate'];
  const states={waiting:'等待目标',acquiring:'连续帧锁定中',centered:'目标已居中',tracking:'跟踪中',lost:'目标丢失',multiple:'多个目标，停止跟踪',changed:'目标变化',stale:'画面过期',invalid:'无效目标'};
  function values(){ $('panValue').textContent=$('pan').value+'°';$('tiltValue').textContent=$('tilt').value+'°'; }
  function presetFields(){
    const slot=$('presetSlot').value,p=current?.presets?.slots?.[slot];
    $('presetName').value=p?.name||`观察位${slot}`;
  }
  function presets(s,blocked){
    const data=s.presets||{slots:{},revision:0},slot=$('presetSlot').value;
    if(presetRevision!==data.revision){
      $('presetSlot').replaceChildren(...['1','2','3'].map(key=>{
        const p=data.slots[key],option=document.createElement('option');option.value=key;
        option.textContent=`${key} · ${p?p.name:'未保存'}`;return option;
      }));
      $('presetSlot').value=slot;presetRevision=data.revision;presetFields();
    }
    const selected=data.slots[slot],idle=s.mode==='idle';
    $('presetSlot').disabled=blocked||!['idle','manual'].includes(s.mode);
    $('presetApply').disabled=blocked||!selected||!['idle','manual'].includes(s.mode);
    ['presetName','presetSave'].forEach(id=>$(id).disabled=blocked||!idle||Boolean(data.error));
    $('presetDelete').disabled=blocked||!idle||!selected||Boolean(data.error);
    $('presetStatus').textContent=data.error||(selected?`${selected.pan}° / ${selected.tilt}° · 水平${selected.panSign===1?'正向':'反向'} · 俯仰${selected.tiltSign===1?'正向':'反向'}`:'尚未保存');
  }
  function visibleFrame(){
    const frame=targetPicking?displayFrame:videoFrame;
    if(!frame||!online||frame.camera.generation!==current?.camera.generation
        ||frame.frameAgeMs+performance.now()-frame.receivedAt>=(targetPicking?400:2000))return null;
    if(targetPicking)return current.frameFresh?frame:null;
    const aligned=current.frameFresh&&frame.frameAgeMs+performance.now()-frame.receivedAt<400
      &&current.width===frame.width&&current.height===frame.height;
    return {...frame,faces:aligned?current.faces:[],selection:aligned?current.selection:{box:null}};
  }
  function rates(){
    const now=performance.now();while(displaySamples.length&&now-displaySamples[0]>2000)displaySamples.shift();
    const count=displaySamples.length,elapsed=count>1?displaySamples[count-1]-displaySamples[0]:0;
    const shown=online&&count>1&&now-displaySamples[count-1]<400&&elapsed>0?(count-1)*1000/elapsed:0;
    const fps=current?.camera?.fps||{};
    $('videoRates').textContent=`取流 ${online?(fps.capture||0).toFixed(1):'--'} · 显示 ${shown.toFixed(1)} · 人脸 ${online?(fps.face||0).toFixed(1):'--'} · 表情 ${online?(fps.emotion||0).toFixed(1):'--'} fps${targetPicking?' · 选目标画面':' · 显示上限30fps'}`;
  }
  function pickMode(enabled){
    targetPicking=enabled;viewEpoch++;displayFrame=videoFrame=choiceFrame=null;displaySamples.length=0;
    if(imageUrl)URL.revokeObjectURL(imageUrl);imageUrl=null;
    $('pickTarget').textContent=enabled?'返回实时画面':'选择目标';$('pickTarget').setAttribute('aria-pressed',String(enabled));
    if(current){drawFrame();if(enabled)fetchFrame(current);}
  }
  function patrolSlots(){return ['patrolPoint1','patrolPoint2','patrolPoint3'].map(id=>$(id).value).filter(Boolean);}
  function patrolView(s,blocked){
    const data=s.presets||{slots:{},revision:0},points=['patrolPoint1','patrolPoint2','patrolPoint3'];
    if(patrolRevision!==data.revision){
      const keys=Object.keys(data.slots).sort();
      points.forEach((id,index)=>{
        const prior=$(id).value,options=[new Option(index===2?'不设置':'请选择',''),...keys.map(key=>new Option(`${key} · ${data.slots[key].name}`,key))];
        $(id).replaceChildren(...options);$(id).value=keys.includes(prior)?prior:index<2?(keys[index]||''):'';
      });patrolRevision=data.revision;
    }
    const slots=patrolSlots(),idle=s.mode==='idle',active=s.mode==='patrol',plan=s.patrol;
    let prior=[s.status?.pan??s.pan,s.status?.tilt??s.tilt],total=0;
    for(const key of slots){const p=data.slots[key];if(!p)continue;total+=Math.ceil(Math.max(Math.abs(p.pan-prior[0]),Math.abs(p.tilt-prior[1]))*100)+Number($('patrolDwell').value);prior=[p.pan,p.tilt];}
    const limit=s.patrolLimitMs??10000;
    const bad=!$('patrolPoint1').value||!$('patrolPoint2').value||slots.length<2||new Set(slots).size!==slots.length||slots.some(slot=>!data.slots[slot]);
    const faceBlocked=$('patrolFacePause').checked&&(!s.frameFresh||s.faces.length>0);
    [...points,'patrolDwell','patrolFacePause'].forEach(id=>$(id).disabled=blocked||!idle);
    $('patrolStart').disabled=blocked||!idle||bad||total>limit||faceBlocked||Boolean(s.hardware&&s.status?.trip);
    $('patrolPause').disabled=blocked||!active||(plan?.paused&&plan.pauseOnFace&&(!s.frameFresh||s.faces.length>0));
    $('patrolPause').textContent=plan?.paused&&active?'继续巡航':'暂停换点';
    $('patrolEstimate').textContent=active?`目标变化限速10°/秒；暂停不延长${limit/1000}秒截止时间`:bad?'请选择2至3个不同的已保存预置位':faceBlocked?'人脸暂停需要新鲜且暂无人脸的画面':`目标路线预计 ${(total/1000).toFixed(1)} 秒 / 上限${limit/1000}秒${total>limit?'，请缩短角差或停留':''}`;
    const point=plan?.points?.[plan.index];
    $('patrolProgress').textContent=!plan?'尚未运行巡航':active?`${plan.paused?'已暂停':plan.phase==='moving'?'目标渐变':'目标停留'} · 第${plan.index+1}/${plan.points.length}站 · ${point.name}${plan.pauseReason==='face'?' · 人脸触发，需人工继续':''}`:`上次巡航：${plan.reason}`;
  }
  function voiceView(s,blocked){
    const voice=s.voiceControl||{},armed=Boolean(voice.armed);
    $('voiceWindow').textContent=armed?`${(voice.remainingMs/1000).toFixed(1)} 秒`:'未启用';
    $('voiceEnable').disabled=blocked||armed||s.mode!=='idle'||Boolean(s.hardware&&(!voice.associatedDataFresh||s.status?.trip));
    $('voiceDisable').disabled=pending||!armed;
    $('voiceExtensionState').textContent=voice.message||'扩展语音未启用';
    $('voicePreview').hidden=Boolean(s.hardware);
    $('voiceChoice').disabled=$('voiceSimulate').disabled=blocked||!armed||s.hardware;
    $('voiceEvidence').textContent=s.hardware?'需要增强 voice_cmd 回执及对应词表；本次未配置语音板':'软件口令预演，非语音识别；语音板新词表尚未配置';
    const last=voice.lastCommand;
    $('voiceLast').textContent=last?`${last.phrase} · ${last.result}`:'尚无扩展口令';
    if(armed)['patrolPoint1','patrolPoint2','patrolPoint3','patrolDwell','patrolFacePause','presetSave','presetDelete'].forEach(id=>$(id).disabled=true);
  }
  function drawFrame(){
    const view=visibleFrame(),canvas=$('overlay'),ctx=canvas.getContext('2d');
    const cw=Math.round(canvas.clientWidth*devicePixelRatio),ch=Math.round(canvas.clientHeight*devicePixelRatio);
    if(canvas.width!==cw||canvas.height!==ch){canvas.width=cw;canvas.height=ch;}
    const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
    canvas.dataset.emotionBoxes='0';
    $('empty').hidden=Boolean(view&&imageUrl);$('camera').hidden=!view||!imageUrl;
    const delayed=Boolean(view&&!targetPicking&&view.frameAgeMs+performance.now()-view.receivedAt>=400);
    $('videoDelay').hidden=!delayed;
    $('videoDelay').textContent=delayed?'最近画面 · 已延迟，自动控制暂停':'';
    ctx.strokeStyle='#8b9c94';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();ctx.moveTo(w/2-12,h/2);ctx.lineTo(w/2+12,h/2);ctx.moveTo(w/2,h/2-12);ctx.lineTo(w/2,h/2+12);ctx.stroke();
    const selectable=Boolean(targetPicking&&view&&current.mode==='idle'&&!pending&&!cameraBusy);
    $('pickTarget').disabled=!online||pending||cameraBusy||current?.mode!=='idle'||!current?.frameFresh;
    canvas.dataset.selectable=String(selectable);$('targetChoice').disabled=!selectable||!view.faces.length;
    $('clearTarget').disabled=!online||pending||current?.mode!=='idle'||current?.selection?.mode!=='selected';
    if(view){
      const scale=Math.min(w/view.width,h/view.height),ox=(w-view.width*scale)/2,oy=(h-view.height*scale)/2;
      ctx.strokeStyle='#819b90';ctx.setLineDash([4*devicePixelRatio,4*devicePixelRatio]);
      ctx.strokeRect(ox+view.width*.45*scale,oy+view.height*.45*scale,view.width*.10*scale,view.height*.10*scale);
      ctx.setLineDash([]);
      ctx.lineWidth=2*devicePixelRatio;ctx.font=`${14*devicePixelRatio}px sans-serif`;
      view.faces.forEach(([x1,y1,x2,y2],index)=>{
        const selected=view.selection.box&&view.selection.box.every((v,i)=>v===view.faces[index][i]);
        ctx.strokeStyle=selected?'#126f57':'#a56b08';ctx.fillStyle=ctx.strokeStyle;
        ctx.strokeRect(ox+x1*scale,oy+y1*scale,(x2-x1)*scale,(y2-y1)*scale);
        ctx.fillText(String(index+1)+(selected?' ✓':''),ox+x1*scale+4*devicePixelRatio,oy+y1*scale+18*devicePixelRatio);
      });
      // Recent inference is a display overlay only, never a selectable tracking target.
      const p=emotionPreview();
      if(!targetPicking&&p&&p.width===view.width&&p.height===view.height
          &&view.frameId>=p.frameId&&view.frameId-p.frameId<emotionOverlayMs
          &&p.frameAgeMs+performance.now()-stateReceivedAt<emotionOverlayMs){
        ctx.strokeStyle='#126f9b';ctx.fillStyle='#126f9b';ctx.setLineDash([5*devicePixelRatio,3*devicePixelRatio]);
        p.detections.forEach(item=>{
          const [x1,y1,x2,y2]=item.xyxy;
          ctx.strokeRect(ox+x1*scale,oy+y1*scale,(x2-x1)*scale,(y2-y1)*scale);
          const label=`候选 ${emotionLabels[item.label]||item.label} ${(item.confidence*100).toFixed(1)}%`;
          const tw=Math.min(w-8*devicePixelRatio,ctx.measureText(label).width+8*devicePixelRatio);
          const tx=Math.max(0,Math.min(w-tw,ox+x1*scale)),ty=Math.max(0,oy+y1*scale-22*devicePixelRatio);
          ctx.fillRect(tx,ty,tw,22*devicePixelRatio);ctx.fillStyle='#ffffff';
          ctx.fillText(label,tx+4*devicePixelRatio,ty+16*devicePixelRatio,tw-8*devicePixelRatio);ctx.fillStyle='#126f9b';
        });
        canvas.dataset.emotionBoxes=String(p.detections.length);ctx.setLineDash([]);
      }
      const sample=view.trackingMetrics?.lastSample;
      if(current.mode==='auto'&&view.trackingMetrics?.sampleFresh&&sample?.frameId===view.frameId){
        const [x1,y1,x2,y2]=sample.box;
        ctx.strokeStyle='#1b698c';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();
        ctx.moveTo(w/2,h/2);ctx.lineTo(ox+(x1+x2)/2*scale,oy+(y1+y2)/2*scale);ctx.stroke();
      }
    }
    const optionsFrame=targetPicking?view:(current?.frameFresh?current:null);
    if(document.activeElement!==$('targetChoice')&&choiceFrame?.frameId!==optionsFrame?.frameId){
      const options=[new Option('点选目标',''),...(optionsFrame?.faces||[]).map((_,index)=>new Option(`目标 ${index+1}`,String(index)))];
      $('targetChoice').replaceChildren(...options);
      choiceFrame=optionsFrame;
    }
  }
  function fetchFrame(s){
    if(!targetPicking||imageBusy||!s.frameFresh||s.frameId===displayFrame?.frameId)return;
    imageBusy=true;const receivedAt=performance.now(),epoch=viewEpoch;
    fetch('/frame.jpg',{cache:'no-store',signal:AbortSignal.timeout(600)})
      .then(async r=>{if(!r.ok)throw Error();return {blob:await r.blob(),meta:JSON.parse(r.headers.get('X-Frame-Meta'))}})
      .then(async ({blob,meta})=>{
        const url=URL.createObjectURL(blob),image=new Image();image.src=url;
        try{await image.decode();}catch(error){URL.revokeObjectURL(url);throw error;}
        if(!meta||!targetPicking||epoch!==viewEpoch||!online||meta.camera.generation!==current?.camera.generation
            ||!current?.frameFresh||meta.frameAgeMs+performance.now()-receivedAt>=400){URL.revokeObjectURL(url);return;}
        const old=imageUrl;imageUrl=url;$('camera').src=url;
        displayFrame={...s,...meta,receivedAt};if(old)URL.revokeObjectURL(old);displaySamples.push(performance.now());drawFrame();
      }).catch(()=>{}).finally(()=>{imageBusy=false});
  }
  async function videoLoop(){
    const began=performance.now(),epoch=viewEpoch,generation=current?.camera?.generation;
    if(!targetPicking&&online&&!document.hidden&&!videoBusy&&['connecting','streaming'].includes(current?.camera?.state)){
      videoBusy=true;
      try{
        const response=await fetch('/video.jpg?after='+(videoFrame?.frameId||0),{cache:'no-store',signal:AbortSignal.timeout(600)});
        if(response.status===200){
          const meta={frameId:Number(response.headers.get('X-Frame-Id')),frameAgeMs:Number(response.headers.get('X-Frame-Age-Ms')),
            width:Number(response.headers.get('X-Frame-Width')),height:Number(response.headers.get('X-Frame-Height')),
            camera:{generation:Number(response.headers.get('X-Camera-Generation'))},receivedAt:began};
          const blob=await response.blob(),url=URL.createObjectURL(blob),image=new Image();image.src=url;
          try{await image.decode();}catch(error){URL.revokeObjectURL(url);throw error;}
          if(targetPicking||epoch!==viewEpoch||!online||generation!==current?.camera?.generation||meta.camera.generation!==generation
              ||meta.frameAgeMs+performance.now()-began>=2000){URL.revokeObjectURL(url);}
          else{const old=imageUrl;imageUrl=url;videoFrame=meta;$('camera').src=url;if(old)URL.revokeObjectURL(old);displaySamples.push(performance.now());}
        }
      }catch{}finally{videoBusy=false;}
    }
    drawFrame();rates();recognitionView();
    setTimeout(videoLoop,Math.max(0,1000/30-(performance.now()-began)));
  }
  async function selectTarget(index,view=visibleFrame()){
    if(!targetPicking||!view||current.mode!=='idle'||pending||cameraBusy)return;
    const result=await command('select_target',{frameId:view.frameId,index});
    if(result)pickMode(false);
  }
  function deviceFeedback(s){
    const d=s.device||{},report=d.report?.fresh?d.report.data:null,st=d.status?.fresh?d.status.data:null,voice=d.voiceRx?.fresh?d.voiceRx.data:null;
    const health=report?.sensorHealth?.dht11;
    const number=(value,unit)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(1)+unit:'--';
    $('temperature').textContent=health==='ok'?number(report.temp,' °C'):'--';
    $('humidity').textContent=health==='ok'?number(report.hum,' %RH'):'--';
    $('sensorHealth').textContent=report?({ok:'正常',read_failed:'读取失败',stale:'过期',disabled:'无数据'}[health]||'未知'):d.report?'过期':'无数据';
    if(health==='read_failed'&&st?.dhtResult===1&&st.dhtRaw){
      $('sensorHealth').textContent=`读取超时（收到 ${st.dhtRaw.bytesReceived}/5 字节）`;
    }
    $('sensorHealth').dataset.warning=Boolean(d.report&&(!report||health!=='ok'));
    $('deviceConnection').textContent=!s.hardware&&!d.readOnly?'无主控连接':!d.connected?'主控回执中断':st||report?'遥测在线':'等待遥测';
    $('executionState').textContent={preview:'离线预演，无硬件输出',offline:'回执中断',stopping:'停止指令待发送',observing:'等待视觉触发',idle:'未发动作',pending:'等待主控接纳',accepted:'控制会话已接纳（非实测）'}[s.controlState]||'--';
    const trips={1:'STOP 回路触发',2:'ARM / READY 反馈异常',3:'软件保护触发',4:'执行器电源反馈异常'};
    $('protectionState').textContent=!st?'--':st.il_trip?trips[st.il_trip]:st.il_stop!==1?'STOP 未就绪':st.il_arm!==1?'等待 ARM':st.il_pending===1?'等待执行器电源':st.il_en===1?'执行使能已请求':st.il_ready===1?'保护就绪，未使能':'READY 未就绪';
    $('protectionState').dataset.warning=Boolean(st&&st.il_trip);
    $('protectionInputs').textContent=st?`${st.il_stop} / ${st.il_arm} / ${st.il_pg}`:'--';
    const id=st?.voice;
    const voiceNames={0:'尚无命令',1:'停止',2:'停止',3:'休眠',4:'前进',5:'后退',6:'向左',7:'向右',8:'左旋',9:'右旋',38:'报警'};
    const phrase=voiceNames[id]||s.voiceControl?.commands?.find(command=>command.id===id)?.phrase;
    $('voiceCommand').textContent=typeof id==='number'?`${phrase||'其他命令'} · 0x${id.toString(16).padStart(2,'0').toUpperCase()}`:'--';
    $('voiceCounts').textContent=voice?`${voice.stops} / ${voice.v===2?voice.applied:'--'}`:'--';
    $('deviceAlert').textContent=d.alert?.fresh?d.alert.data.alert:'--';
  }
  function careView(s){
    const care=s.care||{},inbox=care.alertInbox||{items:[],pending:0},run=care.followup;
    $('carePending').textContent=inbox.pending?`${inbox.pending} 项待查看`:'无待查看';
    const revision=JSON.stringify([inbox.items.map(item=>[item.id,item.title,item.detail,item.source,item.condition,item.acknowledged]),care.events,careBusy]);
    if(revision!==careRevision){
      careRevision=revision;
      const items=[...inbox.items].reverse();
      const visible=[...items.filter(item=>!item.acknowledged),...items.filter(item=>item.acknowledged)].slice(0,6);
      $('careAlerts').replaceChildren(...visible.map(item=>{
        const row=document.createElement('article');row.className='care-alert';row.dataset.condition=item.condition;row.dataset.alertId=item.id;
        const text=document.createElement('div'),title=document.createElement('strong'),detail=document.createElement('p'),state=document.createElement('span');
        title.textContent=item.title;detail.textContent=item.detail;
        state.textContent=`${item.source} · ${{active:'条件仍存在',unverified:'当前状态待核实',recovered:'条件已恢复'}[item.condition]||'待核实'}${item.acknowledged?' · 已查看':''}`;
        text.append(title,detail,state);row.append(text);
        if(!item.acknowledged){const button=document.createElement('button');button.type='button';button.textContent='已查看';button.disabled=careBusy||!online;button.onclick=()=>acknowledge(item.id);row.append(button);}
        return row;
      }));
      if(!visible.length){const empty=document.createElement('p');empty.className='muted';empty.textContent='暂无提醒';$('careAlerts').append(empty);}
      $('careEvents').replaceChildren(...[...(care.events||[])].reverse().slice(0,10).map(event=>{
        const row=document.createElement('li'),time=document.createElement('time'),text=document.createElement('span');
        const date=new Date(event.utc);time.textContent=Number.isNaN(date.valueOf())?'--':date.toLocaleTimeString('zh-CN',{hour12:false});
        text.textContent=event.text;row.append(time,text);return row;
      }));
    }
    const states={executing:'等待本次控制结束',following:'只读观察中',interrupted:'流程已中断',unavailable:'观察未开始',insufficient:'结果不足',cry_persists:'Cry 仍持续',label_changed:'视觉标签已变化',observed:'观察已完成'};
    $('followupState').textContent=run?(states[run.state]||'未知')+(run.state==='following'?` · ${(run.remainingMs/1000).toFixed(0)} 秒`:''):'尚未开始';
    $('followupReason').textContent=run?run.reason:'暂无本场摇摆观察';
    $('beforeLabel').textContent=run?run.before:'--';$('afterLabel').textContent=run?run.after:'--';
    $('followupEvidence').textContent=run?`${run.hardware?'硬件控制候选':'软件预演'} · ${run.samples} 份新画面 · ${run.confirmedSamples} 份连续确认 · 非实测动作或安抚效果`:'--';
  }
  function trackingView(s){
    const m=s.trackingMetrics,sample=online&&s.mode==='auto'&&m?.sampleFresh?m.lastSample:null;
    const displacement=(value,positive,negative)=>Math.abs(value)<.05?'居中':`${value>0?positive:negative} ${Math.abs(value).toFixed(1)}%`;
    $('trackingDx').textContent=sample?displacement(sample.dxPercent,'右','左'):'--';
    $('trackingDy').textContent=sample?displacement(sample.dyPercent,'下','上'):'--';
    $('trackingTime').textContent=m?(m.validTrackingMs/1000).toFixed(1)+' 秒':'--';
    $('trackingCentered').textContent=m?.centeredFramePercent!==null&&m?.centeredFramePercent!==undefined?m.centeredFramePercent.toFixed(1)+'%':'--';
    $('trackingAcquire').textContent=m?.acquisitionMs!==null&&m?.acquisitionMs!==undefined?(m.acquisitionMs/1000).toFixed(2)+' 秒':'--';
    $('trackingFrames').textContent=m?`${m.lockedFrames} 帧`:'--';
    $('trackingExport').disabled=!online||!m||trackingExportBusy;
    const limits=sample?.targetAngleLimits||[];
    $('trackingReason').textContent=!m?'尚无本次跟踪':!online?'控制页断联，当前偏差已隐藏':!m.active?m.reason:limits.length?`${limits.map(axis=>axis==='pan'?'水平':'俯仰').join('、')}目标角度已到限位，不扩大行程`:!sample?'等待新画面或连续帧锁定':sample.centered?'目标位于原始画面居中范围内':'正在修正画面偏差';
  }
  $('trackingExport').onclick=async()=>{
    if(!online||!current?.trackingMetrics||trackingExportBusy)return;
    trackingExportBusy=true;trackingView(current);
    try{
      const response=await fetch('/api/tracking-report',{cache:'no-store',signal:AbortSignal.timeout(1000)});
      if(!response.ok)throw Error('统计导出失败');
      const report=await response.json();if(!report.run)throw Error('尚无本次跟踪');
      const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})),link=document.createElement('a');
      link.href=url;link.download=`tracking-run-${report.run.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }catch(error){$('trackingReason').textContent=error.message;}
    finally{trackingExportBusy=false;$('trackingExport').disabled=!online||!current?.trackingMetrics;}
  };
  async function acknowledge(id){
    if(!online||!current||careBusy)return;
    careBusy=true;$('careError').hidden=true;careView(current);
    try{
      const response=await fetch('/api/care',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':current.token},body:JSON.stringify({action:'ack',id}),signal:AbortSignal.timeout(1000)});
      if(!response.ok)throw Error(await response.text());
      // Local acknowledgement does not modify the control generation, lease or heartbeat.
      const state=await response.json();current.care=state.care;
    }catch(error){$('careError').textContent=error.message;$('careError').hidden=false;}
    finally{careBusy=false;careView(current);}
  }
  function paint(s,receivedAt=performance.now()){
    current=s;online=true;stateReceivedAt=receivedAt;
    $('connection').textContent=s.servoBench?(s.status?'GB01 云台实机 · 已连接':'GB01 云台实机 · 离线'):s.hardware?'硬件控制候选':'离线预演 · 无硬件输出';
    $('mode').textContent={idle:'未启动',manual:'手动角度',auto:'人脸自动跟踪',observe:'等待持续 Cry',rock:'限时电机摇摆',patrol:s.patrol?.paused?'巡航已暂停':'预置位巡航'}[s.mode];
    $('mobileMode').textContent=$('mode').textContent;
    $('countdown').textContent=s.mode==='idle'?'--':(s.remainingMs/1000).toFixed(1)+' 秒';
    $('message').textContent=s.message;
    deviceFeedback(s);
    careView(s);
    trackingView(s);
    $('faceState').textContent=s.frameFresh?(states[s.tracker]||s.tracker):'画面不可用';
    $('feedback').textContent=s.status?`${s.status.pan}° / ${s.status.tilt}°（非实测）`:'未连接';
    $('link').textContent=s.hardware?(s.status?(s.servoBench?'GB01 云台实机':'GC03 控制候选'):'等待新固件回执'):'无设备输出';
    const cam=s.camera||{state:'disconnected'};
    $('cameraState').textContent={disconnected:'相机未连接',connecting:'正在连接相机',streaming:'相机已连接',error:'取图失败，等待连接恢复'}[cam.state]||'相机状态未知';
    if(cam.state==='streaming')$('cameraState').textContent+=` · ${cam.frames} 帧`;
    if(cam.transport)$('cameraState').textContent+=cam.transport==='mjpeg'?' · 连续视频':' · 逐张取图';
    if(cam.fallbackReason)$('cameraState').textContent+=' · 视频流暂不可用，将自动重试';
    if(cam.error)$('cameraState').textContent+=` · ${cam.error}`;
    if(cameraError)$('cameraState').textContent=cameraError;
    if(cam.source&&!$('source').value)$('source').value=cam.source;
    $('connectCamera').disabled=cameraBusy;
    $('disconnectCamera').disabled=cameraBusy||cam.state==='disconnected';
    $('analysisRotation').value=String(cam.analysisRotation||0);
    $('analysisRotation').disabled=cameraBusy||pending||s.mode!=='idle'||s.controlState==='stopping'
      ||Boolean(s.hardware&&(!s.status||s.status.lease!==0));
    if(!editing){$('pan').value=s.pan;$('tilt').value=s.tilt;values();}
    const blocked=pending||(s.hardware&&!s.status);
    controls.forEach(id=>$(id).disabled=blocked);
    $('auto').disabled=blocked||!s.frameFresh||s.mode!=='idle'||(s.selection?.mode==='selected'&&!s.selection.box);
    ['pan','tilt','apply','center','up','down','left','right'].forEach(id=>$(id).disabled=blocked||!['idle','manual'].includes(s.mode));
    ['reversePan','reverseTilt'].forEach(id=>$(id).disabled=blocked||s.mode!=='idle');
    if(!axesLoaded&&s.axes){$('reversePan').checked=s.axes.panSign===-1;$('reverseTilt').checked=s.axes.tiltSign===-1;axesLoaded=true;}
    presets(s,blocked);
    patrolView(s,blocked);
    voiceView(s,blocked);
    const selection=s.selection||{mode:'single'};
    $('selectionState').textContent=selection.mode==='single'?'自动选择单个人脸':!selection.box?'所选目标已失效，请重新选择':s.mode==='auto'?(['tracking','centered'].includes(s.tracker)?'正在跟踪所选目标':'正在锁定所选目标'):'目标已选，尚未启动跟踪';
    ['rock','rockSpeed','rockDuration'].forEach(id=>$(id).disabled=blocked||s.mode!=='idle');
    const armed=s.mode==='observe'||(s.mode==='rock'&&s.motor.source==='vision');
    if(armed){$('rockSpeed').value='20';$('rockDuration').value='3000';}
    $('armSoothe').disabled=blocked||(!armed&&(s.mode!=='idle'||!s.emotion.ready||!s.emotion.fresh||!s.frameFresh));
    $('armSoothe').textContent=armed?'取消自动安抚':'启用自动安抚';
    recognitionView();
    $('sootheState').textContent=armed?(s.mode==='observe'?'等待单目标持续Cry 3秒，触发一次3秒摇摆':'本次自动摇摆进行中，完成后停止'):'自动安抚未启用';
    const motorTarget=s.hardware?(s.status?.motor||0):s.motor.targetPercent;
    $('motorState').textContent=motorTarget?`${motorTarget>0?'正向':'反向'} ${Math.abs(motorTarget)}%（目标）`:'未运行';
    $('stop').disabled=false;
    $('tiltProfile').hidden=!s.servoBench;
    document.querySelector('.voice-extension').hidden=Boolean(s.servoBench);
    document.querySelector('.soothe').hidden=Boolean(s.servoBench);
    if(s.servoBench){
      $('servoConnect').disabled=pending||!!s.status||s.mode!=='idle';
      $('servoDisconnect').disabled=pending||!s.status||s.mode!=='idle'||s.controlState==='stopping';
      $('tilt').min=String(s.tiltRange[0]);$('tilt').max=String(s.tiltRange[1]);
      if (s.panRange) { $('pan').min=String(s.panRange[0]);$('pan').max=String(s.panRange[1]); }
      $('servoAxes').value=s.servoAxes;
      $('servoAxes').disabled=!s.canSelectServoAxes||blocked||s.mode!=='idle'||s.tiltReady||s.controlState==='stopping';
      $('servoScope').textContent=s.tiltOnly?'仅俯仰，85–95°，单次6秒。切换接线范围前先断舵机电。':`双轴实机：手动角度最长10秒，巡航最长${(s.patrolLimitMs||10000)/1000}秒，人脸跟踪最长${(s.trackingSessionLimitMs||10000)/1000}秒。`;
      if (s.trackingCameraMount==='fixed') document.querySelector('.tracking h2').textContent='固定镜头跟踪';
      $('servoReadyLabel').textContent=(s.tiltOnly?'仅俯仰接A7，水平三线断开；':'水平接A6、俯仰接A7；')+'独立5V/CV且共地，空载固定，可随时断电';
      $('tiltReady').checked=s.tiltReady;
      $('tiltReady').disabled=blocked||s.mode!=='idle'||s.controlState==='stopping';
      ['voiceEnable','voiceDisable','voiceSimulate','rock','armSoothe'].forEach(id=>{
         $(id).disabled=true;$(id).title='GB01固件未开放此接口，需对应固件和实物接线';
       });
      $('voiceExtensionState').textContent='当前GB01未启用外部语音接收，需恢复对应固件和语音接线';
      $('sootheState').textContent='当前GB01不输出电机控制，需电机固件、驱动与供电接线';
      ['pan','left','right','tilt','apply','center','up','down','presetApply','auto','patrolStart','patrolPause'].forEach(id=>{
        $(id).disabled=$(id).disabled||!s.tiltReady||s.controlState==='stopping';
      });
      if(s.tiltOnly){
        $('pan').value='90';values();
        ['pan','left','right','auto','reversePan','reverseTilt','patrolStart','patrolPause'].forEach(id=>$(id).disabled=true);
      }
      $('up').disabled=$('up').disabled||+$('tilt').value>=+$('tilt').max;
      $('down').disabled=$('down').disabled||+$('tilt').value<=+$('tilt').min;
    }
    if(displayFrame?.frameId===s.frameId)displayFrame.selection=s.selection;
    drawFrame();fetchFrame(s);rates();
  }
  async function cameraCommand(action){
    if(!current||!online||cameraBusy)return;
    cameraBusy=true;cameraError=null;$('connectCamera').disabled=$('disconnectCamera').disabled=true;
    $('analysisRotation').disabled=true;
    try{
      const data={action,generation:current.generation};
      if(action==='connect')data.source=$('source').value;
      if(action==='analysis_rotation')data.rotation=Number($('analysisRotation').value);
      const receivedAt=performance.now();
      const response=await fetch('/api/camera',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':current.token},body:JSON.stringify(data),signal:AbortSignal.timeout(1200)});
      if(!response.ok)throw Error(await response.text());
      paint(await response.json(),receivedAt);
    }catch(error){cameraError=error.message;$('cameraState').textContent=cameraError;}
    finally{cameraBusy=false;}
  }
  $('cameraForm').addEventListener('submit',e=>{e.preventDefault();cameraCommand('connect')});
  $('source').addEventListener('input',()=>{cameraError=null});
  $('disconnectCamera').onclick=()=>cameraCommand('disconnect');
  $('analysisRotation').onchange=()=>cameraCommand('analysis_rotation');
  async function command(action,extra={}){
    if(!current||!online)return;
    const heartbeat=action==='heartbeat';
    if(pending&&action!=='stop')return;
    const revision=heartbeat?commandRevision:++commandRevision;
    if(!heartbeat){pending=true;controls.forEach(id=>$(id).disabled=true)}
    try{
      const receivedAt=performance.now();
      const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':current.token},body:JSON.stringify({action,client,generation:current.generation,...extra}),signal:AbortSignal.timeout(1000)});
      if(!response.ok)throw Error(await response.text());
      if(heartbeat)return;
      const data=await response.json();
      if(revision!==commandRevision)return;
      pending=false;
      if(!['preset_save','preset_delete','select_target','clear_target'].includes(action))editing=false;
      paint(data,receivedAt);return data;
    }catch(error){
      if(!heartbeat&&revision===commandRevision){
        pending=false;
        $('message').textContent=error.message;
      }
    }
  }
  ['pan','tilt'].forEach(id=>$(id).addEventListener('input',()=>{editing=true;values()}));
  $('angles').addEventListener('submit',e=>{e.preventDefault();command('manual',{pan:+$('pan').value,tilt:+$('tilt').value})});
  $('center').onclick=()=>command('center');$('stop').onclick=$('mobileStop').onclick=()=>command('stop');
  $('tiltReady').onchange=()=>command('tilt_ready',{value:$('tiltReady').checked});
  $('servoAxes').onchange=()=>command('servo_setup',{axes:$('servoAxes').value});
  async function servoConnection(action){
    if(!current||!online||pending)return;
    pending=true;
    $('servoConnect').disabled=$('servoDisconnect').disabled=true;
    try{
      const receivedAt=performance.now();
      const r=await fetch('/api/controller',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':current.token},
        body:JSON.stringify({action,generation:current.generation}),signal:AbortSignal.timeout(5000)});
      if(!r.ok)throw Error(await r.text());
      paint(await r.json(),receivedAt);
    }catch(error){$('message').textContent=error.message;}
    finally{pending=false;}
  }
  $('servoConnect').onclick=()=>servoConnection('connect');
  $('servoDisconnect').onclick=()=>servoConnection('disconnect');
  $('presetSlot').onchange=()=>{presetFields();if(current)presets(current,pending)};
  $('presetSave').onclick=()=>command('preset_save',{slot:$('presetSlot').value,name:$('presetName').value,
    pan:+$('pan').value,tilt:+$('tilt').value,panSign:$('reversePan').checked?-1:1,tiltSign:$('reverseTilt').checked?-1:1});
  $('presetDelete').onclick=()=>command('preset_delete',{slot:$('presetSlot').value});
  $('presetApply').onclick=async()=>{const s=await command('preset_apply',{slot:$('presetSlot').value});if(s){$('reversePan').checked=s.axes.panSign===-1;$('reverseTilt').checked=s.axes.tiltSign===-1;}};
  $('clearTarget').onclick=()=>command('clear_target');
  $('pickTarget').onclick=()=>pickMode(!targetPicking);
  $('patrolForm').addEventListener('submit',event=>{event.preventDefault();command('patrol',{slots:patrolSlots(),dwell:Number($('patrolDwell').value),pauseOnFace:$('patrolFacePause').checked});});
  $('patrolPause').onclick=()=>command(current?.patrol?.paused?'patrol_resume':'patrol_pause');
  $('voiceEnable').onclick=()=>{
    const slots=patrolSlots(),valid=slots.length>=2&&new Set(slots).size===slots.length;
    command('voice_arm',{slots:valid?slots:[],dwell:Number($('patrolDwell').value),pauseOnFace:$('patrolFacePause').checked});
  };
  $('voiceDisable').onclick=()=>command('voice_disarm');
  $('voiceSimulate').onclick=()=>command('voice_simulate',{id:Number($('voiceChoice').value)});
  ['patrolPoint1','patrolPoint2','patrolPoint3','patrolDwell','patrolFacePause'].forEach(id=>$(id).addEventListener('change',()=>{if(current)patrolView(current,pending)}));
  $('targetChoice').onchange=()=>{if($('targetChoice').value!=='')selectTarget(+$('targetChoice').value,choiceFrame)};
  $('overlay').onclick=event=>{
    const view=visibleFrame();if(!view)return;
    const r=$('overlay').getBoundingClientRect(),scale=Math.min(r.width/view.width,r.height/view.height);
    const x=(event.clientX-r.left-(r.width-view.width*scale)/2)/scale,y=(event.clientY-r.top-(r.height-view.height*scale)/2)/scale;
    const hits=view.faces.map((box,index)=>({box,index})).filter(({box:[x1,y1,x2,y2]})=>x>=x1&&x<=x2&&y>=y1&&y<=y2);
    if(hits.length===1)selectTarget(hits[0].index);
  };
  $('auto').onclick=()=>command('auto',{panSign:$('reversePan').checked?-1:1,tiltSign:$('reverseTilt').checked?-1:1});
  $('rock').onclick=()=>command('rock',{speed:+$('rockSpeed').value,duration:+$('rockDuration').value});
  $('armSoothe').onclick=()=>command(current?.mode==='observe'||(current?.mode==='rock'&&current.motor.source==='vision')?'stop':'arm_soothe');
  for(const [id,dp,dt] of [['left',-5,0],['right',5,0],['up',0,5],['down',0,-5]]){
    $(id).onclick=()=>command('manual',{pan:Math.max(+$('pan').min,Math.min(+$('pan').max,+$('pan').value+dp)),tilt:Math.max(+$('tilt').min,Math.min(+$('tilt').max,+$('tilt').value+dt))});
  }
  async function poll(){
    const revision=commandRevision, startedWhilePending=pending;
    const isCurrent=()=>!startedWhilePending&&!pending&&revision===commandRevision;
    try{const receivedAt=performance.now();const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(700)});if(!response.ok)throw Error();const state=await response.json();if(isCurrent())paint(state,receivedAt);
      if((current.mode!=='idle'||current.voiceControl?.armed)&&!document.hidden&&!pending)await command('heartbeat');
    }catch{if(isCurrent()){online=false;controls.forEach(id=>$(id).disabled=true);$('analysisRotation').disabled=true;$('connectCamera').disabled=$('disconnectCamera').disabled=true;$('connection').textContent=$('mobileMode').textContent='控制连接中断';$('message').textContent='停止发送目标，主控通信看门狗将停动';$('camera').hidden=true;$('empty').hidden=false;deviceFeedback({hardware:current?.hardware,controlState:'offline',device:{connected:false}});$('careAlerts').querySelectorAll('button').forEach(button=>button.disabled=true);careRevision='';$('followupState').textContent='控制页断联，观察状态待核实';trackingView(current||{});}}
    if(!online)recognitionView();
    setTimeout(poll,150);
  }
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&(current?.mode!=='idle'||current?.voiceControl?.armed))command('stop')});
  window.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();command('stop')}});
  poll();
  videoLoop();
})();
