/* Local acknowledgement and optional sound; no actuator-control endpoint. */
(() => {
  const el = id => document.getElementById(id);
  let inbox = {items: [], pending: 0, dropped: 0}, token = '', online = false;
  let signature = '', audio = null, soundReady = false, lastBeep = -Infinity;
  let deliveredAt = -Infinity;
  const busy = new Set();
  const sounded = new Set();
  function error(text) {
    el('reminderError').textContent = text;
    el('reminderError').hidden = !text;
  }
  function draw() {
    el('reminderCount').textContent = inbox.pending;
    el('reminderDropped').textContent = inbox.dropped ? ` 仅保留最近100条，已移出 ${inbox.dropped} 条。` : '';
    const visible = inbox.items.filter(item => el('reminderHistory').checked || !item.acknowledged).slice().reverse();
    const next = JSON.stringify([online, visible.map(item => [item.id, item.title, item.detail,
      item.source, item.firstUtc, item.condition, item.acknowledged, busy.has(item.id)])]);
    if (signature === next) return;
    signature = next;
    const rows = visible.map(item => {
      const row = document.createElement('li'), body = document.createElement('div');
      row.dataset.id = item.id;
      row.dataset.condition = item.condition;
      const title = document.createElement('strong'), tag = document.createElement('span');
      title.textContent = item.title;
      tag.className = 'reminder-tag';
      tag.textContent = {active: '仍在报告', unverified: '当前待核实', recovered: '已不再报告'}[item.condition] || '待核实';
      const detail = document.createElement('p'), meta = document.createElement('small');
      detail.textContent = item.detail;
      const date = new Date(item.firstUtc);
      meta.textContent = `${item.source} · ${Number.isNaN(date.getTime()) ? item.firstUtc : date.toLocaleTimeString('zh-CN')}`;
      title.append(tag);
      body.append(title, detail, meta);
      const ack = document.createElement('button');
      ack.type = 'button';
      ack.textContent = item.acknowledged ? '已查看' : busy.has(item.id) ? '正在确认' : '标记已查看';
      ack.setAttribute('aria-label', `${item.acknowledged ? '已查看' : '标记已查看'}：${item.title}`);
      ack.disabled = item.acknowledged || busy.has(item.id) || !online;
      ack.addEventListener('click', () => acknowledge(item.id));
      row.append(body, ack);
      return row;
    });
    if (!rows.length) {
      const empty = document.createElement('li');
      empty.className = 'reminder-empty';
      empty.textContent = inbox.items.length ? '本场提醒均已查看' : '暂无提醒';
      rows.push(empty);
    }
    el('reminderList').replaceChildren(...rows);
  }
  async function acknowledge(id) {
    if (!online || busy.has(id)) return;
    const requestToken = token;
    busy.add(id); draw(); error('');
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch('/api/alerts/ack', {
        method: 'POST', headers: {'Content-Type': 'application/json', 'X-Alert-Token': requestToken},
        body: JSON.stringify({id}), signal: controller.signal,
      });
      if (!response.ok) throw new Error('acknowledgement failed');
      const result = await response.json();
      if (result.acknowledged !== true || result.deviceCommandSent !== false) throw new Error('invalid response');
      const item = requestToken === token ? inbox.items.find(item => item.id === id) : null;
      if (item && !item.acknowledged) { item.acknowledged = true; inbox.pending = Math.max(0, inbox.pending - 1); }
    } catch (_) {
      error('未能确认提醒，请检查看板连接后重试。');
    } finally {
      clearTimeout(timeout); busy.delete(id); draw();
    }
  }
  function beep() {
    if (!online || performance.now() - deliveredAt > 3500 || !soundReady || audio?.state !== 'running') return;
    const pending = inbox.items.filter(item => !item.acknowledged && item.condition === 'active');
    const fresh = pending.some(item => !sounded.has(item.id));
    if (!pending.length || performance.now() - lastBeep < (fresh ? 1500 : 30000)) return;
    const oscillator = audio.createOscillator(), gain = audio.createGain(), start = audio.currentTime;
    oscillator.frequency.value = 660;
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(.08, start + .03);
    gain.gain.linearRampToValueAtTime(0, start + .4);
    oscillator.connect(gain); gain.connect(audio.destination);
    oscillator.start(start); oscillator.stop(start + .42);
    oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
    lastBeep = performance.now();
    pending.forEach(item => sounded.add(item.id));
    if (sounded.size > 200) {
      const retained = new Set(inbox.items.map(item => item.id));
      for (const id of sounded) if (!retained.has(id)) sounded.delete(id);
    }
  }
  el('reminderSound').addEventListener('change', async () => {
    soundReady = false;
    if (!el('reminderSound').checked) { if (audio) await audio.suspend(); return; }
    try {
      audio ||= new (window.AudioContext || window.webkitAudioContext)();
      await audio.resume();
      if (!el('reminderSound').checked) { await audio.suspend(); return; }
      soundReady = audio.state === 'running';
      if (!soundReady) throw new Error('audio suspended');
      error(''); beep();
    } catch (_) {
      el('reminderSound').checked = false;
      error('提示音未启用；请允许浏览器播放声音后重新勾选。');
    }
  });
  el('reminderHistory').addEventListener('change', draw);
  setInterval(beep, 1000);
  window.ReminderUI = {
    render(snapshot) {
      online = Boolean(snapshot.alertInbox && snapshot.ackToken);
      el('reminderConnection').hidden = online;
      el('reminderConnection').textContent = '提醒连接不可用，保留此前待处理记录。';
      if (online) {
        if (token !== snapshot.ackToken) { sounded.clear(); lastBeep = -Infinity; signature = ''; }
        inbox = snapshot.alertInbox; token = snapshot.ackToken; deliveredAt = performance.now();
      }
      draw(); beep();
    },
  };
})();
