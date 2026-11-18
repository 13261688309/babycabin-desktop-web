'use strict';
let latest = null;
let releasing = false;
const releaseButton = document.getElementById('release-controller');
function renderController(state) {
  const device = state.device || {};
  const controller = device.controller || {};
  const active = controller.fresh && ['lease','pwm','enabled','remaining','rock','motor']
    .some(key => Boolean((controller.data || {})[key]));
  document.getElementById('controller-state').textContent = device.readOnly
    ? device.connected ? '已连接 · 只读日志' : '未连接 · 串口已释放或连接中断'
    : state.hardware ? '硬件控制中，不可从这里释放' : '未接入主控日志';
  releaseButton.disabled = releasing || state.hardware || !device.readOnly || !device.connected
    || state.mode !== 'idle' || state.controlState === 'stopping' || active;
}
async function refresh() {
  try {
    const [response, sessionResponse] = await Promise.all([
      fetch('/api/mobile-pairing', {cache: 'no-store'}), fetch('/api/state', {cache: 'no-store'})
    ]);
    if (!response.ok || !sessionResponse.ok) throw new Error('手机网关未启用');
    const state = await response.json();
    latest = await sessionResponse.json();
    document.getElementById('host').textContent = new URL(state.url).hostname;
    document.getElementById('code').textContent = state.code || (state.paired ? '已配对' : '已过期');
    document.getElementById('mode').textContent = state.hardware ? '硬件控制网关' : '软件预演，无硬件输出';
    document.getElementById('status').textContent = state.paired ? '手机已配对；可在 App 中手动连接。' : state.code ? '剩余 ' + Math.ceil(state.remainingMs / 1000) + ' 秒' : '请重新启动手机网关，生成新的配对码。';
    renderController(latest);
  } catch (error) {
    latest = null;
    releaseButton.disabled = true;
    document.getElementById('code').textContent = '不可用';
    document.getElementById('status').textContent = error.message;
    document.getElementById('controller-state').textContent = '网关不可用';
  }
}
releaseButton.addEventListener('click', async () => {
  if (!latest || releaseButton.disabled) return;
  releasing = true;
  releaseButton.disabled = true;
  try {
    const response = await fetch('/api/controller', {
      method: 'POST', headers: {'Content-Type':'application/json','X-Control-Token':latest.token},
      body: JSON.stringify({action:'disconnect',generation:latest.generation})
    });
    if (!response.ok) throw new Error(await response.text());
    latest = await response.json();
    document.getElementById('controller-result').textContent = '串口已释放，手机和相机连接保留';
  } catch (error) {
    document.getElementById('controller-result').textContent = error.message;
  } finally {
    releasing = false;
    await refresh();
  }
});
async function poll() { await refresh(); setTimeout(poll, 1000); }
poll();
