/* All rules run locally; applying them cannot enable an actuator. */
(() => {
  const el = id => document.getElementById(id);
  const names = {temperature: '环境温度 (°C)', humidity: '环境湿度 (%RH)', no_target: '持续无目标'};
  const states = {disabled: '未启用', waiting: '数据不足', unverified: '异常待核实', normal: '未触发',
    confirming: '持续确认中', active: '已触发提醒', recovering: '恢复确认中'};
  let current = null, token = '', online = false, editing = false, saving = false, formRevision = null, formToken = '';
  function fields(config) {
    el('rulesFields').replaceChildren(...Object.entries(config).map(([name, rule]) => {
      const row = document.createElement('fieldset');
      const legend = document.createElement('legend'); legend.textContent = names[name]; row.append(legend);
      for (const [key, value] of Object.entries(rule)) {
        const label = document.createElement('label'), input = document.createElement('input');
        input.id = `rule-${name}-${key}`; input.name = `${name}.${key}`;
        input.type = key === 'enabled' ? 'checkbox' : 'number';
        if (key === 'enabled') input.checked = value;
        else {
          input.value = value; input.required = true;
          input.step = key === 'holdSeconds' ? '1' : '0.1';
          input.min = key === 'holdSeconds' ? '5' : key === 'hysteresis' ? '0.1' : '0';
          input.max = key === 'holdSeconds' ? '600' : name === 'temperature' ? '50' : '100';
        }
        label.append(document.createTextNode({enabled:'启用',low:'下限',high:'上限',hysteresis:'恢复回差',holdSeconds:'持续秒数'}[key]), input);
        row.append(label);
      }
      const status = document.createElement('output'); status.id = `rule-${name}-status`; row.append(status);
      return row;
    }));
  }
  function render(snapshot) {
    online = Boolean(snapshot.monitorRules && snapshot.ackToken);
    if (online) {
      current = snapshot.monitorRules; token = snapshot.ackToken;
      if (!editing && !saving && (formRevision !== current.revision || formToken !== token)) {
        fields(current.config); formRevision = current.revision; formToken = token;
      }
      let active = 0;
      for (const result of current.results) {
        if (current.config[result.name].enabled) active++;
        const field = el(`rule-${result.name}-status`);
        if (field) field.textContent = (states[result.state] || '数据不足') +
          (result.state === 'confirming' ? ` ${Math.floor(result.elapsedMs / 1000)} / ${result.holdMs / 1000}s` : '');
      }
      el('rulesSummary').textContent = `已启用 ${active} 项`;
    } else {
      el('rulesSummary').textContent = '规则连接不可用';
      for (const name of Object.keys(names)) {
        const field = el(`rule-${name}-status`); if (field) field.textContent = '连接不可用';
      }
    }
    el('rulesSave').disabled = !online || saving;
  }
  el('rulesForm').addEventListener('input', () => { editing = true; });
  el('rulesForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (!online || saving || !current) return;
    const config = {};
    for (const [name, rule] of Object.entries(current.config)) {
      config[name] = {};
      for (const key of Object.keys(rule)) {
        const input = el(`rule-${name}-${key}`);
        config[name][key] = key === 'enabled' ? input.checked : Number(input.value);
      }
    }
    saving = true; el('rulesSave').disabled = true;
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch('/api/rules', {method: 'POST', signal: controller.signal,
        headers: {'Content-Type': 'application/json', 'X-Alert-Token': token},
        body: JSON.stringify({revision: formRevision, config})});
      if (response.status === 409) throw new Error('其他页面已修改规则，请刷新后重设。');
      if (!response.ok) throw new Error('设置未生效：请检查上下限、回差和持续时间。');
      const result = await response.json();
      if (result.configured !== true || result.deviceCommandSent !== false) throw new Error('响应异常，尚未确认生效。');
      editing = false; el('rulesMessage').textContent = '规则已应用，仅影响本地提醒。';
    } catch (error) { el('rulesMessage').textContent = error.message || '规则提交失败'; }
    finally { clearTimeout(timer); saving = false; el('rulesSave').disabled = !online; }
  });
  window.MonitorRulesUI = {render};
})();
