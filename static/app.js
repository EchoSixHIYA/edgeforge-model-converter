const form = document.querySelector('#convertForm');
const vendor = document.querySelector('#vendor');
const arch = document.querySelector('#architecture');
const platform = document.querySelector('#platform');
const precision = document.querySelector('#precision');
const modelFile = document.querySelector('#modelFile');
const calibration = document.querySelector('#calibrationFile');
const errorBox = document.querySelector('#formError');
const watchers = new Map();
const taskRows = new Map();
const cancelRequests = new Set();
const terminalStatuses = ['completed', 'failed', 'canceled'];

function syncOptions() {
  const selectedVendor = vendor.value;
  [...arch.options].forEach(item => {
    const itemSupport = JSON.parse(item.dataset.vendorSupport || '{}');
    const support = itemSupport[selectedVendor];
    // HF directories have their own workflow and never use the CV upload route.
    item.hidden = !support?.available;
    if (item.dataset.label && support?.available) {
      item.textContent = item.dataset.label;
    }
  });
  [...arch.querySelectorAll('optgroup')].forEach(group => {
    group.hidden = ![...group.children].some(item => !item.hidden && !item.disabled);
  });
  if (arch.selectedOptions[0]?.hidden || !arch.selectedOptions.length) {
    const next = [...arch.options].find(item => !item.hidden && !item.disabled);
    arch.value = next?.value || '';
  }
  const option = arch.selectedOptions[0];
  document.querySelector('#submitButton').disabled = !option;
  if (!option) return;
  const vendorSupport = JSON.parse(option.dataset.vendorSupport || '{}');
  const support = vendorSupport[selectedVendor] || {};
  const allowed = support.platforms || JSON.parse(option.dataset.platforms || '[]');
  const formats = support.weight_formats || JSON.parse(option.dataset.formats || '[]');
  const size = support.input_size || JSON.parse(option.dataset.size || '[640,640]');
  const precisions = support.precisions || JSON.parse(option.dataset.precisions || '["fp"]');
  [...platform.options].forEach(item => item.hidden = !allowed.includes(item.value));
  if (!allowed.includes(platform.value)) platform.value = allowed[0];
  [...precision.options].forEach(item => item.hidden = !precisions.includes(item.value));
  if (!precisions.includes(precision.value)) precision.value = precisions[0];
  modelFile.accept = formats.join(',');
  document.querySelector('#sourceHint').textContent =
    `接受原始权重 ${formats.join(' / ').toUpperCase()}，友好 ONNX 由架构专用导出器在服务内部生成`;
  document.querySelector('#inputShape').value = `1, 3, ${size[1] || size[0]}, ${size[0]}`;
  document.querySelector('#calibrationBox').classList.toggle('hidden', !['i8', 'u8'].includes(precision.value));
}

vendor.addEventListener('change', syncOptions);
arch.addEventListener('change', syncOptions);
precision.addEventListener('change', syncOptions);
syncOptions();

function showModelName() {
  document.querySelector('#modelLabel').textContent = modelFile.files[0]?.name || '拖入原始模型权重';
}
modelFile.addEventListener('change', showModelName);
calibration.addEventListener('change', () => {
  document.querySelector('#calibrationName').textContent = calibration.files[0]?.name || '';
});
const drop = document.querySelector('#modelDrop');
['dragenter', 'dragover'].forEach(name => drop.addEventListener(name, event => {
  event.preventDefault();
  drop.classList.add('drag');
}));
['dragleave', 'dragend'].forEach(name => drop.addEventListener(name, () => drop.classList.remove('drag')));
drop.addEventListener('drop', event => {
  event.preventDefault();
  drop.classList.remove('drag');
  if (event.dataTransfer.files.length) {
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    modelFile.files = transfer.files;
    showModelName();
  }
});

async function readResponse(response) {
  let data;
  try { data = await response.json(); }
  catch (_) { throw new Error(`服务返回了非 JSON 响应（HTTP ${response.status}），请检查服务日志。`); }
  if (!response.ok) {
    const error = new Error(data.error || `请求失败（HTTP ${response.status}）`);
    error.details = data;
    throw error;
  }
  return data;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = document.querySelector('#submitButton');
  if (button.disabled) return;
  errorBox.textContent = '';
  if (['i8', 'u8'].includes(precision.value) && !calibration.files.length) {
    errorBox.textContent = 'INT8 / UINT8 转换必须上传量化校准图片 ZIP。';
    return;
  }
  button.disabled = true;
  button.querySelector('span').textContent = '正在上传权重...';
  try {
    const data = await readResponse(await fetch('/api/tasks', {method: 'POST', body: new FormData(form)}));
    renderTask(data, true);
    if (!terminalStatuses.includes(data.status)) watch(data.id);
    form.reset();
    syncOptions();
    showModelName();
    document.querySelector('#calibrationName').textContent = '';
    document.querySelector('#jobs').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { errorBox.textContent = error.message; }
  finally {
    button.disabled = !arch.value;
    button.querySelector('span').textContent = '开始转换';
  }
});

const vlmForm = document.querySelector('#vlmForm');
const vlmSubmit = document.querySelector('#vlmSubmit');
const vlmCheck = document.querySelector('#vlmPreflight');
const vlmError = document.querySelector('#vlmError');
const vlmResult = document.querySelector('#vlmPreflightResult');
let vlmCheckVersion = 0;
let vlmController = null;
let vlmCheckedFingerprint = null;
let vlmSubmitting = false;

function setWorkflow(workflow) {
  const isVlm = workflow === 'vlm';
  form.classList.toggle('hidden', isVlm);
  vlmForm.classList.toggle('hidden', !isVlm);
  document.querySelectorAll('[data-workflow]').forEach(tab => {
    const active = tab.dataset.workflow === workflow;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
}
document.querySelectorAll('[data-workflow]').forEach(tab => {
  tab.addEventListener('click', () => setWorkflow(tab.dataset.workflow));
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const workflow = event.key === 'Home' ? 'cv' : event.key === 'End' ? 'vlm' : tab.dataset.workflow === 'cv' ? 'vlm' : 'cv';
    setWorkflow(workflow);
    document.querySelector(`[data-workflow="${workflow}"]`).focus();
  });
});
setWorkflow('cv');

function vlmParameters() {
  return {
    architecture: document.querySelector('#vlmArchitecture').value,
    model_dir: document.querySelector('#vlmModelDir').value.trim(),
    platform: 'bm1684x',
    precision: document.querySelector('#vlmPrecision').value,
    seq_length: Number(document.querySelector('#vlmSeqLength').value),
    max_pixels: Number(document.querySelector('#vlmMaxPixels').value),
    jobs: Number(document.querySelector('#vlmJobs').value),
  };
}
function vlmFingerprint() { return JSON.stringify(vlmParameters()); }
function updateVlmSubmit() {
  const checked = vlmCheckedFingerprint !== null && vlmCheckedFingerprint === vlmFingerprint();
  vlmSubmit.disabled = vlmSubmitting || !checked;
  document.querySelector('#vlmLowMemoryPreset').disabled = vlmSubmitting;
  vlmSubmit.querySelector('span').textContent = vlmSubmitting ? '正在创建任务…'
    : !checked ? '请先完成预检' : '开始转换';
}
function invalidateVlmCheck() {
  vlmCheckVersion += 1;
  vlmController?.abort();
  vlmController = null;
  vlmCheckedFingerprint = null;
  vlmCheck.disabled = false;
  vlmCheck.textContent = '检查目录与编译环境';
  if (!vlmResult.classList.contains('hidden')) {
    vlmResult.className = 'preflight-result';
    vlmResult.textContent = '配置已修改，请重新预检。';
  }
  vlmError.textContent = '';
  const pixels = Number(document.querySelector('#vlmMaxPixels').value);
  document.querySelector('#vlmVisualTokens').value = Number.isFinite(pixels) && pixels > 0
    ? `${Math.floor(pixels / 1024)}（每图 / 每时序组）` : '请填写有效像素预算';
  updateVlmSubmit();
}
function applyVlmLowMemoryPreset() {
  if (vlmSubmitting) return;
  document.querySelector('#vlmPrecision').value = 'w4bf16';
  document.querySelector('#vlmSeqLength').value = '1024';
  document.querySelector('#vlmMaxPixels').value = '65536';
  document.querySelector('#vlmJobs').value = '1';
  invalidateVlmCheck();
  document.querySelector('#vlmPresetHint').textContent = '已应用低内存预设；旧预检已失效，请重新检查。';
}
vlmForm.querySelectorAll('input:not([readonly]):not([type="checkbox"]), select').forEach(input => {
  input.addEventListener('input', invalidateVlmCheck);
  input.addEventListener('change', invalidateVlmCheck);
});
document.querySelector('#vlmLowMemoryPreset').addEventListener('click', applyVlmLowMemoryPreset);

function escapeHtml(value) {
  const element = document.createElement('span');
  element.textContent = String(value ?? '');
  return element.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function formatBytes(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '未知';
  return `${(Number(value) / (1024 ** 3)).toFixed(digits)} GiB`;
}
function renderDeviceMemoryEstimate(estimate) {
  const heading = '<h4>板端内存估算</h4>';
  if (!estimate || typeof estimate !== 'object') {
    return `<section class="device-memory-note">${heading}<p>暂无估算数据。</p></section>`;
  }
  const scenarios = Array.isArray(estimate.scenarios) ? estimate.scenarios : [];
  const rows = scenarios.map(scenario => `<div><dt>${escapeHtml(scenario.label || scenario.id)} · 理论权重</dt><dd>${escapeHtml(formatBytes(scenario.weight_bytes, 2))}</dd></div>
    <div><dt>${escapeHtml(scenario.label || scenario.id)} · 预计内存</dt><dd>${escapeHtml(formatBytes(scenario.known_component_sum_bytes, 2))}</dd></div>`).join('');
  const measuredLabels = {
    kv_cache_bytes: 'KV 缓存计算量', explicit_device_buffers_bytes: '显式设备 buffer 计算量',
    peak_memory_bytes: '真实峰值内存', tpu_heap_available_bytes: 'TPU 可分配 heap',
    board_system_total_bytes: '板端 Linux 可见总内存', board_system_available_bytes: '板端 Linux MemAvailable',
  };
  const measuredRows = Object.entries(measuredLabels).map(([key, label]) => `<div><dt>${label}</dt><dd>${escapeHtml(formatBytes(estimate[key], 2))}</dd></div>`).join('');
  return `<section class="device-memory-note">${heading}
    <dl class="resource-summary">${rows}${measuredRows}</dl>
    </section>`;
}
function renderPreflight(data) {
  vlmResult.className = `preflight-result ${data.ready === true ? 'check-ready' : 'check-blocked'}`;
  const errors = Array.isArray(data.errors) ? data.errors : [];
  const resources = data.resources || {};
  const resourceLabels = {
    weight_bytes: '源权重大小', available_memory_bytes: '可用主机内存', minimum_memory_bytes: '最低预检内存阈值（非峰值）',
    disk_free_bytes: '可用工作磁盘', estimated_work_disk_bytes: '工作磁盘估算（启发式）',
  };
  const resourceRows = Object.entries(resourceLabels).filter(([key]) => key in resources)
    .map(([key, label]) => `<div><dt>${label}</dt><dd>${escapeHtml(formatBytes(resources[key]))}</dd></div>`).join('');
  const {warnings: _warnings, assumptions: _assumptions, ...memoryDetails} = data.device_memory_estimate || {};
  const metadata = {model: data.model || {}, parameters: data.parameters || {}, toolchain: data.toolchain || {}, device_memory_estimate: memoryDetails};
  vlmResult.innerHTML = `<h3>${data.ready === true ? '预检通过' : '请先处理以下问题'}</h3>
    ${errors.length ? `<ul class="check-errors">${errors.map(error => `<li>${escapeHtml(error)}</li>`).join('')}</ul>` : ''}
    ${resourceRows ? `<h4 class="resource-section-title">转换电脑资源</h4><dl class="resource-summary">${resourceRows}</dl>` : ''}
    ${renderDeviceMemoryEstimate(data.device_memory_estimate)}
    <details class="check-metadata"><summary>模型、参数与工具链检测详情</summary><pre>${escapeHtml(JSON.stringify(metadata, null, 2))}</pre></details>`;
}
async function requestVlm(url, parameters, signal) {
  return readResponse(await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-EdgeForge-Local-Import': '1'},
    body: JSON.stringify({...parameters, acknowledge_experimental: true}),
    signal,
  }));
}
vlmCheck.addEventListener('click', async () => {
  if (vlmCheck.disabled || !vlmForm.reportValidity()) return;
  vlmError.textContent = '';
  vlmCheckedFingerprint = null;
  updateVlmSubmit();
  const parameters = vlmParameters();
  const fingerprint = JSON.stringify(parameters);
  const version = ++vlmCheckVersion;
  vlmController?.abort();
  vlmController = new AbortController();
  vlmCheck.disabled = true;
  vlmCheck.textContent = '正在检查…';
  try {
    const data = await requestVlm('/api/vlm/preflight', parameters, vlmController.signal);
    if (version !== vlmCheckVersion || fingerprint !== vlmFingerprint()) return;
    renderPreflight(data);
    if (data.ready === true) vlmCheckedFingerprint = fingerprint;
  } catch (error) {
    if (version !== vlmCheckVersion || error.name === 'AbortError') return;
    vlmError.textContent = error.message;
    vlmResult.className = 'preflight-result check-blocked';
    vlmResult.textContent = '预检未完成。请确认本机访问、模型根目录配置与服务状态后重试。';
  } finally {
    if (version === vlmCheckVersion) {
      vlmController = null;
      vlmCheck.disabled = false;
      vlmCheck.textContent = '重新检查目录与编译环境';
      updateVlmSubmit();
    }
  }
});
vlmForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (vlmSubmit.disabled || !vlmForm.reportValidity()) return;
  const parameters = vlmParameters();
  if (JSON.stringify(parameters) !== vlmCheckedFingerprint) return;
  vlmSubmitting = true;
  vlmError.textContent = '';
  updateVlmSubmit();
  try {
    const task = await requestVlm('/api/vlm/tasks', parameters);
    renderTask(task, true);
    if (!terminalStatuses.includes(task.status)) watch(task.id);
    vlmCheckedFingerprint = null;
    vlmResult.className = 'preflight-result';
    vlmResult.textContent = '任务已创建。再次提交前请重新预检；请在下方查看进度、日志与产物。';
    document.querySelector('#jobs').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    vlmError.textContent = error.message;
    if (Array.isArray(error.details?.errors)) renderPreflight(error.details);
    vlmCheckedFingerprint = null;
  } finally {
    vlmSubmitting = false;
    updateVlmSubmit();
  }
});

const statusLabels = {queued: '排队中', running: '转换中', canceling: '正在停止', canceled: '已取消', completed: '已完成', failed: '失败'};
function formatTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {hour12: false});
}
function isVlmTask(task) { return task.input_kind === 'hf_directory' || task.architecture === 'cosmos_reason2_2b'; }
function renderTask(task, prepend = false) {
  document.querySelector('#empty').style.display = 'none';
  let row = taskRows.get(task.id);
  if (!row) {
    row = document.createElement('article');
    row.className = 'job';
    row.dataset.job = task.id;
    taskRows.set(task.id, row);
    prepend ? document.querySelector('#jobList').prepend(row) : document.querySelector('#jobList').append(row);
  }
  const expanded = row.querySelector('details')?.open;
  const done = task.status === 'completed';
  const vlm = isVlmTask(task);
  const progress = Math.max(0, Math.min(100, Number(task.progress) || 0));
  const url = `/api/tasks/${encodeURIComponent(task.id)}`;
  const status = statusLabels[task.status] ? task.status : 'unknown';
  const log = typeof task.log_tail === 'string' ? task.log_tail : '';
  const message = vlm && done ? 'BModel 与 SDK 已生成' : task.message || '';
  row.innerHTML = `
    <div class="job-model">
      <h3 title="${escapeHtml(task.original_filename)}">${escapeHtml(task.original_filename || task.architecture)}</h3>
      <small>#${escapeHtml(String(task.id).slice(0, 8).toUpperCase())} · ${escapeHtml(String(task.architecture || '').toUpperCase())}${vlm ? ' · HF VLM' : ''}</small>
    </div>
    <div class="job-platform">${escapeHtml(String(task.platform || '').toUpperCase())}<small>${escapeHtml((task.vendor || 'rockchip').toUpperCase())} · ${escapeHtml(String(task.precision || '').toUpperCase())}</small></div>
    <div class="job-state">
      <div class="state-line"><span class="status ${status}">${escapeHtml(statusLabels[task.status] || task.status)}</span><span class="percent">${progress}%</span></div>
      <div class="progress-track"><div class="progress-bar" style="width:${progress}%"></div></div>
      <div class="job-message" title="${escapeHtml(message)}">${escapeHtml(message)}</div>
    </div>
    <time class="job-time">${escapeHtml(formatTime(task.created_at))}</time>
    <div class="downloads">${done ? `<a href="${url}/download/model">模型</a><a class="sdk" href="${url}/download/package">SDK 包</a>` : ''}${vlm && !terminalStatuses.includes(task.status) ? `<button type="button" class="cancel-task" data-cancel="${escapeHtml(task.id)}" ${task.status === 'canceling' || cancelRequests.has(task.id) ? 'disabled' : ''}>${task.status === 'canceling' ? '正在停止' : '取消'}</button>` : ''}</div>
    ${vlm || task.status === 'failed' ? `<details class="task-details" ${expanded ? 'open' : ''}>
      <summary>${task.status === 'failed' ? '查看失败原因' : '任务参数与日志'}</summary>
      <p class="task-message">${escapeHtml(message)}</p>
      ${vlm ? `<p class="task-budget">${escapeHtml(JSON.stringify(task.parameters || {}, null, 2))}</p>${log ? `<a class="log-download" href="${url}/download/log">下载完整转换日志</a>` : '<span class="log-pending">日志尚未生成</span>'}<pre class="task-log">${escapeHtml(log || '暂未产生日志；任务运行后会实时更新。')}</pre>` : ''}
    </details>` : ''}`;
}
function watch(id) {
  if (watchers.has(id)) return;
  const stream = new EventSource(`/api/tasks/${encodeURIComponent(id)}/events`);
  watchers.set(id, stream);
  stream.onmessage = event => {
    try {
      const task = JSON.parse(event.data);
      renderTask(task);
      if (terminalStatuses.includes(task.status)) {
        stream.close();
        watchers.delete(id);
      }
    } catch (_) { /* Keep the last valid task; a later event or refresh can recover. */ }
  };
  // EventSource reconnects automatically after a temporary disconnect.
  stream.onerror = () => {
    const notice = document.querySelector('#tasksError');
    notice.classList.remove('hidden');
    notice.textContent = '实时连接暂时中断，正在自动重连；也可刷新任务状态。';
  };
  stream.onopen = () => document.querySelector('#tasksError').classList.add('hidden');
}
let tasksLoadVersion = 0;
async function loadTasks() {
  const version = ++tasksLoadVersion;
  const button = document.querySelector('#refresh');
  button.disabled = true;
  try {
    const tasks = await readResponse(await fetch('/api/tasks'));
    if (version !== tasksLoadVersion) return;
    if (!Array.isArray(tasks)) throw new Error('任务列表响应格式错误');
    document.querySelector('#tasksError').classList.add('hidden');
    document.querySelector('#jobList').innerHTML = '';
    taskRows.clear();
    document.querySelector('#empty').style.display = tasks.length ? 'none' : 'flex';
    tasks.forEach(task => {
      renderTask(task);
      if (!terminalStatuses.includes(task.status)) watch(task.id);
      else if (watchers.has(task.id)) {
        watchers.get(task.id).close();
        watchers.delete(task.id);
      }
    });
  } catch (error) {
    const notice = document.querySelector('#tasksError');
    notice.classList.remove('hidden');
    notice.textContent = `读取任务失败：${error.message}`;
  } finally { if (version === tasksLoadVersion) button.disabled = false; }
}
document.querySelector('#jobList').addEventListener('click', async event => {
  const button = event.target.closest('[data-cancel]');
  if (!button || button.disabled) return;
  const id = button.dataset.cancel;
  if (!window.confirm('取消这个转换任务？已产生的日志会保留。')) return;
  cancelRequests.add(id);
  button.disabled = true;
  try {
    const task = await readResponse(await fetch(`/api/tasks/${encodeURIComponent(id)}/cancel`, {method: 'POST', headers: {'X-EdgeForge-Local-Import': '1'}}));
    renderTask(task);
    if (!terminalStatuses.includes(task.status)) watch(task.id);
    else if (watchers.has(task.id)) {
      watchers.get(task.id).close();
      watchers.delete(task.id);
    }
  } catch (error) {
    const notice = document.querySelector('#tasksError');
    notice.classList.remove('hidden');
    notice.textContent = `取消失败：${error.message}`;
    button.disabled = false;
  } finally { cancelRequests.delete(id); }
});
document.querySelector('#refresh').addEventListener('click', loadTasks);

fetch('/health').then(readResponse).then(health => {
  const element = document.querySelector('#health');
  const rockchipReady = Boolean(health.rockchip?.ready ?? (health.rknn_toolkit2 && health.model_zoo));
  const sophonReady = Boolean(health.sophon?.ready);
  if (rockchipReady && sophonReady) {
    element.classList.add('ready');
    element.querySelector('span').textContent = 'Rockchip / SOPHGO 环境就绪';
  } else if (rockchipReady) {
    element.classList.add('ready');
    element.querySelector('span').textContent = 'Rockchip 环境就绪';
  } else if (sophonReady) {
    element.classList.add('ready');
    element.querySelector('span').textContent = 'SOPHGO 环境就绪';
  } else {
    element.classList.add('warn');
    element.querySelector('span').textContent = '转换环境组件待安装';
  }
  const vlm = health.sophon?.vlm;
  const readiness = document.querySelector('#vlmReadiness');
  readiness.textContent = !vlm ? 'HF VLM 接口不可用：请更新并启动支持此流程的服务。'
    : !vlm.enabled || vlm.local_import?.enabled === false ? '本机目录导入未启用：请先配置 EDGEFORGE_MODEL_ROOTS。'
    : !vlm.ready ? 'HF VLM 编译环境未就绪，请通过预检查看缺少的组件。'
    : 'HF VLM 工具链就绪';
}).catch(error => {
  document.querySelector('#health span').textContent = '环境状态暂不可用';
  document.querySelector('#vlmReadiness').textContent = `无法检查 VLM 环境：${error.message}`;
});
loadTasks();
