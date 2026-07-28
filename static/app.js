const form = document.querySelector('#convertForm');
const vendor = document.querySelector('#vendor');
const arch = document.querySelector('#architecture');
const platform = document.querySelector('#platform');
const precision = document.querySelector('#precision');
const modelFile = document.querySelector('#modelFile');
const calibration = document.querySelector('#calibrationFile');
const errorBox = document.querySelector('#formError');
const watchers = new Map();

function syncOptions() {
  const option = arch.selectedOptions[0];
  const vendorSupport = JSON.parse(option.dataset.vendorSupport || '{}');
  const selectedVendor = vendor.value;
  const support = vendorSupport[selectedVendor] || {};
  const allowed = support.platforms || JSON.parse(option.dataset.platforms || '[]');
  const formats = JSON.parse(option.dataset.formats || '[]');
  const size = JSON.parse(option.dataset.size || '[640,640]');
  const precisions = support.precisions || JSON.parse(option.dataset.precisions || '["fp"]');
  [...arch.options].forEach(item => {
    const itemSupport = JSON.parse(item.dataset.vendorSupport || '{}');
    item.hidden = !itemSupport[selectedVendor]?.available;
  });
  if (arch.selectedOptions[0]?.hidden) {
    const next = [...arch.options].find(item => !item.hidden && !item.disabled);
    if (next) arch.value = next.value;
    return syncOptions();
  }
  [...platform.options].forEach(item => item.hidden = !allowed.includes(item.value));
  if (!allowed.includes(platform.value)) platform.value = allowed[0];
  [...precision.options].forEach(item => item.hidden = !precisions.includes(item.value));
  if (!precisions.includes(precision.value)) precision.value = precisions[0];
  modelFile.accept = formats.join(',');
  document.querySelector('#sourceHint').textContent =
    `接受原始权重 ${formats.join(' / ').toUpperCase()}，友好 ONNX 由架构专用导出器在服务内部生成`;
  document.querySelector('#inputShape').value = `1, 3, ${size[1] || size[0]}, ${size[0]}`;
  document.querySelector('#calibrationBox').classList.toggle('hidden', precision.value === 'fp');
}

vendor.addEventListener('change', syncOptions);
arch.addEventListener('change', syncOptions);
precision.addEventListener('change', syncOptions);
syncOptions();

function showModelName() {
  const file = modelFile.files[0];
  document.querySelector('#modelLabel').textContent = file ? file.name : '拖入原始模型权重';
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

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorBox.textContent = '';
  if (precision.value !== 'fp' && !calibration.files.length) {
    errorBox.textContent = 'INT8 / UINT8 转换必须上传量化校准图片 ZIP。';
    return;
  }
  const button = document.querySelector('#submitButton');
  button.disabled = true;
  button.querySelector('span').textContent = '正在上传权重…';
  try {
    const response = await fetch('/api/tasks', {method: 'POST', body: new FormData(form)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '创建转换任务失败');
    renderTask(data, true);
    watch(data.id);
    form.reset();
    syncOptions();
    document.querySelector('#modelLabel').textContent = '拖入原始模型权重';
    document.querySelector('#calibrationName').textContent = '';
    document.querySelector('#jobs').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    errorBox.textContent = error.message;
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = '开始转换';
  }
});

const statusLabels = {queued: '排队中', running: '转换中', completed: '已完成', failed: '失败'};
function escapeHtml(value) {
  const element = document.createElement('span');
  element.textContent = String(value ?? '');
  return element.innerHTML;
}
function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {hour12: false});
}

function renderTask(task, prepend = false) {
  document.querySelector('#empty').style.display = 'none';
  let row = document.querySelector(`[data-job="${task.id}"]`);
  if (!row) {
    row = document.createElement('article');
    row.className = 'job';
    row.dataset.job = task.id;
    prepend ? document.querySelector('#jobList').prepend(row) : document.querySelector('#jobList').append(row);
  }
  const done = task.status === 'completed';
  const progress = Number(task.progress) || 0;
  row.innerHTML = `
    <div class="job-model">
      <h3 title="${escapeHtml(task.original_filename)}">${escapeHtml(task.original_filename)}</h3>
      <small>#${escapeHtml(task.id.slice(0, 8).toUpperCase())} · ${escapeHtml(task.architecture.toUpperCase())}</small>
    </div>
    <div class="job-platform">${escapeHtml(task.platform.toUpperCase())}<small>${escapeHtml((task.vendor || 'rockchip').toUpperCase())} · ${escapeHtml(task.precision.toUpperCase())}</small></div>
    <div class="job-state">
      <div class="state-line"><span class="status ${task.status}">${statusLabels[task.status] || task.status}</span><span class="percent">${progress}%</span></div>
      <div class="progress-track"><div class="progress-bar" style="width:${progress}%"></div></div>
      <div class="job-message" title="${escapeHtml(task.message)}">${escapeHtml(task.message)}</div>
    </div>
    <time class="job-time">${escapeHtml(formatTime(task.created_at))}</time>
    <div class="downloads">${done ? `<a href="/api/tasks/${task.id}/download/model">模型</a><a class="sdk" href="/api/tasks/${task.id}/download/package">SDK</a>` : ''}</div>`;
}

function watch(id) {
  if (watchers.has(id)) return;
  const stream = new EventSource(`/api/tasks/${id}/events`);
  watchers.set(id, stream);
  stream.onmessage = event => {
    const task = JSON.parse(event.data);
    renderTask(task);
    if (['completed', 'failed'].includes(task.status)) {
      stream.close();
      watchers.delete(id);
    }
  };
  stream.onerror = () => {
    stream.close();
    watchers.delete(id);
  };
}

async function loadTasks() {
  const response = await fetch('/api/tasks');
  const tasks = await response.json();
  document.querySelector('#jobList').innerHTML = '';
  document.querySelector('#empty').style.display = tasks.length ? 'none' : 'flex';
  tasks.forEach(task => {
    renderTask(task);
    if (!['completed', 'failed'].includes(task.status)) watch(task.id);
  });
}

document.querySelector('#refresh').addEventListener('click', loadTasks);
fetch('/health').then(response => response.json()).then(health => {
  const element = document.querySelector('#health');
  if (health.rknn_toolkit2 && health.model_zoo) {
    element.classList.add('ready');
    element.querySelector('span').textContent = '转换环境就绪';
  } else {
    element.classList.add('warn');
    element.querySelector('span').textContent = '环境组件待安装';
  }
}).catch(() => {});
loadTasks();
