/* Logic/DOM-stub unit tests. Not a substitute for a real browser render or conversion. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../static/app.js'), 'utf8');
function section(start, end) {
  const index = source.indexOf(start);
  assert.ok(index >= 0, `missing ${start}`);
  const endIndex = source.indexOf(end, index + start.length);
  assert.ok(endIndex > index, `missing ${end}`);
  return source.slice(index, endIndex);
}
function classList(initial = []) {
  const values = new Set(initial);
  return {contains: name => values.has(name), add: name => values.add(name), remove: name => values.delete(name)};
}
const nodes = {
  '#vlmArchitecture': {value: 'cosmos_reason2_2b'}, '#vlmModelDir': {value: ' /models/Cosmos '},
  '#vlmPrecision': {value: 'w4bf16'}, '#vlmSeqLength': {value: '2048'},
  '#vlmMaxPixels': {value: '147456'}, '#vlmJobs': {value: '1'}, '#vlmVisualTokens': {value: ''},
  '#empty': {style: {}}, '#jobList': {rows: [], append(row) {this.rows.push(row);}, prepend(row) {this.rows.unshift(row);}},
};
const label = {textContent: ''};
const document = {
  querySelector: selector => nodes[selector],
  createElement(tag) {
    if (tag === 'span') return {set textContent(value) {this.innerHTML = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');}};
    return {dataset: {}, innerHTML: '', querySelector: () => null};
  },
};
const context = vm.createContext({document, AbortController, Math, Number, String, JSON, Set, Map,
  vlmSubmit: {disabled: true, querySelector: () => label}, vlmSubmitting: false, vlmAck: {checked: false},
  vlmCheckedFingerprint: null, vlmCheckVersion: 1, vlmController: null,
  vlmCheck: {disabled: false, textContent: ''}, vlmError: {textContent: ''},
  vlmResult: {className: '', classList: classList(), innerHTML: '', textContent: ''},
  taskRows: new Map(), cancelRequests: new Set(), terminalStatuses: ['completed', 'failed', 'canceled'],
  statusLabels: {queued: '排队中', running: '转换中', canceling: '正在停止', canceled: '已取消', completed: '已完成', failed: '失败'},
});
for (const [start, end] of [
  ['function vlmParameters()', "vlmForm.querySelectorAll("],
  ['function escapeHtml(', 'async function requestVlm('],
  ['async function requestVlm(', "vlmCheck.addEventListener("],
  ['async function readResponse(', "form.addEventListener("],
  ['function formatTime(', 'function watch('],
]) vm.runInContext(section(start, end), context);

const malicious = 'x" onmouseover="alert(1)\' <img src=x onerror=alert(2)>';
assert.equal(context.escapeHtml(malicious), 'x&quot; onmouseover=&quot;alert(1)&#39; &lt;img src=x onerror=alert(2)&gt;');
assert.equal(context.formatBytes(null), '未知');
assert.equal(context.formatBytes(16 * 1024 ** 3), '16.0 GiB');
assert.deepEqual(JSON.parse(JSON.stringify(context.vlmParameters())), {
  architecture: 'cosmos_reason2_2b', model_dir: '/models/Cosmos', platform: 'bm1684x',
  precision: 'w4bf16', seq_length: 2048, max_pixels: 147456, jobs: 1,
});
context.updateVlmSubmit();
assert.equal(context.vlmSubmit.disabled, true);
context.vlmCheckedFingerprint = context.vlmFingerprint();
context.updateVlmSubmit();
assert.equal(context.vlmSubmit.disabled, true);
assert.equal(label.textContent, '请确认实验性说明');
context.vlmAck.checked = true;
context.updateVlmSubmit();
assert.equal(context.vlmSubmit.disabled, false);
context.vlmSubmitting = true;
context.updateVlmSubmit();
assert.equal(context.vlmSubmit.disabled, true);
context.vlmSubmitting = false;
let aborted = 0;
context.vlmController = {abort() {aborted += 1;}};
nodes['#vlmSeqLength'].value = '2176';
context.invalidateVlmCheck();
assert.equal(aborted, 1);
assert.equal(context.vlmCheckedFingerprint, null);
assert.equal(context.vlmSubmit.disabled, true);
assert.equal(context.vlmCheckVersion, 2);
assert.equal(nodes['#vlmVisualTokens'].value, '144（每图 / 每时序组）');

context.renderPreflight({ready: false, errors: [malicious], warnings: ['not board verified'],
  model: {name: malicious}, resources: {available_memory_bytes: 16 * 1024 ** 3}, parameters: context.vlmParameters()});
assert.match(context.vlmResult.innerHTML, /尚不可转换/);
assert.match(context.vlmResult.innerHTML, /16.0 GiB/);
assert.ok(!context.vlmResult.innerHTML.includes('<img'));
assert.ok(context.vlmResult.innerHTML.includes('&lt;img'));
const task = {id: 'example', input_kind: 'hf_directory', architecture: 'cosmos_reason2_2b',
  original_filename: malicious, platform: 'bm1684x', vendor: 'sophon', precision: 'w4bf16',
  created_at: '2026-10-22T00:00:00Z', status: 'failed', message: malicious, log_tail: '<script>alert(1)</script>', progress: -10};
context.renderTask(task);
let markup = context.taskRows.get('example').innerHTML;
assert.ok(markup.includes('title="x&quot; onmouseover=&quot;'));
assert.ok(!markup.includes('title="x" onmouseover='));
assert.ok(!markup.includes('<img'));
assert.ok(!markup.includes('<script>'));
assert.ok(markup.includes('&lt;script&gt;'));
assert.ok(markup.includes('width:0%'));
assert.ok(markup.includes('/download/log'));
assert.ok(!markup.includes('/download/package'));
context.renderTask({...task, status: 'completed', progress: 123});
markup = context.taskRows.get('example').innerHTML;
assert.ok(markup.includes('/download/package'));
assert.ok(markup.includes('width:100%'));
assert.ok(!markup.includes('data-cancel='));
context.renderTask({...task, status: 'running'});
assert.ok(context.taskRows.get('example').innerHTML.includes('data-cancel='));
context.renderTask({...task, status: 'canceled'});
assert.ok(!context.taskRows.get('example').innerHTML.includes('data-cancel='));
assert.equal(nodes['#jobList'].rows.length, 1);  // Updates replace, not duplicate, the row.

(async () => {
  let request;
  context.fetch = async (url, options) => {request = {url, ...options}; return {ok: true, json: async () => ({ready: true})};};
  await context.requestVlm('/api/vlm/preflight', context.vlmParameters());
  assert.equal(request.headers['X-EdgeForge-Local-Import'], '1');
  const body = JSON.parse(request.body);
  assert.equal(body.seq_length, 2176);
  assert.equal(body.max_pixels, 147456);
  assert.equal(body.jobs, 1);
  assert.equal(body.acknowledge_experimental, true);
  assert.equal(body.model_dir, '/models/Cosmos');
  assert.equal('max_frames' in body, false);
  await assert.rejects(context.readResponse({ok: false, status: 403, json: async () => ({error: 'loopback only'})}), /loopback only/);
  await assert.rejects(context.readResponse({ok: false, status: 502, json: async () => {throw new Error('HTML response');}}), /HTTP 502/);
  console.log('PASS: VLM JS units (DOM stubs): parameter wiring, current-preflight+consent gate, edit invalidation+abort, resource output, XSS quotes/tags, task states/log/package controls, HTTP errors. No browser render/compiler/board exercised.');
})().catch(error => {console.error(error); process.exitCode = 1;});
