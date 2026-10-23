/* Browser-only regression; mocked task success is NOT a real conversion.
 * Run a local Flask app, then:
 * EDGEFORGE_UI_URL=http://127.0.0.1:5127 node tests/ui/vlm_workflow.cjs
 * Install the pinned test-only dependency with npm ci --prefix tests/ui.
 * Defaults to Playwright-managed Chromium; CHROMIUM_PATH is an explicit override.
 */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const baseURL = process.env.EDGEFORGE_UI_URL || 'http://127.0.0.1:5127';
const output = process.env.EDGEFORGE_UI_OUTPUT || path.join(os.tmpdir(), 'edgeforge-vlm-ui');
fs.mkdirSync(output, {recursive: true});

(async () => {
  const launchOptions = {headless: true};
  if (process.env.CHROMIUM_PATH) launchOptions.executablePath = process.env.CHROMIUM_PATH;
  const browser = await chromium.launch(launchOptions);
  const context = await browser.newContext({viewport: {width: 1440, height: 1100}});
  const page = await context.newPage();
  page.setDefaultTimeout(10000);
  page.setDefaultNavigationTimeout(15000);
  await context.tracing.start({screenshots: true, snapshots: true, sources: true});
  try {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let tasks = [];
    let preflightMode = 'blocked';
    let preflightRequests = [];
    let createdRequests = [];
    let cvRequests = [];
    let releaseCheck;
    const task = {id: 'uitest001', input_kind: 'hf_directory', architecture: 'cosmos_reason2_2b',
      platform: 'bm1684x', precision: 'w4bf16', vendor: 'sophon', original_filename: 'Cosmos <test> "file"',
      created_at: '2026-10-22T08:00:00Z', progress: 0, message: '等待转换', status: 'queued',
      parameters: {seq_length: 2048, max_pixels: 147456, jobs: 1}, log_tail: ''};
    await page.route('**/api/tasks', async route => {
      if (route.request().method() === 'POST') {
        cvRequests.push(route.request().postData());
        return route.fulfill({json: {...task, id: 'cv001', input_kind: 'file', architecture: 'mobilenet', vendor: 'rockchip', platform: 'rk3588', precision: 'fp', original_filename: 'sample.pth', status: 'completed', progress: 100}});
      }
      return route.fulfill({json: tasks});
    });
    await page.route('**/api/tasks/*/events', route => {
      const id = route.request().url().split('/').at(-2);
      const current = tasks.find(item => item.id === id) || {...task, id, status: 'failed', progress: 17, message: '测试错误 <img src=x onerror=alert(1)>', log_tail: '<script>not executable</script>\ncompiler unavailable'};
      return route.fulfill({status: 200, contentType: 'text/event-stream', body: `data: ${JSON.stringify(current)}\n\n`});
    });
    await page.route('**/api/vlm/preflight', async route => {
      const request = route.request();
      preflightRequests.push({body: request.postDataJSON(), headers: request.headers()});
      if (preflightMode === 'delayed') await new Promise(resolve => { releaseCheck = resolve; });
      if (preflightMode === 'forbidden') return route.fulfill({status: 403, json: {error: '只允许本机回环访问'}});
      return route.fulfill({json: {ready: preflightMode !== 'blocked',
        errors: preflightMode === 'blocked' ? ['缺少真实 .safetensors 权重', '缺少 tokenizer.json', 'TPU-MLIR 编译器未安装'] : [],
        warnings: ['实验性：未进行 BM1684X 板端验收'],
        model: {architecture: 'Qwen3VLForConditionalGeneration'}, parameters: request.postDataJSON(),
        resources: {available_memory_bytes: 16 * 1024 ** 3, minimum_memory_bytes: 32 * 1024 ** 3, disk_free_bytes: 120 * 1024 ** 3, estimated_work_disk_bytes: 80 * 1024 ** 3},
        toolchain: {ready: preflightMode !== 'blocked', version: '1.28.1'}}});
    });
    await page.route('**/api/vlm/tasks', async route => {
      createdRequests.push({body: route.request().postDataJSON(), headers: route.request().headers()});
      await new Promise(resolve => setTimeout(resolve, 150));
      return route.fulfill({status: 202, json: task});
    });
    await page.route('**/api/tasks/*/cancel', route => {
      tasks = [{...task, id: 'cancel001', status: 'canceled', message: '用户取消'}];
      return route.fulfill({json: tasks[0]});
    });
    await page.goto(baseURL, {waitUntil: 'networkidle'});
    assert.equal(await page.locator('#convertForm').isVisible(), true);
    assert.equal(await page.locator('#vlmForm').isVisible(), false);
    await page.locator('#vlmTab').click();
    assert.equal(await page.locator('#vlmForm').isVisible(), true);
    assert.equal(await page.locator('#modelFile').isVisible(), false);
    assert.equal(await page.locator('#vlmForm input[type=file]').count(), 0);
    assert.equal(await page.locator('#vlmSubmit').isDisabled(), true);
    assert.equal(await page.locator('#vlmSeqLength').getAttribute('step'), '128');
    assert.equal(await page.locator('#vlmSeqLength').getAttribute('max'), '8192');
    assert.equal(await page.locator('#vlmMaxPixels').getAttribute('step'), '1024');
    assert.equal(await page.locator('#vlmJobs').getAttribute('readonly'), '');
    await page.locator('#vlmModelDir').fill('/models/Cosmos-Reason2-2B');
    await page.locator('#vlmPreflight').click();
    await page.waitForFunction(() => document.querySelector('#vlmPreflightResult').textContent.includes('缺少真实'));
    assert.equal(await page.locator('#vlmSubmit').isDisabled(), true);
    assert.match(await page.locator('#vlmPreflightResult').textContent(), /tokenizer/);
    assert.match(await page.locator('#vlmPreflightResult').textContent(), /16.0 GiB/);
    assert.equal(preflightRequests[0].headers['x-edgeforge-local-import'], '1');
    assert.equal(preflightRequests[0].body.jobs, 1);
    assert.equal(preflightRequests[0].body.seq_length, 2048);
    assert.equal(preflightRequests[0].body.max_pixels, 147456);
    assert.equal('model' in preflightRequests[0].body, false);
    assert.equal('max_frames' in preflightRequests[0].body, false);
    await page.screenshot({path: path.join(output, 'vlm-desktop-blocked.png'), fullPage: true});

    // A stale successful response must not re-enable submission after edits.
    preflightMode = 'delayed';
    await page.locator('#vlmPreflight').click();
    await page.waitForFunction(() => document.querySelector('#vlmPreflight').disabled);
    for (let i = 0; !releaseCheck && i < 50; i += 1) await page.waitForTimeout(20);
    assert.ok(releaseCheck, 'delayed preflight reached the mock API');
    await page.locator('#vlmSeqLength').fill('2176');
    releaseCheck();
    await page.waitForTimeout(200);
    assert.equal(await page.locator('#vlmSubmit').isDisabled(), true);
    assert.match(await page.locator('#vlmPreflightResult').textContent(), /重新预检/);

    // HTTP denial cannot become a ready state.
    preflightMode = 'forbidden';
    await page.locator('#vlmPreflight').click();
    await page.waitForFunction(() => document.querySelector('#vlmError').textContent.includes('回环'));
    assert.equal(await page.locator('#vlmSubmit').isDisabled(), true);

    preflightMode = 'ready';
    await page.locator('#vlmPreflight').click();
    await page.waitForFunction(() => document.querySelector('#vlmPreflightResult').textContent.includes('预检通过'));
    assert.equal(await page.locator('#vlmAcknowledge').count(), 0);
    assert.equal(await page.locator('#vlmSubmit').isEnabled(), true);
    await page.locator('#vlmLowMemoryPreset').click();
    assert.equal(await page.locator('#vlmSubmit').isDisabled(), true);
    assert.equal(await page.locator('#vlmSeqLength').inputValue(), '1024');
    assert.equal(await page.locator('#vlmMaxPixels').inputValue(), '65536');
    assert.equal(await page.locator('#vlmPrecision').inputValue(), 'w4bf16');
    assert.match(await page.locator('#vlmPresetHint').textContent(), /旧预检已失效/);
    await page.locator('#vlmPreflight').click();
    await page.waitForFunction(() => !document.querySelector('#vlmSubmit').disabled);
    await page.evaluate(() => {
      const form = document.querySelector('#vlmForm');
      form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
      form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
    });
    await page.waitForFunction(() => document.querySelector('[data-job="uitest001"] .status')?.textContent === '失败');
    assert.equal(createdRequests.length, 1);
    assert.equal(createdRequests[0].body.seq_length, 1024);
    assert.equal(createdRequests[0].body.max_pixels, 65536);
    assert.equal(createdRequests[0].body.acknowledge_experimental, true);
    assert.equal(createdRequests[0].headers['x-edgeforge-local-import'], '1');
    assert.equal(await page.locator('[data-job="uitest001"] a[href$="/download/package"]').count(), 0);
    assert.equal(await page.locator('[data-job="uitest001"] img').count(), 0);
    assert.equal(await page.locator('[data-job="uitest001"] script').count(), 0);
    await page.locator('[data-job="uitest001"] summary').click();
    assert.match(await page.locator('[data-job="uitest001"] .task-log').textContent(), /compiler unavailable/);
    assert.equal(await page.locator('[data-job="uitest001"] a[href$="/download/log"]').count(), 1);
    assert.equal(await page.locator('#vlmSubmit').isDisabled(), true);

    tasks = [{...task, id: 'complete001', status: 'completed', progress: 100, message: '模拟 SDK 已生成'}];
    await page.locator('#refresh').click();
    await page.waitForSelector('[data-job="complete001"] a[href$="/download/package"]');
    tasks = [{...task, id: 'cancel001'}];
    await page.locator('#refresh').click();
    await page.waitForSelector('[data-cancel="cancel001"]');
    page.once('dialog', dialog => dialog.accept());
    await page.locator('[data-cancel="cancel001"]').click();
    await page.waitForFunction(() => document.querySelector('[data-job="cancel001"] .status')?.textContent === '已取消');
    assert.equal(await page.locator('[data-job="cancel001"] a[href$="/download/package"]').count(), 0);

    // Switching back retains the original upload protocol and acceptance hint.
    await page.locator('#cvTab').click();
    await page.locator('#vendor').selectOption('rockchip');
    await page.locator('#architecture').selectOption('mobilenet');
    await page.locator('#precision').selectOption('fp');
    await page.locator('#modelFile').setInputFiles({name: 'sample.pth', mimeType: 'application/octet-stream', buffer: Buffer.from('UI fixture, not real weights')});
    await page.locator('#submitButton').click();
    await page.waitForSelector('[data-job="cv001"]');
    assert.equal(cvRequests.length, 1);
    assert.match(cvRequests[0], /name="model"; filename="sample.pth"/);
    assert.match(cvRequests[0], /name="vendor"\r\n\r\nrockchip/);
    assert.equal(await page.locator('#vlmForm').isVisible(), false);
    await page.locator('#vlmTab').focus();
    await page.keyboard.press('Home');
    assert.equal(await page.locator('#cvTab').getAttribute('aria-selected'), 'true');
    await page.keyboard.press('End');
    assert.equal(await page.locator('#vlmTab').getAttribute('aria-selected'), 'true');
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(output, 'vlm-mobile.png'), fullPage: true});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log('PASS: HF VLM UI / payloads / stale preflight / denial / clean UI copy / duplicate submit / logs / cancellation / downloads / CV upload / mobile layout.');
  } catch (error) {
    await page.screenshot({path: path.join(output, 'vlm-failure.png'), fullPage: true}).catch(() => {});
    throw error;
  } finally {
    try { await context.tracing.stop({path: path.join(output, 'trace.zip')}); }
    finally { await browser.close(); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
