// Real board/browser acceptance: no mocked model responses.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = path.resolve(__dirname, '../evidence');
fs.mkdirSync(output, {recursive: true});
(async () => {
  const browser = await chromium.launch({headless: true});
  const context = await browser.newContext({viewport: {width: 1360, height: 980}});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(process.env.COSMOS_URL || 'http://192.168.1.117:8080/', {waitUntil: 'domcontentloaded'});
    await page.locator('#submit').waitFor({state: 'visible'});
    await page.waitForFunction(() => !document.getElementById('submit').disabled, null, {timeout: 120000});
    await page.locator('textarea[name=prompt]').fill('Describe this image in one concise sentence.');
    if (!process.env.COSMOS_TEST_IMAGE) throw new Error('Set COSMOS_TEST_IMAGE to an actual test photograph.');
    await page.locator('#media').setInputFiles(process.env.COSMOS_TEST_IMAGE);
    await page.locator('input[name=max_new_tokens]').fill('96');
    await page.locator('#submit').click();
    await page.waitForFunction(() => document.getElementById('json').textContent.includes('"completed"') || !document.getElementById('notice').hidden, null, {timeout: 240000});
    assert.equal(await page.locator('#notice').isVisible(), false, await page.locator('#notice').textContent());
    const result = JSON.parse(await page.locator('#json').textContent());
    assert.equal(result.ok, true);
    assert.ok(result.result.text.length > 0);
    fs.writeFileSync(path.join(output, 'browser-image-result.json'), JSON.stringify(result, null, 2));
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    await page.locator('summary').click();
    assert.equal(await page.locator('#json').isVisible(), true);
    assert.equal(await page.locator('#copy').isEnabled(), true);
    await page.locator('summary').click();
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({browser: 'passed', elapsed_ms: result.elapsed_ms, text: result.result.text}));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
