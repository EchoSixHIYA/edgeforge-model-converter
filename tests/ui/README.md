# HF VLM UI checks

These tests validate the browser-facing integration only. None constitutes a real
TPU-MLIR compilation, model quality assessment, SDK native-runtime build or
BM1684X board acceptance. See [the validation guide](../../docs/testing.md) for CI,
skip reasons and the remaining Draft gates.

## No browser required

```bash
python -m pytest -q -ra tests/test_vlm_ui.py
node tests/ui/vlm_dom_unit.cjs
node --check static/app.js
```

The Python tests render the actual Flask/Jinja template and verify separate
CV/file versus HF/directory workflows, local-import explanations and numeric
bounds. If Node.js is unavailable, its pytest DOM-stub test explicitly skips;
that does not mean the JavaScript passed. CI installs Node and requires zero
unexpected Linux skips.

The JS tests use small DOM stubs, not a browser engine: parameter serialization
and local-import header, preflight/consent gates, edit invalidation, safe
HTML/attribute escaping, resource output, task states, log/package links, and
HTTP error handling.

## Real browser regression (mocked conversion APIs)

Use a separate lightweight Python environment with `requirements-test.txt`.
Install the pinned test-only Node dependencies and Playwright-managed Chromium:

```bash
npm ci --prefix tests/ui
(cd tests/ui && npx playwright install --with-deps chromium)
```

The `--with-deps` command installs OS browser libraries and may need administrator
permissions; it is explicit, not part of any model compiler installer. The
[official Playwright CI instructions](https://playwright.dev/docs/ci) describe this
setup. No TPU-MLIR, model weights or board credentials are installed.

Start the local app in one terminal:

```bash
EDGEFORGE_MODEL_ROOTS='' python -m flask --app app run \
  --host 127.0.0.1 --port 5127 --no-reload
```

Run the browser test in another terminal:

```bash
EDGEFORGE_UI_URL=http://127.0.0.1:5127 node tests/ui/vlm_workflow.cjs
```

By default Playwright launches its managed Chromium. Set `CHROMIUM_PATH` only for
an explicit existing-browser override; `/usr/bin/chromium` is no longer assumed.
Set `EDGEFORGE_UI_OUTPUT` for screenshots and `trace.zip`; the default is
`edgeforge-vlm-ui` inside the operating system temporary directory. The CI job
adds a 180-second process timeout and always uploads available logs/artifacts.
Missing Playwright or a failed browser launch **fails**, not skips, this job.

The test mocks all conversion/task API mutations. It checks missing files,
preflight denial, stale responses after edits, duplicate submission protection,
experimental consent, low-memory preset invalidation, logs/errors, cancellation,
successful-package links, legacy CV multipart upload, keyboard workflow switching
and mobile overflow. Its success fixtures do not prove any real conversion.
Page operations have explicit timeouts; a failure after launch saves a failure
screenshot and trace. Browser launch failure can have logs only.

## Verification record, not a blanket pass claim

The 2026-10-22 integration session could not launch Chromium because sandbox Unix
socket creation was denied; its real-browser harness was not verified.
On 2026-10-23 the pinned npm install and local Flask startup succeeded, but an
actual Chromium launch again failed at `process_singleton_posix.cc` with
`socket() failed: Operation not permitted`. That run reached no browser assertions
and produced no screenshots; no permission settings were changed.

The new GitHub Actions browser job is the portable verification path. Its actual
result must be checked on the PR commit before claiming browser success; adding
this workflow is not itself a passing run. Even a passing browser job covers
mocked API UI behavior, not a model end-to-end path.
