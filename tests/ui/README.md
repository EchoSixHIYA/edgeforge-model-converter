# HF VLM UI checks

These tests validate the browser-facing integration only. None constitutes a real
TPU-MLIR compilation, model quality assessment, SDK native-runtime test or
BM1684X board acceptance.

## No browser required

```bash
python -m pytest -q tests/test_vlm_ui.py
node tests/ui/vlm_dom_unit.cjs
node --check static/app.js
```

The Python tests render the actual Flask/Jinja template and verify separate
CV/file versus HF/directory workflows, local-import explanations and numeric
bounds. If Node.js is available, pytest also executes the JS logic tests. The
latter use small DOM stubs, not a browser engine, to check real frontend functions:
parameter serialization and local-import header, preflight/consent gates, edit
invalidation, safe HTML/attribute escaping, resource output, task states,
log/package links, and HTTP error handling.

## Optional real browser regression

With the app running on localhost and an existing Playwright + Chromium install:

```bash
EDGEFORGE_UI_URL=http://127.0.0.1:5000 node tests/ui/vlm_workflow.cjs
```

Set `CHROMIUM_PATH` to override `/usr/bin/chromium`; set `EDGEFORGE_UI_OUTPUT` to
choose screenshot output (default `/tmp/edgeforge-vlm-ui`). The test mocks all
conversion/task API mutations. It checks missing files, preflight denial, stale
responses after edits, duplicate submission protection, experimental consent,
logs/errors, cancellation, successful-package links, legacy CV multipart upload,
keyboard workflow switching and mobile overflow. Its success response fixtures
are explicitly not proof of any real conversion.

During this integration session, actual Chromium launch was blocked by the
execution environment: sandbox Unix socket creation was denied; the one approved
outside-sandbox attempt failed during environment setup. Consequently the real
browser harness and screenshots are **not yet verified**. No permissions were
changed and no screenshots are claimed as evidence.
