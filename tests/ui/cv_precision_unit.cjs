const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../static/app.js'), 'utf8');
const start = source.indexOf('function choosePrecision(');
const end = source.indexOf('function syncOptions()', start);
assert.ok(start >= 0 && end > start);
const context = vm.createContext({});
vm.runInContext(source.slice(start, end), context);
const select = context.choosePrecision;

// First render, model/vendor/chip changes, and form resets prefer floating point.
assert.equal(select(['i8', 'fp'], 'i8', true), 'fp');
assert.equal(select(['i8', 'f16', 'f32'], 'i8', true), 'f32');
assert.equal(select(['f16', 'i8'], 'i8', true), 'f16');
assert.equal(select(['bf16', 'i8'], 'i8', true), 'bf16');
assert.equal(select(['i8'], 'i8', true), '');
// An explicitly chosen INT8 remains available within the same context.
assert.equal(select(['f32', 'i8'], 'i8', false), 'i8');
assert.equal(select(['f32', 'i8'], 'f32', false), 'f32');
assert.equal(select(['f32', 'i8'], 'fp', false), 'f32');
assert.match(source, /precisionContext !== precisionKey/);
assert.match(source, /form\.reset\(\);\s*inputSizeKey = '';\s*precisionKey = '';/);
console.log('PASS: CV floating defaults, context changes, reset and manual INT8 selection');
