// 介面檢查（不需要瀏覽器）：所有 data-i18n 的文字三種語言都有；JS 語法正確；錯誤訊息都翻譯了。
const fs = require('fs'), path = require('path'), vm = require('vm');
const web = path.join(__dirname, '..', 'web');
const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
const ctx = { window: {} }; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(web, 'i18n.js'), 'utf8'), ctx);
new vm.Script(fs.readFileSync(path.join(web, 'app.js'), 'utf8'));
const I = ctx.window.I18N;
const keys = [...html.matchAll(/data-i18n="([^"]+)"/g)].map(m => m[1]);
const js = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
const used = [...js.matchAll(/\bt\('([a-z_A-Z]+)'/g)].map(m => m[1]).filter(k => !k.endsWith('_'));
const py = fs.readdirSync(path.join(__dirname, '..')).filter(f => f.endsWith('.py'))
  .map(f => fs.readFileSync(path.join(__dirname, '..', f), 'utf8')).join('\n');
const errs = [...new Set([...py.matchAll(/'((?:ERR|WARN)_[A-Z_]+)/g)].map(m => m[1]))].filter(e => e !== 'ERR_');
let bad = [];
for (const lang of ['zh', 'ja', 'en'])
  for (const k of [...keys, ...used, ...errs, 'ph_downloading_model', 'ph_loading_model', 'ph_analyzing', 'ph_writing'])
    if (!(k in I[lang])) bad.push(`${lang}:${k}`);
for (const lang of ['ja', 'en']) for (const k in I.zh) if (!(k in I[lang])) bad.push(`${lang} missing ${k}`);
if (bad.length) { console.error('缺少翻譯：', bad.join(', ')); process.exit(1); }
console.log(`OK：${keys.length} 個介面文字、${errs.length} 個錯誤訊息`);
