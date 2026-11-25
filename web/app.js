// DesktopFBT 介面
(() => {
  let api = null, cfg = null, env = null, lang = 'zh', view = 'live';
  let liveKind = 'window', livePath = null, videoPath = null, imagePaths = [];
  let lastState = null, polling = false;
  let motions = [], motionBusy = '', motionBusyUntil = 0;
  const $ = id => document.getElementById(id);

  // ---------- 文字
  function pickLang() {
    const want = cfg && cfg.language && cfg.language !== 'auto' ? cfg.language : (navigator.language || 'zh');
    lang = /^ja/.test(want) ? 'ja' : /^zh/.test(want) ? 'zh' : 'en';      // 其他語言的系統用英文
    document.documentElement.lang = { zh: 'zh-Hant', ja: 'ja', en: 'en' }[lang];
  }
  function t(key, args) {
    let s = (I18N[lang] && I18N[lang][key]) ?? I18N.zh[key] ?? key;
    for (const k in args || {}) s = s.replaceAll('{' + k + '}', args[k]);
    return s;
  }
  function applyText() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
      if (el.id === 'live-file' && livePath) return;
      if (el.id === 'video-file' && videoPath) return;
      if (el.id === 'images-file' && imagePaths.length) return;
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
  }
  function errText(e) {
    if (!e) return '';
    const [code, arg = ''] = e.split(/:(.*)/s);
    if (code === 'ERR_MODEL_DOWNLOAD') {
      const [file, why] = arg.split('|');
      return t(code, { file, why: t('MODEL_WHY_' + (why || 'unknown')) });
    }
    return t(code, { arg });
  }
  const base = p => (p || '').split(/[\\/]/).pop();
  function toast(msg) {
    const el = $('toast'); el.textContent = msg; el.hidden = false;
    clearTimeout(toast.h); toast.h = setTimeout(() => { el.hidden = true; }, 1600);
  }

  // ---------- 設定
  const getPath = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);
  function patchFor(path, value) {
    const out = {}; let o = out; const ks = path.split('.');
    ks.slice(0, -1).forEach(k => { o = o[k] = {}; });
    o[ks[ks.length - 1]] = value; return out;
  }
  // ---------- 新手導覽
  let obPath = [];
  function obShow(step) {
    document.querySelectorAll('.ob-step').forEach(s => s.classList.toggle('on', s.dataset.step === step));
    $('ob-back').hidden = obPath.length <= 1;
  }
  function obGo(step) { obPath.push(step); obShow(step); }
  function openOnboard() { obPath = []; $('ob-import-done').textContent = ''; $('onboard').hidden = false; obGo('unity'); }
  async function finishOnboard(kind) {
    $('onboard').hidden = true;
    cfg = await api.update_config({ app: { onboarded: true } });
    fillConfig(); show('live'); if (kind) setKind(kind);
  }
  function bindOnboard() {
    document.querySelectorAll('.ob-step .choice').forEach(b => b.onclick = () => finishOnboard(b.dataset.pick));
    document.querySelectorAll('[data-next]').forEach(b => b.onclick = () => obGo(b.dataset.next));
    $('ob-back').onclick = () => { obPath.pop(); obShow(obPath[obPath.length - 1]); };
    $('ob-skip').onclick = async () => { $('onboard').hidden = true; cfg = await api.update_config({ app: { onboarded: true } }); };
    $('ob-import').onclick = async () => {
      const r = await api.install_unity_tools();
      $('ob-import-done').textContent = r && r.error ? errText(r.error) : t('ob_imported');
    };
  }

  function fillConfig() {
    document.querySelectorAll('[data-cfg]').forEach(el => {
      const v = getPath(cfg, el.dataset.cfg);
      if (el.type === 'checkbox') el.checked = !!v; else if (v != null) el.value = v;
    });
    syncDerived();
  }
  function syncDerived() {
    $('delay-out').textContent = Number(cfg.live.delay).toFixed(1) + ' s';
    $('batch-out').textContent = cfg.pose.batch;
    $('delay-field').hidden = !cfg.live.precise;
    $('smooth-field').hidden = !!cfg.live.precise;
    $('tmodel-field').hidden = !cfg.live.temporal;
    $('osc-addr').textContent = $('osc-addr2').textContent = `${cfg.vrchat.host}:${cfg.vrchat.port}`;
    $('speed-out').textContent = '×' + Number((cfg.motion || {}).speed ?? 1).toFixed(2);
  }
  async function save(path, value) {
    try {
      cfg = await api.update_config(patchFor(path, value));
      if (path === 'language') { pickLang(); applyText(); }
      if (path === 'video.method') fillConfig();          // 影片和圖片兩頁各有一個同樣的選單
      syncDerived(); toast(t('saved'));
    } catch (e) { showError(String(e && e.message || e)); fillConfig(); }
  }
  function bindConfig() {
    document.querySelectorAll('[data-cfg]').forEach(el => {
      const ev = el.type === 'range' ? 'input' : 'change';
      el.addEventListener(ev, () => {
        let v = el.type === 'checkbox' ? el.checked : el.value;
        if ('num' in el.dataset) { v = Number(v); if (!isFinite(v)) return; }
        if (el.type === 'range') {                      // 拖動時只更新顯示，放開才存
          getPath(cfg, el.dataset.cfg.split('.').slice(0, -1).join('.') || '')[el.dataset.cfg.split('.').pop()] = v;
          syncDerived(); clearTimeout(el._h); el._h = setTimeout(() => save(el.dataset.cfg, v), 350); return;
        }
        save(el.dataset.cfg, v);
      });
    });
  }

  // ---------- 畫面切換
  function show(v) {
    view = v;
    document.querySelectorAll('.nav').forEach(b => b.classList.toggle('on', b.dataset.view === v));
    document.querySelectorAll('.view').forEach(s => s.classList.toggle('on', s.dataset.view === v));
    if (v === 'live' && liveKind === 'phone') pollPhone();
    if (v === 'motion') refreshMotions();
  }
  function setKind(k) {
    liveKind = k;
    document.querySelectorAll('#src-kind button').forEach(b => b.classList.toggle('on', b.dataset.kind === k));
    document.querySelectorAll('[data-for]').forEach(f => { f.hidden = f.dataset.for !== k; });
    if (k === 'camera' && !$('cam-list').options.length) refreshCameras();
    if (k === 'phone') pollPhone();
    lastState && render(lastState);
  }
  async function refreshCameras() {
    const sel = $('cam-list'); const keep = sel.value;
    const list = await api.list_cameras();
    sel.innerHTML = '';
    if (!list.length) { const o = new Option(t('cam_empty'), ''); o.disabled = true; sel.add(o); }
    list.forEach(c => sel.add(new Option(c.name, c.id)));
    if (keep && list.some(c => String(c.id) === keep)) sel.value = keep;
    lastState && render(lastState);
  }
  // 手機：選到這個分頁時才開伺服器，之後每秒更新連線狀態
  let phone = null, phoneTimer = 0;
  async function pollPhone() {
    clearTimeout(phoneTimer);
    if (liveKind !== 'phone' || view !== 'live') return;
    phone = await api.phone_info(true);
    const st = $('phone-st');
    if (phone.error) { st.className = 'phone-st'; st.textContent = errText(phone.error); }
    else {
      if (phone.qr && $('phone-qr').getAttribute('src') !== phone.qr) $('phone-qr').src = phone.qr;
      $('phone-url').textContent = phone.url || '';
      st.className = 'phone-st' + (phone.connected ? ' on' : '');
      st.textContent = phone.connected
        ? t('phone_on', { name: phone.name || '', fps: Math.round(phone.fps || 0), size: phone.size ? phone.size.join('×') : '' })
        : t('phone_wait');
    }
    lastState && render(lastState);
    phoneTimer = setTimeout(pollPhone, 1000);
  }
  function liveReady() {
    if (liveKind === 'window') return !!$('win-list').value;
    if (liveKind === 'file') return !!livePath;
    if (liveKind === 'camera') return !!($('cam-url').value.trim() || $('cam-list').value);
    return !!(phone && phone.connected);
  }
  async function refreshWindows() {
    const sel = $('win-list'); const keep = sel.value;
    const list = await api.list_windows();
    sel.innerHTML = '';
    if (!list.length) { const o = new Option(t('win_empty'), ''); o.disabled = true; sel.add(o); }
    list.forEach(w => sel.add(new Option(w.title, w.id)));
    if (keep && list.some(w => String(w.id) === keep)) sel.value = keep;
  }
  // ---------- 動作庫：點一下就播到 Avatar
  const isMotion = st => st && st.mode === 'live' && st.job && st.job.kind === 'motion';
  const clock = s => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
  async function refreshMotions() {
    let hidden = 0;
    try { const r = await api.list_motions() || {}; motions = r.items || []; hidden = r.hidden || 0; } catch (e) { motions = []; }
    $('motion-restore').hidden = !hidden;
    $('motion-restore').textContent = t('motion_hidden', { n: hidden });
    const box = $('motion-list'); box.innerHTML = '';
    $('motion-empty').hidden = motions.length > 0;
    motions.forEach(m => {
      const row = document.createElement('div'); row.className = 'motion'; row.dataset.path = m.path;
      const play = document.createElement('button'); play.className = 'play'; play.title = m.path;
      const name = document.createElement('b'); name.textContent = m.name;
      const sub = document.createElement('span');
      sub.textContent = [m.pose ? t('motion_pose') : clock(m.seconds), t(m.kind === 'added' ? 'motion_tag_added' : 'motion_tag_export')].join(' · ');
      const bar = document.createElement('i');
      play.append(name, sub, bar);
      play.onclick = async () => {
        if (lastState && lastState.mode === 'export') return;
        $('motion-result').hidden = true;
        motionBusy = m.path; motionBusyUntil = Date.now() + 1500; markMotions(lastState);
        try { await api.start_motion(m.path); } catch (e) { showError(String(e && e.message || e)); }
      };
      row.append(play);
      const del = document.createElement('button'); del.className = 'del'; del.textContent = '×';
      del.title = t(m.kind === 'added' ? 'motion_remove' : 'motion_hide');      // 轉換結果：只是不列出來，檔案還在匯出資料夾
      del.setAttribute('aria-label', del.title);
      del.onclick = async () => {
        if (isMotion(lastState) && lastState.job.path === m.path) await api.stop();
        await api.remove_motion(m.path); refreshMotions();
      };
      row.append(del);
      box.append(row);
    });
    markMotions(lastState);
  }
  function markMotions(st) {
    const on = isMotion(st) ? st.job.path : '';
    if (on || Date.now() > motionBusyUntil) motionBusy = '';         // 剛點下去、狀態還沒跟上的那一瞬間先亮起來
    const cur = on || motionBusy;
    document.querySelectorAll('#motion-list .motion').forEach(row => {
      const mine = row.dataset.path === cur;
      row.classList.toggle('on', mine);
      const j = st && st.job, bar = row.querySelector('i');
      bar.style.width = mine && on && j.duration > 0 && !j.pose ? (100 * Math.min(1, j.position / j.duration)).toFixed(1) + '%' : mine && on ? '100%' : '0';
    });
  }
  function motionReport(r) {
    const box = $('motion-result'); box.innerHTML = '';
    const bad = (r && r.errors) || [];
    const warn = (r && r.warnings) || [];
    bad.forEach(e => { const d = document.createElement('div'); d.textContent = e.name + '：' + errText(e.error); box.append(d); });
    warn.forEach(e => { const d = document.createElement('div'); d.className = 'soft'; d.textContent = e.name + '：' + t(e.warning); box.append(d); });
    box.className = 'result ' + (bad.length ? 'warn' : 'soft');
    box.hidden = !(bad.length || warn.length);
    if (r && r.added && r.added.length) toast(t('motion_added', { n: r.added.length }));
  }

  function setFile(el, path, label) {
    el.textContent = label || base(path) || t('src_none');
    el.parentElement.classList.toggle('set', !!path);
    el.parentElement.title = path || '';
  }

  // ---------- 狀態
  function showError(e) {
    $('error').hidden = !e; $('error-text').textContent = errText(e);
  }
  function busy(st) { return st.mode !== 'idle'; }
  let lastMode = 'idle';
  function render(st) {
    lastState = st;
    const running = busy(st);
    const motion = isMotion(st), live = st.mode === 'live' && !motion, exp = st.mode === 'export';
    if (lastMode === 'export' && st.mode === 'idle' && view === 'motion') refreshMotions();   // 剛轉好的動作馬上出現在清單
    lastMode = st.mode;
    // 指示燈
    const tal = $('tally');
    tal.className = 'tally' + ((live || motion) && st.phase === 'running' ? ' live' : running ? ' busy' : '');
    $('tally-text').textContent = motion ? t('st_motion', { name: st.job.name || '' }) : t(live ? 'st_live' : exp ? 'st_export' : 'st_idle');
    // 預覽
    const img = $('preview');
    if (st.preview) { if (img.src !== st.preview) img.src = st.preview; img.classList.add('on'); }
    else if (!running) img.classList.remove('on');
    $('empty').hidden = img.classList.contains('on');
    // 進度
    const ph = st.phase && st.phase !== 'running' ? st.phase : '';
    let pct = null;
    if (ph === 'downloading_model' || ph === 'downloading_gemx') pct = st.model.progress;
    else if (exp) pct = st.progress;
    $('phase').hidden = !ph && !(exp && st.progress > 0 && st.progress < 1);
    $('countdown').hidden = ph !== 'countdown' || !st.countdown;
    if (ph === 'countdown') $('countdown').textContent = st.countdown || '';
    let txt = ph ? t('ph_' + ph, { p: ph === 'countdown' ? st.countdown : Math.round(100 * (st.model.progress || 0)) }) : '';
    if (exp && !['downloading_model', 'downloading_gemx', 'loading_model', 'compiling_model', 'preparing_accel', 'benchmarking'].includes(ph)) {
      const j = st.job || {};
      txt = (txt ? txt + '  ' : '') + (j.total ? t('imgs_progress', { n: j.done || 0, total: j.total }) : t('progress', { p: Math.round(100 * st.progress) }));
    }
    $('phase-text').textContent = txt;
    $('phase-bar').style.width = pct == null ? '100%' : Math.round(100 * pct) + '%';
    // 數字
    const on = running && st.fps > 0;
    $('m-fps').textContent = on ? Math.round(st.fps) : '–';
    $('m-src').textContent = live && st.src_fps > 0 ? Math.round(st.src_fps) : '–';
    $('m-model').textContent = running && !motion && st.model_ms > 0 ? st.model_ms.toFixed(1) : '–';
    $('m-prec').textContent = 'ms' + (st.precision ? ' · ' + st.precision.toUpperCase() : '');
    $('m-delay').textContent = live && st.delay ? st.delay.toFixed(2) : '–';
    const md = st.models;
    $('m-using').textContent = running && !motion && md
      ? [md.pose + ' · ' + ({ dml: 'DirectML', migraphx: 'MIGraphX', cuda: 'CUDA', cpu: 'CPU' }[md.device] || md.device),
         md.temporal ? { agformer_b: 'MotionAGFormer-B', agformer_s: 'MotionAGFormer-S', motionbert: 'MotionBERT-Lite' }[md.temporal] || md.temporal : null]
        .filter(Boolean).join('\n') : '–';
    const big = document.querySelector('.meters .big');
    big.classList.toggle('good', (live || motion) && st.fps >= 55);
    big.classList.toggle('low', (live || motion) && st.fps > 0 && st.fps < 55);
    // 提示
    const c = st.check; let hint = '';
    if (running && st.phase === 'running' && c) hint = !c.person ? t('chk_noperson') : c.edge ? t('chk_edge') : !c.feet ? t('chk_feet') : '';
    $('hint').hidden = !hint; $('hint').textContent = hint;
    showError(st.error);
    if (st.osc) $('osc-sent').textContent = $('osc-sent2').textContent = st.osc.sent;
    const tm = live && st.temporal;
    $('tm-line').hidden = !tm;
    if (tm) {
      const all = tm.used + tm.passed;
      $('tm-line').textContent = t('tm_line', { ms: tm.ms.toFixed(0), dev: tm.device.toUpperCase(), pct: all ? Math.round(100 * tm.used / all) : 0 });
    }
    // 按鈕
    const liveBtn = $('live-go');
    liveBtn.textContent = t(live ? 'stop' : 'start');
    liveBtn.classList.toggle('stop', live);
    liveBtn.disabled = exp || (!live && !liveReady());
    for (const [id, have] of [['video-go', !!videoPath], ['images-go', imagePaths.length > 0]]) {
      const b = $(id), mine = exp && st.job && ((id === 'video-go') === (st.job.kind === 'file'));
      b.textContent = mine ? t('stop') : t(id === 'video-go' ? 'export_video' : 'export_images');
      b.classList.toggle('stop', !!mine);
      b.disabled = (running && !mine) || (!running && !have);
    }
    $('motion-stop').disabled = !motion;
    $('motion-add').disabled = exp;
    document.querySelectorAll('#motion-list .play').forEach(b => { b.disabled = exp; });
    markMotions(st);
    document.querySelectorAll('.panel select, .panel input').forEach(el => {
      if (el.closest('[data-view="settings"]') && el.dataset.cfg !== 'language') el.disabled = running;
    });
    renderResult(st);
  }
  let resultKey = '';
  function renderResult(st) {
    const j = st.job;
    if (!j || !j.export || st.mode !== 'idle') return;
    const box = $(j.kind === 'image' ? 'images-result' : 'video-result');
    const key = j.kind + '|' + j.export + '|' + (j.unity || '');
    if (key === resultKey && !box.hidden) return;   // 每次輪詢都重建按鈕的話，點下去的那一刻按鈕已經被換掉了
    resultKey = key;
    box.innerHTML = '';
    const p = document.createElement('div');
    p.textContent = j.kind === 'image' ? t('done_images', { saved: j.saved, skip: (j.skipped || []).length })
      : t('done_saved', { name: base(j.unity || j.export) });
    box.append(p);
    if (j.project) { const q = document.createElement('div'); q.textContent = t('done_project'); box.append(q); }
    if (j.project_err) { const q = document.createElement('div'); q.textContent = errText(j.project_err); box.append(q); }
    const row = document.createElement('div'); row.className = 'row';
    const b = document.createElement('button'); b.className = 'ghost'; b.textContent = t('open_exports');
    b.onclick = () => api.open_exports(j.unity || j.export); row.append(b);
    if (j.unity) {
      const u = document.createElement('button'); u.className = 'ghost'; u.textContent = t('import_unity');
      u.onclick = async () => { const r = await api.import_to_unity(j.unity); if (r && r.error) showError(r.error); };
      row.append(u);
    }
    if (j.anim) {
      const m = document.createElement('button'); m.className = 'ghost'; m.textContent = t('motion_play_now');
      m.onclick = async () => { show('motion'); motionBusy = j.anim; motionBusyUntil = Date.now() + 1500; await api.start_motion(j.anim); };
      row.append(m);
    } else if (j.kind === 'image') {
      const m = document.createElement('button'); m.className = 'ghost'; m.textContent = t('motion_open');
      m.onclick = () => show('motion'); row.append(m);
    }
    box.append(row);
    box.hidden = false;
  }
  async function poll() {
    if (polling) return; polling = true;
    try { render(await api.get_state()); } catch (e) { /* 視窗關閉中 */ }
    polling = false;
  }

  // ---------- 動作
  function bindActions() {
    document.querySelectorAll('.nav').forEach(b => b.onclick = () => show(b.dataset.view));
    document.querySelectorAll('#src-kind button').forEach(b => b.onclick = () => { setKind(b.dataset.kind); if (lastState) render(lastState); });
    $('win-refresh').onclick = refreshWindows;
    $('cam-refresh').onclick = refreshCameras;
    $('cam-list').onchange = () => lastState && render(lastState);
    $('cam-url').oninput = () => lastState && render(lastState);
    $('ob-open').onclick = () => openOnboard();
    bindOnboard();
    $('win-list').onchange = () => lastState && render(lastState);
    $('live-pick').onclick = async () => { const p = await api.pick_video(); if (p) { livePath = p; setFile($('live-file'), p); lastState && render(lastState); } };
    $('video-pick').onclick = async () => { const p = await api.pick_video(); if (p) { videoPath = p; setFile($('video-file'), p); $('video-result').hidden = true; lastState && render(lastState); } };
    $('images-pick').onclick = async () => {
      const ps = await api.pick_images();
      if (ps && ps.length) { imagePaths = ps; setFile($('images-file'), ps[0], ps.length > 1 ? t('n_images', { n: ps.length }) : base(ps[0])); $('images-result').hidden = true; lastState && render(lastState); }
    };
    $('live-go').onclick = async () => {
      if (lastState && lastState.mode === 'live' && !isMotion(lastState)) return api.stop();
      if (liveKind === 'window') {
        const sel = $('win-list'); const o = sel.options[sel.selectedIndex];
        await api.start_live({ kind: 'window', id: Number(sel.value), title: o ? o.text : '' });
      } else if (liveKind === 'camera') {
        const url = $('cam-url').value.trim();
        if (url) await api.start_live({ kind: 'url', url });
        else {
          const sel = $('cam-list'); const o = sel.options[sel.selectedIndex];
          await api.start_live({ kind: 'camera', id: Number(sel.value), title: o ? o.text : '' });
        }
      } else if (liveKind === 'phone') await api.start_live({ kind: 'phone' });
      else await api.start_live({ kind: 'file', path: livePath });
    };
    $('video-go').onclick = () => (lastState && lastState.mode === 'export') ? api.stop() : api.start_export(videoPath);
    $('images-go').onclick = () => (lastState && lastState.mode === 'export') ? api.stop() : api.start_image_export(imagePaths);
    $('unity-install').onclick = async () => {
      const r = await api.install_unity_tools(), box = $('unity-result');
      if (r.error) { box.hidden = true; return showError(r.error); }
      box.textContent = t('unity_package', { name: base(r.package) });
      box.hidden = false;
    };
    $('motion-add').onclick = async () => { const r = await api.add_motions(); motionReport(r); refreshMotions(); };
    $('motion-refresh').onclick = refreshMotions;
    $('motion-restore').onclick = async () => { await api.restore_motions(); refreshMotions(); };
    $('motion-stop').onclick = () => api.stop();
    $('open-folder').onclick = () => api.open_folder();
    $('error-close').onclick = () => { api.clear_error(); $('error').hidden = true; };
  }

  async function init(a) {
    api = a;
    [cfg, env] = await Promise.all([api.get_config(), api.environment()]);
    pickLang(); applyText();
    $('version').textContent = 'v' + env.version;
    const devs = new Set(['auto', ...(env.devices || [])]);
    document.querySelectorAll('#dev-select option').forEach(o => { o.hidden = !devs.has(o.value); });
    fillConfig(); bindConfig(); bindActions();
    show('live'); setKind('window');
    await refreshWindows();
    if (!(cfg.app && cfg.app.onboarded)) openOnboard();
    poll(); setInterval(poll, 100);
  }

  window.addEventListener('DOMContentLoaded', () => {
    if (window.pywebview && window.pywebview.api) init(window.pywebview.api);
    else if (/[?&]mock/.test(location.search)) {
      const sc = document.createElement('script'); sc.src = '../tests/mock_api.js';
      sc.onload = () => init(window.MockApi); document.body.append(sc);
    } else window.addEventListener('pywebviewready', () => init(window.pywebview.api));
  });
  window.YFBT = { t, errText };
})();
