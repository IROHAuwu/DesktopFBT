// 介面預覽用的假 API：web/index.html?mock=live | export | idle
(() => {
  const scene = (location.search.match(/mock=(\w+)/) || [])[1] || 'live';
  function frame(phase) {
    const c = document.createElement('canvas'); c.width = 640; c.height = 360;
    const g = c.getContext('2d');
    g.fillStyle = '#3a3f4a'; g.fillRect(0, 0, 640, 360); g.fillStyle = '#2c3038'; g.fillRect(0, 260, 640, 100);
    const x = 320 + Math.sin(phase) * 60;
    g.lineWidth = 4; g.lineCap = 'round';
    const L = (a, b, col) => { g.strokeStyle = col; g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); };
    L([x, 100], [x, 180], '#e6e6e6');
    L([x - 22, 108], [x - 40, 160], '#ffaa3c'); L([x + 22, 108], [x + 40, 160], '#5a78ff');
    L([x - 12, 180], [x - 20, 245], '#ffaa3c'); L([x - 20, 245], [x - 22, 300], '#ffaa3c');
    L([x + 12, 180], [x + 20, 245], '#5a78ff'); L([x + 20, 245], [x + 24, 300], '#5a78ff');
    g.fillStyle = '#fff'; g.beginPath(); g.arc(x, 80, 15, 0, 7); g.fill();
    return c.toDataURL('image/jpeg', 0.8);
  }
  let t = 0;
  const calls = window.MockCalls = [];
  const cfg = {
    language: 'zh', user_height_m: 1.65,
    pose: { device: 'auto', precision: 'auto', batch: 1, tta: false, head_cm: 10 },
    vrchat: { host: '127.0.0.1', port: 9000 },
    live: { fps: 30, precise: false, delay: 0.8, smoothness: 0.85, grounded: true, temporal: true, temporal_model: 'motionbert' },
    motion: { loop: true, speed: 1.0 },
    app: { onboarded: !/onboard/.test(location.search) },
    video: { hfov_deg: 60, height_m: 0, mirror: false, loop: true, unity_project: 'C:\\Users\\me\\My project', method: 'gemx' },
  };
  const merge = (a, b) => { for (const k in b) { if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k]) merge(a[k], b[k]); else a[k] = b[k]; } return a; };
  window.MockApi = {
    get_config: async () => JSON.parse(JSON.stringify(cfg)),
    update_config: async p => JSON.parse(JSON.stringify(merge(cfg, p))),
    environment: async () => ({ version: '1.0.0', devices: ['migraphx', 'dml', 'cpu'], model: true }),
    list_windows: async () => [{ id: 1, title: 'Dance practice - YouTube - Google Chrome' }, { id: 2, title: 'Discord' }],
    list_cameras: async () => [{ id: 0, name: 'USB Camera' }, { id: 1, name: 'DroidCam Source 3' }],
    phone_info: async () => ({ running: true, url: 'https://192.168.1.20:8443/', connected: true, name: 'iPhone', fps: 30, size: [1280, 720],
      qr: 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="#fff"/><rect x="1" y="1" width="3" height="3"/></svg>') }),
    start_live: async () => { calls.push(['live']); }, start_export: async () => {}, start_image_export: async () => {}, stop: async () => { calls.push(['stop']); },
    open_exports: async () => {}, open_folder: async () => {}, clear_error: async () => {},
    pick_video: async () => 'C:\\Users\\me\\Videos\\dance practice.mp4',
    pick_images: async () => ['C:\\me\\pose1.jpg', 'C:\\me\\pose2.jpg'],
    pick_unity_project: async () => ({ path: 'C:\\Users\\me\\My project' }), clear_unity_project: async () => {},
    install_unity_tools: async () => ({ package: 'C:\\exports\\DesktopFBT_Tools.unitypackage' }), import_to_unity: async (p) => ({ package: p }),
    list_motions: async () => ({ hidden: 2, items: [
      { path: 'C:\\m\\wave.anim', name: 'wave', kind: 'added', seconds: 3.2, pose: false },
      { path: 'C:\\e\\dance practice.anim', name: 'dance practice', kind: 'export', seconds: 74, pose: false },
      { path: 'C:\\e\\pose1_pose.anim', name: 'pose1_pose', kind: 'export', seconds: 0, pose: true }] }),
    restore_motions: async () => { calls.push(['restore']); return true; },
    add_motions: async () => ({ added: ['C:\\m\\new.anim'], errors: [{ name: 'generic.anim', error: 'ERR_ANIM_NOT_HUMANOID' }], warnings: [{ name: 'new.anim', warning: 'WARN_ANIM_TRANSFORMS' }] }),
    remove_motion: async p => { calls.push(['remove', p]); return true; }, start_motion: async p => { calls.push(['start', p]); },
    get_state: async () => {
      t += 0.1;
      if (scene === 'idle') return { mode: 'idle', phase: '', error: null, progress: 0, fps: 0, model: { progress: 1 } };
      if (scene === 'motion') return { mode: 'live', phase: 'running', error: null, progress: 0, fps: 60, src_fps: 0, model_ms: 0, delay: 0, model: { progress: 1 },
        job: { kind: 'motion', path: 'C:\\e\\dance practice.anim', name: 'dance practice', duration: 74, position: (t * 3) % 74, pose: false },
        osc: { host: '127.0.0.1', port: 9000, sent: Math.round(t * 60) }, preview: frame(t) };
      const exp = scene === 'export';
      return {
        mode: exp ? 'export' : 'live', phase: exp ? 'analyzing' : 'running', error: null, progress: exp ? (t / 20) % 1 : 0,
        models: { pose: 'RTMW3D-x', device: 'dml', temporal: 'agformer_b' }, fps: 60 + Math.sin(t) * 2, src_fps: 60, model_ms: 7.4, precision: 'fp16', delay: 1.35,
        model: { status: 'ready', progress: 1 }, check: { person: true, feet: true, joints: 24, edge: false },
        job: { kind: exp ? 'file' : 'window', name: 'dance practice.mp4' }, osc: { host: '127.0.0.1', port: 9000, sent: Math.round(t * 60) }, temporal: { ms: 18.2, used: 970, passed: 30, device: 'dml' },
        preview: frame(t),
      };
    },
  };
})();
