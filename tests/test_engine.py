"""引擎：設定（舊版設定檔、存檔失敗）、工作開始／停止、錯誤回報。不需要模型。執行：python tests/test_engine.py"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp()
import paths  # noqa: E402
paths.base_dir = lambda: TMP

import engine as E  # noqa: E402
import safe_write  # noqa: E402

CFG = os.path.join(TMP, 'config.json')


def wait_idle(eng, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        if eng.get_state(False)['mode'] == 'idle':
            return True
        time.sleep(0.05)
    return False


def test_old_config():
    """舊版（相機追蹤）的設定檔：認得的保留，其他丟掉；壞掉的值不會讓程式出錯。"""
    old = {'language': 'ja', 'cameras': [{'source': 'phone:A'}], 'camera_count': 2,
           'pose': {'mode': 'balanced', 'device': 'cpu', 'det_frequency': 1, 'score_thr': 0.5},
           'vrchat': {'host': '10.0.0.2', 'port': 9001, 'tracker_ids': {'hip': 1}},
           'filter': {'smoothness': 0.3}, 'video': {'hfov_deg': 70, 'desktop': True, 'unity_project': 'X'},
           'live': 'broken'}
    with open(CFG, 'w', encoding='utf-8') as f:
        json.dump(old, f)
    eng = E.Engine()
    c = eng.get_config()
    assert c['language'] == 'ja' and c['pose']['device'] == 'cpu' and c['pose']['score_thr'] == 0.5
    assert c['pose']['det_frequency'] == E.DEFAULT_CONFIG['pose']['det_frequency']     # 內部參數用新版
    assert c['vrchat'] == {'host': '10.0.0.2', 'port': 9001}
    assert c['video']['hfov_deg'] == 70 and c['video']['unity_project'] == 'X' and 'desktop' not in c['video']
    assert 'cameras' not in c and 'filter' not in c and c['live'] == E.DEFAULT_CONFIG['live']
    eng.update_config({'live': {'delay': 1.1}, 'cameras': [], 'pose': {'bogus': 1}})
    saved = json.load(open(CFG, encoding='utf-8'))
    assert saved['live'] == {'delay': 1.1} and 'cameras' not in saved and 'bogus' not in saved.get('pose', {})
    assert E.Engine().get_config()['live']['delay'] == 1.1


def test_removed_modes():
    """v1.6 的 VR 模式、雙相機設定：整段丟掉；VR 模式自動關掉的時間序列 3D 回到預設。"""
    keep = open(CFG, 'rb').read()
    old = {'live': {'output': 'vr', 'temporal': False, 'precise': False, 'smoothness': 0.4},
           'vr': {'trackers': ['hip'], 'align': True}, 'stereo': {'camera:0:1280x720|camera:1:1280x720': {'err_px': 2.0}}}
    with open(CFG, 'w', encoding='utf-8') as f:
        json.dump(old, f)
    c = E.Engine(preload=False).get_config()
    assert 'vr' not in c and 'stereo' not in c and 'output' not in c['live']
    assert c['live']['temporal'] is True and c['live']['smoothness'] == 0.4
    with open(CFG, 'w', encoding='utf-8') as f:
        json.dump({'live': {'temporal': False}}, f)
    assert E.Engine(preload=False).get_config()['live']['temporal'] is False      # 自己關掉的照舊
    with open(CFG, 'wb') as f:
        f.write(keep)


def test_save_failure():
    eng = E.Engine()
    before = open(CFG, 'rb').read()
    real = safe_write._write_text

    def boom(fd, text):
        os.close(fd)
        raise OSError(28, 'No space left on device')
    safe_write._write_text = boom
    try:
        try:
            eng.update_config({'live': {'delay': 0.5}})
            raise AssertionError('應該失敗')
        except RuntimeError as e:
            assert str(e).startswith('ERR_SAVE_CONFIG:config.json'), e
    finally:
        safe_write._write_text = real
    assert open(CFG, 'rb').read() == before and eng.get_config()['live']['delay'] == 1.1
    assert not [f for f in os.listdir(TMP) if f.endswith('.tmp')]


def test_jobs():
    """沒有模型也能測：來源錯誤要回報；停止要能結束。"""
    eng = E.Engine()
    E.models.pose3d_ready = lambda: True
    E.models.ensure_pose3d = lambda progress=None: {}

    class Est:
        precision, device, ms, freq, tta = 'fp32', 'cpu', 1.0, 8, False

        def __init__(self, *a):
            pass
    E.Pose3DEstimator = Est
    eng.start_live({'kind': 'file', 'path': os.path.join(TMP, 'none.mp4')})
    assert wait_idle(eng)
    assert eng.get_state(False)['error'] == 'ERR_VIDEO_OPEN'
    eng.start_image_export([os.path.join(TMP, 'none.png')])
    assert wait_idle(eng)
    assert eng.get_state(False)['error'] == 'ERR_IMAGE_OPEN'
    eng.start_live({'kind': 'nothing'})
    assert wait_idle(eng) and eng.get_state(False)['error'] == 'ERR_VIDEO_OPEN'
    st = eng.get_state()
    assert st['version'] == E.VERSION and 'preview' in st
    env = eng.environment()
    assert env['version'] == E.VERSION and 'cpu' in env['devices']
    eng.shutdown()


def test_slow_stop():
    """上一個工作停得很慢（例如相機還在開）：新的工作照常開始，舊的醒來後自己結束，不會復活、也不會蓋掉新工作的狀態。"""
    import threading
    eng = E.Engine(preload=False)
    real_join = threading.Thread.join
    release, seen = threading.Event(), []

    def slow():
        release.wait(10)                      # 卡在某個不能中斷的等待裡
        seen.append('woke')
        eng._check_stop()                     # 醒來後的第一個檢查點：要發現自己已經被停止
        seen.append('revived')

    def quick():
        while True:
            eng._check_stop()
            time.sleep(0.02)
    threading.Thread.join = lambda self, timeout=None: real_join(self, 0.2 if timeout == 5.0 else timeout)
    try:
        eng._start('live', slow)
        old = eng._worker
        eng._start('export', quick)           # stop() 等不到舊的結束
        assert old.is_alive() and eng.get_state(False)['mode'] == 'export'
        release.set()
        real_join(old, 5)
        assert seen == ['woke'], seen
        assert eng.get_state(False)['mode'] == 'export'          # 舊工作結束時沒有把狀態改成待機
        eng.stop()
        assert wait_idle(eng)
    finally:
        threading.Thread.join = real_join
        release.set()
    eng.shutdown()


def test_accel_wait():
    """第一次準備 AMD 加速要等：介面要顯示「準備中」（以前什麼都不顯示），等的時候按停止要馬上有反應。"""
    import threading
    import accel
    eng = E.Engine(preload=False)
    eng.cfg['pose']['device'] = 'auto'
    gate = threading.Event()
    real = accel.ensure, accel.status, dict(accel._state)

    def slow_ensure(timeout=240):
        gate.wait(20)
        accel._state.update(done=True, ok=False, error='test')
        return False
    accel._state.update(done=False, ok=False, error=None)
    accel.ensure = slow_ensure
    try:
        eng._start('live', eng._accel)
        time.sleep(0.4)
        assert eng.get_state(False)['phase'] == 'preparing_accel'
        w = eng._worker
        t = time.time()
        eng.stop()
        assert time.time() - t < 2.0 and not w.is_alive()          # 不用等查詢結束
        assert eng.get_state(False)['mode'] == 'idle'
    finally:
        gate.set()
        accel.ensure, accel.status = real[0], real[1]
        accel._state.clear()
        accel._state.update(real[2])
    assert accel.status()['done'] is False or accel.status()['error']    # 做完才會是 done
    eng.shutdown()


def test_gpu_vendor():
    """沒有 AMD 顯示卡（NVIDIA、Intel）就不去查 MIGraphX：不會跳出安裝提示、不用等；查不出顯示卡時照舊去查。"""
    import accel
    assert accel.wanted([('nvidia', 'NVIDIA GeForce RTX 4070')]) is False
    assert accel.wanted([('intel', 'Intel(R) Arc(TM) A750 Graphics')]) is False
    assert accel.wanted([('nvidia', 'NVIDIA GeForce RTX 4060 Laptop GPU'), ('amd', 'AMD Radeon(TM) 780M')]) is True
    assert accel.wanted([('amd', 'AMD Radeon RX 7900 XT')]) is True
    assert accel.wanted([]) is True
    found = accel.gpus()                          # 真的查一次（Windows 的 CI 上也跑）：不能出錯，格式要對
    assert isinstance(found, list) and all(v in ('amd', 'nvidia', 'intel', 'qualcomm', 'other') and isinstance(n, str) for v, n in found)
    print('  這台電腦的顯示卡：', found or '（查不到）')
    if sys.platform == 'win32':
        real_run, real_gpus, state = accel._run_helper, accel.gpus, dict(accel._state)
        called = []
        accel._run_helper = lambda timeout: called.append(1) or {'providers': {}, 'error': 'test'}
        try:
            accel.gpus = lambda: [('nvidia', 'NVIDIA GeForce RTX 4070')]
            accel._state.update(done=False, ok=False, error=None)
            assert accel.ensure() is False and not called and 'no AMD GPU' in accel.status()['error']
            accel.gpus = lambda: [('amd', 'AMD Radeon RX 7900 XT')]
            accel._state.update(done=False, ok=False, error=None)
            assert accel.ensure() is False and called
        finally:
            accel._run_helper, accel.gpus = real_run, real_gpus
            accel._state.clear()
            accel._state.update(state)


def main():
    for f in (test_old_config, test_removed_modes, test_save_failure, test_jobs, test_slow_stop, test_accel_wait,
              test_gpu_vendor):
        f()
        print('ok', f.__name__)
    print('全部通過')


if __name__ == '__main__':
    main()
