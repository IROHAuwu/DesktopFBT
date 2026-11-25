"""影片放在中文、日文路徑時的暫存複本：用完一定刪掉，只刪自己建立的（tools/dev/video_tempcopy_contract.md）。
[注入] = 用替換函式強制「沒有 ASCII 短檔名」或模擬複製 / 刪除失敗；影片檔本身是真的、放在真的日文資料夾、真的讀出畫面。
不需要 GPU、模型、網路或硬體；輸出關閉（不送任何網路封包）。
執行：python tests/test_video_tempcopy.py
"""
import hashlib
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import test_video as TV  # noqa: E402  （匯入時：暫存資料夾、假 3D 模型、假頭顯；不執行它的 main）
E, V = TV.E, TV.V

FAILED = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAILED.append(msg)


CREATED = []                                   # video.py 這次建立的暫存資料夾（不靠名稱猜）
_real_mkdtemp = tempfile.mkdtemp


def _recording_mkdtemp(*a, **k):
    d = _real_mkdtemp(*a, **k)
    CREATED.append(d)
    return d


V.tempfile.mkdtemp = _recording_mkdtemp
V._short_ascii = lambda path: None             # [注入] 這台電腦沒有 ASCII 短檔名（強制走「複製」）


def left():
    return [d for d in CREATED if os.path.exists(d)]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    cjk_dir = os.path.join(TV.TMP, '日本語 動画')
    os.makedirs(cjk_dir)
    ascii_video = os.path.join(TV.TMP, 'dance.mp4')
    TV.make_video(ascii_video)
    video = os.path.join(cjk_dir, 'テスト 影片.mp4')
    shutil.copyfile(ascii_video, video)
    h0 = sha(video)

    # --- 函式層
    CREATED.clear()
    v = V.VideoFile(video, loop=False)
    copied = len(CREATED) == 1 and os.path.isfile(os.path.join(CREATED[0], 'video.mp4'))
    frames = sum(1 for _ in range(10) if v.read_next() is not None)
    v.stop()
    check(copied and frames == 10 and not left(), f'[注入] 日文路徑、沒有短檔名 → 複製後讀到 {frames} 幀，stop() 之後暫存資料夾已刪除')
    v.stop()
    check(not left() and sha(video) == h0, 'stop() 重複呼叫沒問題；原本的影片檔沒有改變')

    CREATED.clear()
    real_copy = V.shutil.copyfile

    def half_copy(src, dst):
        with open(src, 'rb') as a, open(dst, 'wb') as b:
            b.write(a.read(1000))
        raise OSError(28, 'No space left on device (injected)')
    V.shutil.copyfile = half_copy
    try:
        V.VideoFile(video)
        err = None
    except OSError as e:
        err = e
    finally:
        V.shutil.copyfile = real_copy
    check(err is not None and len(CREATED) == 1 and not left(), '[注入] 複製到一半失敗 → 錯誤照樣傳出，沒有留下半個檔案或資料夾')

    CREATED.clear()
    not_video = os.path.join(cjk_dir, '壊れた.mp4')
    open(not_video, 'wb').write(b'not a video')
    try:
        V.VideoFile(not_video)
        err = None
    except RuntimeError as e:
        err = str(e)
    check(err == 'ERR_VIDEO_OPEN' and len(CREATED) == 1 and not left(), '打不開的檔案 → ERR_VIDEO_OPEN，複本資料夾已刪除')

    CREATED.clear()
    V.take_warnings()
    v = V.VideoFile(video)
    real_rmtree = V.shutil.rmtree

    def refuse(path, *a, **k):
        raise PermissionError(13, 'Access is denied (injected)')
    V.shutil.rmtree = refuse
    try:
        v.stop()
        raised = False
    except Exception:
        raised = True
    finally:
        V.shutil.rmtree = real_rmtree
    w = V.take_warnings()
    check(not raised and len(w) == 1 and len(left()) == 1, f'[注入] 刪不掉 → stop() 不丟出例外，留下 1 筆提醒（{w[0][:30] if w else ""}…）')
    for d in left():
        shutil.rmtree(d)

    foreign_tmp = _real_mkdtemp(prefix='desktopfbt_')              # 不是這個物件建立的
    foreign_elsewhere = os.path.join(TV.TMP, 'desktopfbt_not_temp')
    os.makedirs(foreign_elsewhere)
    V._remove_owned(foreign_elsewhere)
    w = V.take_warnings()
    check(os.path.isdir(foreign_elsewhere) and len(w) == 1, '不在暫存資料夾正下方的 desktopfbt_ 資料夾 → 不刪，只記提醒')

    CREATED.clear()
    v = V.VideoFile(ascii_video)
    v.stop()
    check(CREATED == [], 'ASCII 路徑 → 照舊直接開啟，沒有複製')

    # --- 引擎（真的引擎流程，假 3D 模型，輸出關閉）
    eng = E.Engine(preload=False)
    eng.update_config({'video': {'hfov_deg': TV.HFOV, 'height_m': TV.M.HEIGHT, 'loop': True},
                       'pose': {'device': 'cpu'}})      # 不要等 Windows 查 AMD 加速（CI 上要 4 分鐘才逾時）

    def run_live(stop_after):
        CREATED.clear()
        eng.start_live({'kind': 'file', 'path': video})
        time.sleep(stop_after)
        st = eng.get_state(False)
        eng.stop()
        return st, TV.wait_idle(eng, 10)

    st, idle = run_live(3.0)
    check(st['mode'] == 'live' and st['error'] is None and idle and len(CREATED) == 1 and not left(),
          '引擎：影片即時模式（日文路徑）→ 使用者按停止 → 暫存複本已刪除')

    import track                                   # 即時模式每一幀的計算在 track.Tracker 裡
    real_lifter = track.Lifter

    def broken_lifter(*a, **k):
        raise RuntimeError('ERR_TEST_SETUP')
    track.Lifter = broken_lifter
    try:
        CREATED.clear()
        eng.start_live({'kind': 'file', 'path': video})
        idle = TV.wait_idle(eng, 10)
        st = eng.get_state(False)
    finally:
        track.Lifter = real_lifter
    check(idle and 'ERR_TEST_SETUP' in str(st['error']) and len(CREATED) == 1 and not left(),
          f'[注入] 引擎：影片即時模式準備時出錯（{st["error"]}）→ 暫存複本已刪除')

    CREATED.clear()
    eng.start_export(video)
    idle = TV.wait_idle(eng, 240)
    st = eng.get_state(False)
    check(idle and st['error'] is None and st['job'] and st['job'].get('export') and len(CREATED) == 1 and not left(),
          '引擎：匯出（日文路徑）完成 → 暫存複本已刪除')

    CREATED.clear()
    eng.start_export(video)
    time.sleep(0.5)
    eng.stop()
    idle = TV.wait_idle(eng, 30)
    check(idle and len(CREATED) == 1 and not left(), '引擎：匯出中按停止 → 暫存複本已刪除')
    eng.shutdown()

    check(os.path.isdir(foreign_tmp) and sha(video) == h0, '別人的 desktopfbt_ 資料夾沒有被動到；原本的影片檔沒有改變')
    shutil.rmtree(foreign_tmp, ignore_errors=True)
    shutil.rmtree(TV.TMP, ignore_errors=True)
    if FAILED:
        print(f'失敗 {len(FAILED)} 項')
        sys.exit(1)
    print('全部通過')


if __name__ == '__main__':
    main()
