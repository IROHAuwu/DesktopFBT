"""程式資料夾不能寫入時的啟動處理（startup_check.py + app.main / write_crash）。
Windows：用真的 ACL（icacls）在暫存資料夾測試，結束時一定恢復權限並刪除（只動 TEMP 底下的測試資料夾）。
啟動前檢查（擋下）：資料夾不能建立檔案 / 資料夾、既有 config.json / calibration.json 不能寫入。
不擋下（之後的操作才處理）：唯讀但模型已在的 models\\、certs\\、磁碟滿。另有標示為「注入」的崩潰紀錄測試。
不開網路埠、不開視窗。執行：python tests/test_writable_startup.py
"""
import ctypes
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import startup_check as S  # noqa: E402

TEMP = os.path.realpath(tempfile.gettempdir())
USER = os.environ.get('USERNAME', '')


def under_temp(p):
    rp = os.path.realpath(p)
    assert rp.startswith(TEMP + os.sep), f'refusing to touch a path outside TEMP: {rp}'
    return rp


def icacls(*args):
    return subprocess.run(['icacls', *args], capture_output=True)


def deny(p, rights='(OI)(CI)(WD,AD)'):
    under_temp(p)
    r = icacls(p, '/deny', f'{USER}:{rights}')
    assert r.returncode == 0, r


def allow(p):
    icacls(p, '/remove:d', USER)


def tree(d):
    return sorted(os.path.relpath(os.path.join(a, n), d) for a, ds, fs in os.walk(d) for n in ds + fs)


def sys_lang():
    return {0x04: 'zh', 0x11: 'ja'}.get(ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF, 'en')


def main():
    root = under_temp(tempfile.mkdtemp(prefix='desktopfbt_wtest_'))
    denied = []
    try:
        # --- 可以寫入的資料夾（一般 / Unicode + 空格）：沒有問題、沒有留下檔案、不存在的檔案仍不存在
        for name in ('plain', 'DesktopFBT 測試 テスト'):
            d = os.path.join(root, name)
            os.makedirs(os.path.join(d, 'models'))
            before = tree(d)
            p, w = S.check(d)
            assert p is None and w == [], (name, p, w)
            assert tree(d) == before and not os.path.exists(os.path.join(d, 'config.json')), f'side effect ({name})'
        print('  ok   writable (plain, Unicode + space): pass, no leftover probe, absent files stay absent')
        if sys.platform != 'win32':
            print('  skip ACL cases (not Windows)')
            return

        # --- 啟動前擋下 1：資料夾不能建立檔案 / 資料夾
        d = os.path.join(root, 'deny_folder')
        os.makedirs(os.path.join(d, 'models'))
        before = tree(d)
        deny(d); denied.append(d)
        p, w = S.check(d)
        assert p and p['kind'] == 'folder' and os.path.realpath(p['path']) == os.path.realpath(d), p
        assert isinstance(p['error'], PermissionError) and w == [], (p, w)
        allow(d); denied.remove(d)
        assert tree(d) == before, 'something was created in the denied folder'
        print('  ok   STARTUP-PREFLIGHT folder deny -> refuse (kind folder), nothing created')

        # --- 啟動前擋下 2：既有 config.json 是唯讀（內容與時間不變）；calibration.json 同樣
        for fname in ('config.json', 'calibration.json'):
            d = os.path.join(root, 'ro_' + fname.split('.')[0])
            os.makedirs(d)
            fp = os.path.join(d, fname)
            json.dump({'language': 'en'}, open(fp, 'w', encoding='utf-8'))
            b0, m0 = open(fp, 'rb').read(), os.stat(fp).st_mtime_ns
            os.chmod(fp, stat.S_IREAD)
            time.sleep(0.05)
            p, w = S.check(d)
            assert p and p['kind'] == 'file' and os.path.basename(p['path']) == fname, p
            assert open(fp, 'rb').read() == b0 and os.stat(fp).st_mtime_ns == m0, f'{fname} modified'
            os.chmod(fp, stat.S_IREAD | stat.S_IWRITE)
            p2, _ = S.check(d)
            assert p2 is None, p2
        print('  ok   STARTUP-PREFLIGHT read-only config.json / calibration.json -> refuse (kind file), bytes and mtime unchanged')

        # --- 不擋下：models\ 不能寫入但模型已在（唯讀的模型快取照常使用）
        d = os.path.join(root, 'ro_model_cache')
        md = os.path.join(d, 'models')
        os.makedirs(md)
        open(os.path.join(md, 'model.onnx'), 'wb').write(b'\0' * 1024)
        deny(md); denied.append(md)
        assert not os.access(md, os.W_OK) or True
        p, w = S.check(d)
        assert p is None, p
        assert open(os.path.join(md, 'model.onnx'), 'rb').read() == b'\0' * 1024
        allow(md); denied.remove(md)
        print('  ok   LATER-OPERATION models/ not writable with cache present -> starts (read-only warm cache kept)')
        # certs\ 不能寫入也不在啟動時擋下
        d = os.path.join(root, 'ro_certs')
        os.makedirs(os.path.join(d, 'certs'))
        deny(os.path.join(d, 'certs')); denied.append(os.path.join(d, 'certs'))
        p, _ = S.check(d)
        assert p is None, p
        allow(os.path.join(d, 'certs')); denied.remove(os.path.join(d, 'certs'))
        print('  ok   LATER-OPERATION certs/ not writable -> not refused at startup (phone mode reports it later; open matrix row)')

        # --- 可以建立但不能刪除（另一種 ACL）：只回報警告與留下的路徑，不擋下、不重試
        d = os.path.join(root, 'no_delete')
        os.makedirs(d)
        deny(d, '(OI)(CI)(DE,DC)'); denied.append(d)
        t0 = time.perf_counter()
        p, w = S.check(d)
        assert p is None and len(w) >= 1 and all('.desktopfbt-write-test-' in x for x, _ in w), (p, w)
        assert time.perf_counter() - t0 < 2.0
        allow(d); denied.remove(d)
        for x, _ in w:                                           # 測試留下的探針只在已確認的 TEMP 測試資料夾裡清掉
            under_temp(x)
            (os.rmdir if os.path.isdir(x) else os.remove)(x)
        print(f'  ok   create-allowed / delete-denied ACL -> declared WARNING ({len(w)} leftover probe(s) reported), not refused, no retry loop')

        # --- 語言：存過的語言 → Windows 介面語言 → 英文；壞掉的 / 非 UTF-8 設定檔也不會出錯
        d = os.path.join(root, 'lang')
        os.makedirs(d)
        cp = os.path.join(d, 'config.json')
        for lang in ('zh', 'ja', 'en'):
            json.dump({'language': lang}, open(cp, 'w', encoding='utf-8'))
            assert S.language(d) == lang
        open(cp, 'w', encoding='utf-8').write('{broken')
        assert S.language(d) == sys_lang()
        open(cp, 'wb').write(b'\x82\xa0\x82\xa2 not utf-8 {')
        assert S.language(d) == sys_lang() and S.language(os.path.join(root, 'missing')) == sys_lang()
        long_path = 'C:\\' + '\\'.join(['很長的資料夾名稱 long folder'] * 30)
        for kind in ('folder', 'file'):
            for lang in ('zh', 'ja', 'en'):
                json.dump({'language': lang}, open(cp, 'w', encoding='utf-8'))
                t = S.message(dict(kind=kind, path=r'C:\x\DesktopFBT\config.json'), d)
                assert r'C:\x\DesktopFBT\config.json' in t
                t2 = S.message(dict(kind=kind, path=long_path), d)
                assert len(t2) < len(S.MESSAGES[kind][lang]) + 300 and ' … ' in t2
        assert 'Program Files' in S.MESSAGES['folder']['en'] and 'Read-only' in S.MESSAGES['file']['en']
        print(f'  ok   language: saved > Windows UI ({sys_lang()}) > en, corrupt / non-UTF-8 config tolerated; '
              'folder / file messages name the failing path; long paths shortened')

        # --- app.main()：--selftest 在不能寫入的資料夾直接以 2 結束，不跳視窗、不啟動執行緒、不先匯入 engine / webview
        import paths
        old_base = paths.base_dir
        d = os.path.join(root, 'main_deny')
        os.makedirs(d)
        before = tree(d)
        deny(d); denied.append(d)
        paths.base_dir = lambda: d
        sys.argv = ['app.py', '--selftest']
        import app
        n0 = threading.active_count()
        t0 = time.perf_counter()
        try:
            app.main()
            raise AssertionError('main() returned on an unwritable folder')
        except SystemExit as e:
            assert e.code == S.EXIT_NOT_WRITABLE, e.code
        dt = time.perf_counter() - t0
        assert threading.active_count() == n0 and dt < 5.0
        assert 'engine' not in sys.modules and 'webview' not in sys.modules, 'imported before the check'
        allow(d); denied.remove(d)
        assert tree(d) == before, 'main() created something before refusing'
        print(f'  ok   app.main --selftest unwritable: exit {S.EXIT_NOT_WRITABLE} in {dt:.2f} s, no dialog, no threads, '
              'engine/webview not imported, nothing created')
        # --- 注入：崩潰紀錄寫不進資料夾時改寫到暫存資料夾，本身不出錯
        deny(d); denied.append(d)
        p = app.write_crash('injected traceback for test_writable_startup')
        fb = os.path.join(tempfile.gettempdir(), 'DesktopFBT_crash.txt')
        assert p and os.path.realpath(p) == os.path.realpath(fb) and 'injected traceback' in open(p, encoding='utf-8').read()
        os.remove(p)
        allow(d); denied.remove(d)
        p = app.write_crash('x')
        assert os.path.realpath(p) == os.path.realpath(os.path.join(d, 'crash.txt'))
        paths.base_dir = old_base
        print('  ok   [injected] crash.txt falls back to %TEMP%\\DesktopFBT_crash.txt when the data folder is unwritable')

        # --- 啟動檢查的警告（刪不掉的測試檔）經過真正的 main() 被記錄下來，不被丟掉
        got = []
        real_selftest = app.selftest
        app.selftest = lambda w=None: got.append(list(w or [])) or True     # selftest 回傳 ok（main 依此決定結束碼）
        try:
            # 真的 ACL：可以建立、不能刪除
            d = os.path.join(root, 'warn_real')
            os.makedirs(d)
            cp = os.path.join(d, 'config.json')
            json.dump({'language': 'en'}, open(cp, 'w', encoding='utf-8'))
            b0 = open(cp, 'rb').read()
            deny(d, '(OI)(CI)(DE,DC)'); denied.append(d)
            paths.base_dir = lambda: d
            sys.argv = ['app.py', '--selftest']
            app.main()
            assert got and len(got[-1]) >= 1 and all('.desktopfbt-write-test-' in str(x[0]) for x in got[-1]), got
            log = open(os.path.join(d, 'log.txt'), encoding='utf-8').read()
            assert log.count('.desktopfbt-write-test-') == len(got[-1]), log
            assert open(cp, 'rb').read() == b0, 'config.json changed'
            allow(d); denied.remove(d)
            for x, _ in got[-1]:
                under_temp(x)
                (os.rmdir if os.path.isdir(x) else os.remove)(x)
            print(f'  ok   real ACL (create allowed, delete denied) through app.main: {len(got[-1])} warning(s) logged to log.txt and '
                  'passed to selftest; config.json unchanged; startup not refused')
            # 注入：有警告但 log.txt 寫不進去 → 不出錯、照常啟動、警告仍傳給 selftest
            d2 = os.path.join(root, 'warn_injected')
            os.makedirs(d2)
            deny(d2); denied.append(d2)
            real_check = S.check
            fake = os.path.join(d2, '.desktopfbt-write-test-injected')
            S.check = lambda base: (None, [(fake, PermissionError('injected'))])
            try:
                paths.base_dir = lambda: d2
                got.clear()
                app.main()
                assert len(got) == 1 and len(got[0]) == 1 and got[0][0][0] == fake, got
            finally:
                S.check = real_check
            allow(d2); denied.remove(d2)
            assert not os.path.exists(os.path.join(d2, 'log.txt'))
            print('  ok   [injected] warning with an unwritable log.txt: no exception, startup continues, warning still passed on')
        finally:
            app.selftest = real_selftest
            paths.base_dir = old_base
    finally:
        for p in denied:
            allow(p)
        shutil.rmtree(under_temp(root), ignore_errors=True)
    print('全部通過')


if __name__ == '__main__':
    main()
