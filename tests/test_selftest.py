"""--selftest：走真的 app.main()（啟動檢查 → selftest → 結束碼），引擎換成假的。執行：python tests/test_selftest.py"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
BASE = tempfile.mkdtemp(prefix='desktopfbt_selftest_')
import paths  # noqa: E402
paths.base_dir = lambda: BASE
import app  # noqa: E402

FAILED = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAILED.append(msg)


class FakeEngine:
    shutdowns = 0
    fail_init = False

    def __init__(self, preload=True):
        if FakeEngine.fail_init:
            raise RuntimeError('engine could not start (test)')

    def environment(self):
        return {'version': 'test'}

    def list_windows(self):
        return []

    def shutdown(self):
        FakeEngine.shutdowns += 1


sys.modules['engine'] = types.SimpleNamespace(Engine=FakeEngine)


def run_main():
    FakeEngine.shutdowns = 0
    report = os.path.join(BASE, 'selftest.txt')
    if os.path.exists(report):
        os.remove(report)
    sys.argv = ['DesktopFBT.exe', '--selftest']
    code = 'returned'
    try:
        app.main()
    except SystemExit as e:
        code = e.code
    rep = json.load(open(report, encoding='utf-8')) if os.path.exists(report) else None
    return code, rep


def main():
    code, rep = run_main()
    want = ['ok', 'env', 'providers', 'opencv', 'web', 'windows', 'unity_script', 'fp16', 'dxcam', 'temporal', 'winml', 'startup_warnings']
    check(rep is not None and set(want) | {'unblocked'} <= set(rep) <= set(want) | {'temporal_models', 'unblocked', 'dotnet'} and FakeEngine.shutdowns == 1, f'報告欄位完整、引擎已關閉（{code}）')
    try:
        import onnxconverter_common  # noqa: F401
        have16 = True
    except ImportError:
        have16 = False
    check((rep['ok'] is True and code == 'returned') if have16 else (rep['ok'] is False and code == app.EXIT_SELFTEST_FAILED),
          '半精度轉換可用 → ok；不能用 → 結束碼 3')
    FakeEngine.fail_init = True
    try:
        run_main()
        raised = False
    except RuntimeError:
        raised = True
    finally:
        FakeEngine.fail_init = False
    check(raised, '沒預料到的錯誤（引擎開不起來）照樣丟出')
    if FAILED:
        print(f'失敗 {len(FAILED)} 項')
        sys.exit(1)
    print('全部通過')


if __name__ == '__main__':
    main()
