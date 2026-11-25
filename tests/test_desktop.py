"""桌面模式：肌肉值反解、參數範圍、Unity 工具與 Python 的參數表一致。"""
import os
import re
import sys
import tarfile
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import desktop as DK  # noqa: E402
import motion  # noqa: E402
import pose as P  # noqa: E402
import unity_pkg  # noqa: E402


def forward(name, mus):
    """Unity 的 ZYRoll 模型（已在 Unity 用 HumanPoseHandler 驗證）。"""
    pre, post, sign, dofs = DK.MODEL[name]
    a = np.zeros(3)
    for d in range(3):
        if dofs[d] is None:
            continue
        i, mn, mx = dofs[d]
        v = mus.get(i, 0.0)
        a[d] = v * mx if v >= 0 else -v * mn
    tx, ty, tz = np.tan(np.radians(a) * np.array(sign) / 2)
    q = np.array([tx, ty + tx * tz, tz - tx * ty, 1.0])
    q /= np.linalg.norm(q)
    return DK.qmul(DK.qmul(np.asarray(pre, float), q), DK.qinv(np.asarray(post, float)))


def test_roundtrip():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        mus = {i: float(rng.uniform(-0.9, 0.9)) for i in DK.MUSCLES}
        for n in DK.ORDER:
            got = DK.muscles_of(n, forward(n, mus))
            for k, v in got.items():
                worst = max(worst, abs(v - mus[k]))
    assert worst < 1e-5, worst


def test_params():
    d = DK.DesktopPose()
    ok = np.ones(P.NUM_JOINTS, bool)
    n = 0
    for i in range(120):
        X = np.array(motion.person(i / 10.0), float)
        out = d(X, ok)
        assert out is not None
        assert len(out) == len(DK.MUSCLES) + len(DK.BODY_PARAMS) == 49
        assert abs(out['YawS'] ** 2 + out['YawC'] ** 2 - 1) < 1e-6
        assert all(-1.0 <= v <= 1.0 and np.isfinite(v) for v in out.values()), out
        n += 1
    assert n == 120
    bad = np.ones(P.NUM_JOINTS, bool)
    bad[P.L_KNEE] = False
    assert d(X, bad) is None


def test_hands():
    import bvh
    HL = np.array([[0, 0, 0]] + [[0.03 + 0.01 * i, 0, 0.03] for i in range(4)] +
                  [[0.08 + 0.02 * i, 0, 0.025] for i in range(4)] + [[0.085 + 0.02 * i, 0, 0.005] for i in range(4)] +
                  [[0.08 + 0.02 * i, 0, -0.015] for i in range(4)] + [[0.07 + 0.01 * i, 0, -0.03] for i in range(4)])
    for side, sgn in (('Left', 1), ('Right', -1)):
        R = DK.hand_rotation(HL * np.array([sgn, 1, 1]), side)
        assert np.allclose(R, np.eye(3), atol=0.06), (side, R)
    # 手腕往上彎 35° → 手的 Down-Up 肌肉 = 35 / 80
    X = np.array(motion.person(0.0), float)
    G = bvh.global_rotations(X)
    a = np.radians(35)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    hands = {'Left': X[P.L_WRI] + HL @ (G['LeftForeArm'] @ Rz).T}
    out = DK.DesktopPose()(X, np.ones(P.NUM_JOINTS, bool), hands)
    assert abs(abs(out['m44']) - 35 / 80) < 0.03, out['m44']


def test_euler():
    for e in ((25, 40, -70), (-80, 170, 10), (0, -120, 45)):
        x, y, z = np.radians(e)
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        got = DK.unity_euler(DK.mat_quat(Ry @ Rx @ Rz))
        assert np.allclose(got, e, atol=1e-6), (got, e)


def test_body_tilt():
    """身體旋轉的參數：可以還原，而且彎腰經過 90° 時連續（尤拉角在這裡會亂跳）。"""
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        y, t = DK.body_tilt(q)
        if np.linalg.norm(t) < 0.82:
            q2 = DK.tilt_quat(y, t)
            assert min(np.linalg.norm(q - q2), np.linalg.norm(q + q2)) < 1e-6
    d = DK.DesktopPose()
    ok = np.ones(P.NUM_JOINTS, bool)
    prev = None
    for i in range(240):
        J, tw, bend = motion.bend_person(i / 20.0)
        out = d(J, ok)
        p = np.array([out[k] for k in ('YawS', 'YawC', 'TiltX', 'TiltZ')])
        if prev is not None:
            assert np.abs(p - prev).max() < 0.12, (i, p, prev)
        prev = p
        if bend > 1.5:
            assert out['TiltX'] > 0.35, out['TiltX']            # 往前彎 = TiltX 正


def test_stale_avatar():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert DK.stale_avatar(d) is None
        av = os.path.join(d, 'usr_x', 'Avatars')
        os.makedirs(av)
        def write(fn, name, params):
            with open(os.path.join(av, fn), 'w', encoding='utf-8-sig') as f:
                json.dump({'name': name, 'parameters': [{'name': p} for p in params]}, f)
        write('a.json', 'Other', ['VRCEmote'])
        assert DK.stale_avatar(d) is None
        write('b.json', 'Old', ['P' + 'FBT/YawS', 'P' + 'FBT/TiltX'])
        assert DK.stale_avatar(d) == 'Old'
        os.utime(os.path.join(av, 'b.json'), (1, 1))
        write('c.json', 'New', ['YFBT/YawS', 'YFBT/TiltX'])
        assert DK.stale_avatar(d) is None


def test_unity_tables():
    src = open(os.path.join(ROOT, 'unity', 'Desktop', 'DesktopFBTDesktop.cs'), encoding='utf-8').read()
    nums = lambda name: [int(x) for x in re.search(name + r'\s*=\s*\{([^}]*)\}', src).group(1).replace(' ', '').split(',') if x]  # noqa: E731
    assert nums('Muscles') == DK.MUSCLES
    assert nums('SyncLite') == DK.SYNC_LITE
    body = re.search(r'Body\s*=\s*\{([^}]*)\}', src).group(1)
    assert [b.strip().strip('"') for b in body.split(',')] == list(DK.BODY_PARAMS)
    assert f'"{DK.PREFIX}"' in src
    assert f'MaxBits = {DK.MAX_BITS}' in src                # 同步的分配表在 tests/test_sync.py 比對


def test_bits():
    for n in (4, 5, 6):
        for v in np.linspace(-1, 1, 101):
            b = DK.encode(v)[:n]                  # Avatar 只收前 n 個
            assert abs(DK.decode(b) - v) <= 2.0 ** -n + 1e-9, (n, v)


def test_param_compressor_port():
    """同梱的 Param Compressor（從 lilMCR 搬來）：沒有殘留 lilMCR 的類別、只在有 NDMF 時編譯、工具用得到的欄位和方法都在。"""
    base = os.path.join(ROOT, 'unity', 'ParamCompressor')
    comp = open(os.path.join(base, 'Runtime', 'DesktopFBTParamCompressor.cs'), encoding='utf-8').read()
    run = open(os.path.join(base, 'Editor', 'DesktopFBTParamCompressorPass.cs'), encoding='utf-8').read()
    tool = open(os.path.join(ROOT, 'unity', 'Desktop', 'DesktopFBTDesktop.cs'), encoding='utf-8').read()
    assert 'namespace DesktopFBT' in comp and 'class DesktopFBTParamCompressor : MonoBehaviour, IEditorOnly' in comp
    for field in ('public int maxSyncSteps', 'public bool excludeFloats', 'public string[] excludedPrefixes'):
        assert field in comp, field                                     # 工具用反射讀寫這幾個欄位
    code = '\n'.join(l for l in run.splitlines() if not l.lstrip().startswith('//'))
    assert 'Macaron' not in code and 'MacaronCore' not in run and 'ML.' not in code
    assert code.count('liltoon_MCR') == 1 and '.AfterPlugin("liltoon_MCR.param-compressor")' in code      # 只剩「排在 lilMCR 的後面」
    assert '[assembly: ExportsPlugin(typeof(DesktopFBT.DesktopFBTParamCompressorPlugin))]' in run
    assert 'public static class DesktopFBTParamCompressorPass' in run and 'internal static bool IsExcluded(string name, VRCExpressionParameters.ValueType type, DesktopFBTParamCompressor s)' in run
    assert 'private const string Root = "DesktopFBT/ParamSync/";' in run and not DK.PREFIX.startswith('DesktopFBT/')
    assert 'private const int MaxNumbersPerState = 8;' in run and 'private const int MaxBoolsPerState = 64;' in run      # 工具的估算用同樣的數字
    assert 'const string OwnCompressorType = "DesktopFBT.DesktopFBTParamCompressor";' in tool
    assert 'const string OwnPassType = "DesktopFBT.DesktopFBTParamCompressorPass";' in tool
    import json
    rt = json.load(open(os.path.join(base, 'Runtime', 'DesktopFBT.ParamCompressor.Runtime.asmdef'), encoding='utf-8'))
    ed = json.load(open(os.path.join(base, 'Editor', 'DesktopFBT.ParamCompressor.Editor.asmdef'), encoding='utf-8'))
    assert rt['includePlatforms'] == [] and ed['includePlatforms'] == ['Editor']
    assert rt['name'] in ed['references'] and 'nadena.dev.ndmf' in ed['references']
    assert 'YFBT_NDMF' in ed['defineConstraints'] and any(v['name'] == 'nadena.dev.ndmf' and v['define'] == 'YFBT_NDMF' for v in ed['versionDefines'])
    assert not any('lil' in r.lower() for r in ed['references'] + rt['references'])
    for src in (comp, run):
        t = re.sub(r'//[^\n]*', '', src)
        t = re.sub(r'"(\\.|[^"\\])*"', '""', t)
        t = re.sub(r"'(\\.|[^'\\])'", "''", t)
        assert t.count('{') == t.count('}') and t.count('(') == t.count(')')


def test_package():
    d = tempfile.mkdtemp()
    p = unity_pkg.build(os.path.join(d, 't.unitypackage'), [])
    paths = []
    with tarfile.open(p) as t:
        for m in t.getmembers():
            if m.name.endswith('/pathname'):
                paths.append(t.extractfile(m).read().decode())
    assert 'Assets/DesktopFBT/Editor/Desktop/DesktopFBTDesktop.cs' in paths
    assert 'Assets/DesktopFBT/Editor/Desktop/DesktopFBT.Desktop.Editor.asmdef' in paths
    for f in ('Runtime/DesktopFBTParamCompressor.cs', 'Runtime/DesktopFBT.ParamCompressor.Runtime.asmdef',
              'Editor/DesktopFBTParamCompressorPass.cs', 'Editor/DesktopFBT.ParamCompressor.Editor.asmdef', 'THIRD_PARTY_NOTICES.txt'):
        assert 'Assets/DesktopFBT/ParamCompressor/' + f in paths, f
    assert 'Assets/DesktopFBT/ParamCompressor' in paths and 'Assets/DesktopFBT/ParamCompressor/Runtime' in paths      # 資料夾也要有項目
    assert len({g for _, _, g, _ in unity_pkg.TOOLS}) == len(unity_pkg.TOOLS)
    proj = os.path.join(d, 'proj')
    os.makedirs(os.path.join(proj, 'Assets'))
    os.makedirs(os.path.join(proj, 'ProjectSettings'))
    assert unity_pkg.install(proj, []) == []
    assert os.path.exists(os.path.join(proj, 'Assets', 'DesktopFBT', 'Editor', 'Desktop', 'DesktopFBTDesktop.cs.meta'))
    comp = os.path.join(proj, 'Assets', 'DesktopFBT', 'ParamCompressor', 'Runtime', 'DesktopFBTParamCompressor.cs')
    with open(comp + '.meta', encoding='utf-8') as fh:                      # 元件的 GUID 固定：重新安裝後 Avatar 上的元件不會變成 Missing Script
        assert 'guid: 03780ebbcd91441abca743676728938b' in fh.read()


def test_install_replaces_old_name():
    """改名前裝過的專案：舊腳本（同一個 GUID）要拿掉，否則 Unity 會有兩個 .bvh 匯入器；動作檔和別人的檔案不能動。"""
    old = 'Youtube' + 'FBT'
    proj = os.path.join(tempfile.mkdtemp(), 'proj')
    ed = os.path.join(proj, 'Assets', old, 'Editor')
    os.makedirs(os.path.join(ed, 'Desktop'))
    os.makedirs(os.path.join(proj, 'ProjectSettings'))
    os.makedirs(os.path.join(proj, 'Assets', old, 'Motions'))
    ours = [(os.path.join(ed, old + 'BvhImporter.cs'), unity_pkg.SCRIPT_GUID),
            (os.path.join(ed, 'Desktop', old + 'Desktop.cs'), unity_pkg.TOOLS[0][2]),
            (os.path.join(ed, 'Desktop', old + '.Desktop.Editor.asmdef'), unity_pkg.TOOLS[1][2])]
    for f, guid in ours:
        open(f, 'w').write('old')
        open(f + '.meta', 'w').write(f'fileFormatVersion: 2\nguid: {guid}\n')
    motion = os.path.join(proj, 'Assets', old, 'Motions', 'dance.anim')
    open(motion, 'w').write('keep')
    unity_pkg.install(proj, [])
    assert not any(os.path.exists(f) or os.path.exists(f + '.meta') for f, _ in ours)
    assert not os.path.exists(ed) and os.path.exists(motion)
    assert os.path.exists(os.path.join(proj, 'Assets', 'DesktopFBT', 'Editor', 'DesktopFBTBvhImporter.cs'))

    # 同名但 GUID 不是我們的（使用者自己的檔案）→ 不刪
    os.makedirs(ed)
    mine = os.path.join(ed, old + 'BvhImporter.cs')
    open(mine, 'w').write('mine')
    open(mine + '.meta', 'w').write('fileFormatVersion: 2\nguid: 00000000000000000000000000000000\n')
    unity_pkg.install(proj, [])
    assert open(mine).read() == 'mine'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
