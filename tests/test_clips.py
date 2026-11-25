"""動作庫：把 .anim 讀回來播成 Avatar 參數。肌肉 → 旋轉的反函式、讀檔與內插、循環接縫、錯誤的檔案、清單、引擎播放。
不需要模型。執行：python tests/test_clips.py"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp()
import paths  # noqa: E402
paths.base_dir = lambda: TMP

import anim as A  # noqa: E402
import clips as CL  # noqa: E402
import desktop as DK  # noqa: E402
import motion  # noqa: E402

FPS = 30.0


def walk(n=90):
    return np.stack([motion.person(i / FPS) for i in range(n)])


def write_text(name, body):
    p = os.path.join(TMP, name)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(body)
    return p


def curve(attr, keys):
    """手寫一條 Unity 曲線。keys = [(時間, 值, 進入斜率, 離開斜率)]"""
    s = '  - curve:\n      serializedVersion: 2\n      m_Curve:\n'
    for t, v, a, b in keys:
        s += (f'      - serializedVersion: 3\n        time: {t}\n        value: {v}\n        inSlope: {a}\n        outSlope: {b}\n'
              '        tangentMode: 0\n        weightedMode: 0\n        inWeight: 0.33333334\n        outWeight: 0.33333334\n')
    return s + f'      m_PreInfinity: 2\n      m_PostInfinity: 2\n      m_RotationOrder: 4\n    attribute: {attr}\n    path: \n    classID: 95\n    script: {{fileID: 0}}\n'


def anim_text(curves, extra=''):
    return ('%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!74 &7400000\nAnimationClip:\n  m_Name: hand\n  m_PositionCurves: []\n'
            + extra + '  m_FloatCurves:\n' + ''.join(curves) + '  m_PPtrCurves: []\n  m_SampleRate: 60\n')


def test_inverse():
    """肌肉值 → 骨頭旋轉 → 肌肉值：要回到原來的數字（預覽和「沒有高度時算腳底」都靠這個）。"""
    rng = np.random.default_rng(0)
    worst = 0.0
    for name in DK.MODEL:
        idx = [d[0] for d in DK.MODEL[name][3] if d is not None]
        for _ in range(20):
            mus = {i: float(rng.uniform(-0.9, 0.9)) for i in idx}
            back = DK.muscles_of(name, CL.local_rotation(name, mus))
            worst = max(worst, max(abs(back[i] - mus[i]) for i in idx))
    assert worst < 1e-6, worst
    print(f'  肌肉 ↔ 旋轉：來回誤差 {worst:.1e}')


def test_roundtrip():
    """自己寫出去的 .anim 讀回來：每個參數和直接算的一樣（關鍵影格被精簡過，容許一點誤差）。"""
    X = walk(150)                                               # 含手臂扭轉突然翻過去的那一幀
    cv = A.curves(X, FPS)
    path = A.write(os.path.join(TMP, 'walk.anim'), X, FPS, 'walk')
    c = CL.Clip(path)
    assert not c.pose and abs(c.seconds - (len(X) - 1) / FPS) < 1e-3 and c.name == 'walk'
    assert {'YawS', 'YawC', 'TiltX', 'TiltZ', 'Y'} <= set(c.P) and sum(k.startswith('m') for k in c.P) == len(DK.MUSCLES)
    worst = 0.0
    for i in range(len(X)):
        prm = c.sample(i / FPS, loop=False)
        for m in DK.MUSCLES:
            name = A.MUSCLE_NAMES[m]
            if name in cv:
                worst = max(worst, abs(prm[f'm{m}'] - cv[name][i]))
        yaw, (tx, tz) = DK.body_tilt(np.array([cv['RootQ.' + a][i] for a in 'xyzw']))
        assert abs(prm['TiltX'] - tx) < 0.02 and abs(prm['TiltZ'] - tz) < 0.02
        d = np.radians(yaw) - np.arctan2(prm['YawS'], prm['YawC'])
        assert abs(np.arctan2(np.sin(d), np.cos(d))) < 0.05
        assert abs(np.hypot(prm['YawS'], prm['YawC']) - 1) < 1e-6
        assert abs(prm['Y'] - np.clip(cv['RootT.y'][i] / 0.75 - 1, -1, 1)) < 0.02
    assert worst < 0.005, worst                                 # 寫檔時精簡關鍵影格的容許誤差是 0.003
    raw = CL.parse_anim(path)
    kt = raw['RootT.y'][0] * FPS
    assert np.abs(kt - np.round(kt)).max() < 1e-5               # 關鍵影格的時間寫得夠準（以前只寫 6 位，長影片後段只準到毫秒）
    fine = np.arange(0, (len(X) - 1) / FPS, 1 / 240)            # 兩幀之間也不能衝過頭（Unity 用斜率做三次內插）
    over = max(np.abs(CL._sample(raw[n], fine) - np.interp(fine, np.arange(len(X)) / FPS, v)).max() for n, v in cv.items())
    assert over < 0.005, over
    assert all(np.isfinite(v) and abs(v) <= 1 + 1e-6 for v in c.sample(1.234).values())
    img = CL.draw(c.frame(10))
    assert img.shape == (560, 560, 3) and img.std() > 5         # 預覽畫得出骨架
    print(f'  讀回 .anim：{c.n} 幀，肌肉最大差 {worst:.4f}，幀與幀之間最大偏離 {over:.4f}')


def test_hermite_and_stepped():
    """別人做的動畫：關鍵影格少、靠斜率內插；斜率是 Infinity 的是階梯。只有肌肉、沒有 Root 曲線也要能播。"""
    names = [A.MUSCLE_NAMES[m] for m in DK.MUSCLES[:8]]
    cs = [curve(names[0], [(0, 0, 0, 2), (1, 1, 0, 0)]),          # 斜率 2 出發、平著到 1
          curve(names[1], [(0, -0.5, 0, 'Infinity'), (0.5, 0.5, 'Infinity', 0), (1, 0.5, 0, 0)])]
    cs += [curve(n, [(0, 0.1, 0, 0), (1, 0.1, 0, 0)]) for n in names[2:]]
    c = CL.Clip(write_text('hand.anim', anim_text(cs, extra='  m_EulerCurves: []\n')))
    assert abs(c.seconds - 1.0) < 1e-6 and not c.pose
    a, b = f'm{DK.MUSCLES[0]}', f'm{DK.MUSCLES[1]}'
    u = 0.25                                                    # Hermite：p0=0, m0=2, p1=1, m1=0
    want = (u ** 3 - 2 * u ** 2 + u) * 2 + (-2 * u ** 3 + 3 * u ** 2)
    assert abs(c.sample(0.25, False)[a] - want) < 0.01, (c.sample(0.25, False)[a], want)
    assert abs(c.sample(0.3, False)[b] + 0.5) < 1e-6 and abs(c.sample(0.7, False)[b] - 0.5) < 1e-6      # 階梯
    p = c.sample(0.5, False)
    assert abs(p['YawS']) < 1e-6 and abs(p['YawC'] - 1) < 1e-6 and abs(p['TiltX']) < 1e-6      # 沒有 RootQ：面向前、不傾斜
    assert -1 < p['Y'] < 1 and abs(p['Y'] - CL.grounded_y(p)) < 1e-6                            # 沒有 RootT.y：照腳底算高度
    print(f'  手寫的動畫：Hermite {c.sample(0.25, False)[a]:.3f}（應為 {want:.3f}），沒有 Root 時高度 {p["Y"]:+.2f}')


def test_real_world_quirks():
    """真的第三方動畫裡看到的：RootQ 沒有正規化（Unity 播放時才正規化）、肌肉值超過 ±1、還有 IK 目標等其他曲線。"""
    names = [A.MUSCLE_NAMES[m] for m in DK.MUSCLES[:6]]
    k2 = lambda v: [(0, v, 0, 0), (0.1666667, v, 0, 0)]                               # noqa: E731
    cs = [curve(names[0], k2(1.47))] + [curve(n, k2(0.2)) for n in names[1:]]
    cs += [curve(f'RootQ.{a}', k2(v)) for a, v in zip('xyzw', (-0.66, 0, 0, 1))]
    cs += [curve(f'RootT.{a}', k2(v)) for a, v in zip('xyz', (0.5, 0.13, 2.0))]
    cs += [curve('LeftFootT.x', k2(0.3)), curve('LeftHand.Index.1 Stretched', k2(0.8))]
    c = CL.Clip(write_text('lay.anim', anim_text(cs)))
    p = c.sample(0.0)
    assert c.pose and p[f'm{DK.MUSCLES[0]}'] == 1.0                                   # 超過範圍的夾到 1
    assert abs(abs(p['TiltX']) - 2 * np.degrees(np.arctan(0.66)) / 180) < 1e-3 and abs(p['TiltZ']) < 1e-6
    assert abs(p['Y'] - (0.13 / 0.75 - 1)) < 1e-6 and len(p) == len(DK.MUSCLES) + 5    # 手指、IK、水平移動都不送
    print(f'  沒正規化的 RootQ → 傾斜 {abs(p["TiltX"]) * 180:.0f}°，躺著的高度 {p["Y"]:+.2f}')


def test_file_variants():
    """同一個動畫的幾種寫法都要讀出一樣的結果：CRLF、BOM、舊版的關鍵影格（沒有 serializedVersion 那一行）、日文檔名、後面還有一份 m_EditorCurves。"""
    names = [A.MUSCLE_NAMES[m] for m in DK.MUSCLES[:6]]
    cs = [curve(n, [(0, 0.1 * i, 0, 1), (0.4, 0.3, 1, -1), (1.0, -0.2, 0, 0)]) for i, n in enumerate(names)]
    text = anim_text(cs) + '  m_EditorCurves:\n' + ''.join(curve(n, [(0, 0.9, 0, 0), (5.0, 0.9, 0, 0)]) for n in names)
    ref = CL.Clip(write_text('ref.anim', text))
    assert abs(ref.seconds - 1.0) < 1e-6                         # 只讀 m_FloatCurves：後面那份（這裡故意寫成不同內容）不算
    variants = {'crlf.anim': text.replace('\n', '\r\n').encode(), 'bom.anim': b'\xef\xbb\xbf' + text.encode(),
                '日本語 手を振る.anim': text.encode(),
                'old.anim': text.replace('      - serializedVersion: 3\n        time:', '      - time:').encode()}
    for name, raw in variants.items():
        p = os.path.join(TMP, name)
        with open(p, 'wb') as f:
            f.write(raw)
        c = CL.Clip(p)
        assert c.n == ref.n and all(np.allclose(c.P[k], ref.P[k]) for k in ref.P), name
        assert CL._info(p, 'added')['seconds'] > 0.9 and not CL._info(p, 'added')['pose']
    print(f'  {len(variants)} 種寫法結果一樣')


def test_new_unity_layout():
    """新版 Unity 的寫法：每條曲線是「- serializedVersion: 2」換行「curve:」（舊版是「- curve:」），後面多一行 flags。
    使用者的 Arona_Dance.anim 就是這種，v1.10.1 以前會被當成「不是 Humanoid」。"""
    names = [A.MUSCLE_NAMES[m] for m in DK.MUSCLES[:7]]
    cs = [curve(n, [(0, 0.1 * i, 0, 1), (0.4, 0.3, 1, -1), (1.0, -0.2, 0, 0)]) for i, n in enumerate(names)]
    old = anim_text(cs)
    new = old.replace('  - curve:\n', '  - serializedVersion: 2\n    curve:\n').replace('    script: {fileID: 0}\n', '    script: {fileID: 0}\n    flags: 16\n')
    blend = ('  - serializedVersion: 2\n    curve:\n      serializedVersion: 2\n      m_Curve:\n      - serializedVersion: 3\n        time: 0\n        value: 100\n'
             '        inSlope: 0\n        outSlope: 0\n      m_PreInfinity: 2\n      m_PostInfinity: 2\n      m_RotationOrder: 4\n'
             '    attribute: blendShape.776014972\n    path: Body\n    classID: 137\n    script: {fileID: 0}\n    flags: 0\n')
    new = new.replace('  m_PPtrCurves: []', blend + '  m_PPtrCurves: []')
    assert new != old and '- curve:' not in new
    a, b = CL.Clip(write_text('layout_old.anim', old)), CL.Clip(write_text('layout_new.anim', new))
    assert a.n == b.n and all(np.allclose(a.P[k], b.P[k]) for k in a.P)
    raw = CL.parse_anim(os.path.join(TMP, 'layout_new.anim'))
    assert len(raw) == 8 and len(raw[names[0]][0]) == 3 and 'blendShape.776014972' in raw
    print('  新版 Unity 的曲線寫法：和舊版讀出一樣的結果')


def test_fit_range():
    """肌肉值超過 ±1（Unity 會照比例轉更多，Avatar 參數只到 ±1）：被夾住的由下一根骨頭補回來，手在空間中的方向接近原本的。"""
    def world(M, n):
        W = {'Hips': np.tile(np.array([0.0, 0.0, 0.0, 1.0])[:, None], (1, n))}
        for name, (par, _) in DK.RIG.items():
            if par is not None:
                W[name] = DK.qmul(W[par], CL._local_v(name, M, n)) if name in DK.MODEL else W[par]
        return W

    def angle(a, b):
        return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(0)), 0, 1)))
    rng = np.random.default_rng(1)
    n = 200
    inside = {i: rng.uniform(-0.95, 0.95, n) for i in DK.MUSCLES}
    fit = CL.fit_range(inside, n)
    assert max(np.abs(fit[i] - inside[i]).max() for i in inside) < 1e-9          # 本來就在範圍內：不變
    sh = [d[0] for d in DK.MODEL['LeftShoulder'][3] if d is not None]
    leg = [d[0] for d in DK.MODEL['LeftUpperLeg'][3] if d is not None]
    M = {i: rng.uniform(-0.4, 0.4, n) for i in DK.MUSCLES}
    for i in sh + leg[-1:]:
        M[i] = rng.uniform(1.2, 2.4, n) * rng.choice([-1, 1], n)                 # 肩膀、大腿扭轉超出很多（實際檔案裡有 2.4）
    fit = CL.fit_range(M, n)
    assert all(np.abs(v).max() <= 1 + 1e-9 for v in fit.values()) and set(fit) == set(DK.MUSCLES)
    want, naive, got = world(M, n), world({i: np.clip(v, -1, 1) for i, v in M.items()}, n), world(fit, n)
    for bone in ('LeftUpperArm', 'LeftHand', 'LeftFoot'):
        e0, e1 = angle(want[bone], naive[bone]).mean(), angle(want[bone], got[bone]).mean()
        assert e1 < 0.35 * e0 and e1 < 6, (bone, e0, e1)
    nan = write_text('nan.anim', anim_text([curve(A.MUSCLE_NAMES[i], [(0, 0.0, 0, 0), (1, 'NaN' if i == 39 else 'Infinity' if i == 40 else 0.3, 0, 0)])
                                            for i in DK.MUSCLES]))
    c = CL.Clip(nan)                                                # 檔案裡有壞掉的數字：不會讓整條手臂變成 NaN
    assert all(np.isfinite(v).all() and np.abs(v).max() <= 1 for v in c.P.values())
    path = write_text('over.anim', anim_text([curve(A.MUSCLE_NAMES[i], [(0, 0.0, 0, 0), (1, float(M[i][0]), 0, 0)]) for i in DK.MUSCLES]))
    c = CL.Clip(path)
    assert 0 < c.over < 1 and all(np.abs(v).max() <= 1 for v in c.P.values())
    assert max(np.abs(np.diff(v)).max() for k, v in c.P.items() if k.startswith('m')) < 0.12      # 補償進來的那一幀沒有跳一下
    e0, e1 = angle(want['LeftHand'], naive['LeftHand']).mean(), angle(want['LeftHand'], got['LeftHand']).mean()
    print(f'  超出範圍：手的方向誤差 直接夾住 {e0:.0f}° → 補償後 {e1:.1f}°')


def test_upper_chest_and_transforms():
    """上胸的肌肉不在送出的參數裡 → 角度加到胸；另外有直接移動骨頭的曲線 → 標記起來（那部分不送）。"""
    chest, upper = 'Chest Front-Back', 'UpperChest Front-Back'
    others = [A.MUSCLE_NAMES[m] for m in DK.MUSCLES if A.MUSCLE_NAMES[m] != chest][:6]
    k2 = lambda v: [(0, v, 0, 0), (1, v, 0, 0)]                                       # noqa: E731
    cs = [curve(chest, k2(0.2)), curve(upper, k2(0.6))] + [curve(n, k2(0.0)) for n in others]
    c = CL.Clip(write_text('upper.anim', anim_text(cs)))
    assert abs(c.sample(0.5)[f'm{A.MUSCLE_NAMES.index(chest)}'] - 0.5) < 1e-6 and not c.transforms
    euler = ('  m_EulerCurves:\n  - curve:\n      serializedVersion: 2\n      m_Curve:\n      - serializedVersion: 3\n        time: 0\n'
             '        value: {x: -207.7, y: 0, z: 0}\n        inSlope: {x: 0, y: 0, z: 0}\n        outSlope: {x: 0, y: 0, z: 0}\n'
             '      m_PreInfinity: 2\n      m_PostInfinity: 2\n      m_RotationOrder: 4\n    path: Armature/Hips/Spine\n')
    p = write_text('mixed.anim', anim_text(cs, extra=euler))
    c = CL.Clip(p)
    assert c.transforms and abs(c.seconds - 1.0) < 1e-6          # 骨頭曲線的 time: 不會被當成肌肉曲線
    r = CL.add([p])
    assert len(r['added']) == 1 and r['warnings'] == [dict(name='mixed.anim', warning='WARN_ANIM_TRANSFORMS')]
    assert CL.add(r['added']) == dict(added=[], errors=[], warnings=[])               # 選到已經在動作庫裡的：不再複製
    CL.remove(r['added'][0])
    print('  上胸 → 胸；骨頭曲線有標記')


def test_pose_and_loop():
    X = walk(1)
    path = A.write(os.path.join(TMP, 'one_pose.anim'), np.repeat(X, 2, 0), FPS, 'one_pose')
    c = CL.Clip(path)
    assert c.pose and c.sample(0.0) == c.sample(123.4)          # 姿勢：一直是同一個
    c = CL.Clip(A.write(os.path.join(TMP, 'loop.anim'), walk(100), FPS, 'loop'))
    k = f'm{DK.MUSCLES[20]}'
    first, last = c.sample(0.0, True), c.sample(c.seconds - 1e-4, True)
    assert max(abs(first[m] - last[m]) for m in first) < 0.02   # 循環：結尾接回開頭，不會跳一下
    step = max(abs(c.sample(t, True)[m] - c.sample(t + 1 / 60, True)[m])
               for t in np.arange(c.seconds - 0.5, c.seconds + 0.5, 1 / 60) for m in first)
    assert step < 0.25, step
    end = c.sample(c.seconds + 5, False)
    assert end == c.sample(c.seconds, False) and abs(end[k] - c.frame(c.n - 1)[k]) < 1e-9      # 不循環：停在最後
    print(f'  循環接縫：每幀最大變化 {step:.3f}')


def test_bad_files():
    cases = {
        'binary.anim': (b'\x00\x01UnityFS binary', 'ERR_ANIM_BINARY'),
        'generic.anim': (anim_text([curve('m_LocalPosition.x', [(0, 0, 0, 0), (1, 1, 0, 0)])]).encode(), 'ERR_ANIM_NOT_HUMANOID'),
        'empty.anim': (anim_text([]).encode(), 'ERR_ANIM_NOT_HUMANOID'),
    }
    for name, (raw, want) in cases.items():
        p = os.path.join(TMP, name)
        with open(p, 'wb') as f:
            f.write(raw)
        try:
            CL.Clip(p)
        except ValueError as e:
            assert str(e) == want, (name, e)
        else:
            raise AssertionError(name)
    try:
        CL.Clip(os.path.join(TMP, 'nothing.anim'))
    except ValueError as e:
        assert str(e) == 'ERR_ANIM_READ'
    else:
        raise AssertionError('missing')
    print('  二進位／不是 Humanoid／不存在：都有明確的錯誤')


def test_library():
    src = A.write(os.path.join(TMP, 'mine.anim'), walk(40), FPS, 'mine')
    r = CL.add([src, src, os.path.join(TMP, 'binary.anim')])
    assert len(r['added']) == 2 and r['added'][0] != r['added'][1]             # 同名：不蓋掉前一個
    assert r['errors'] == [dict(name='binary.anim', error='ERR_ANIM_BINARY')]
    assert r['warnings'] == []
    exp = A.write(os.path.join(CL.V.export_dir(), 'dance.anim'), walk(30), FPS, 'dance')
    A.write(os.path.join(CL.V.export_dir(), 'pic_pose.anim'), np.repeat(walk(1), 2, 0), FPS, 'pic_pose')
    items = CL.list_all()
    assert [i['kind'] for i in items] == ['added', 'added', 'export', 'export']
    by = {i['name']: i for i in items}
    assert by['pic_pose']['pose'] and not by['dance']['pose'] and abs(by['dance']['seconds'] - 29 / FPS) < 1e-3
    assert CL.remove(r['added'][0]) and not os.path.exists(r['added'][0])      # 自己加的：動作庫裡那一份刪掉，原本的檔案還在
    assert os.path.isfile(src) and len(CL.list_all()) == 3 and CL.hidden_count() == 0
    assert CL.remove(exp) and os.path.isfile(exp)                              # 轉換結果：只從清單隱藏，檔案留著
    assert [i['name'] for i in CL.list_all() if i['kind'] == 'export'] == ['pic_pose'] and CL.hidden_count() == 1
    assert CL.Clip(exp).n > 1                                                  # 隱藏的還是可以直接播（例如轉完當下按「播到 Avatar」）
    os.utime(exp, (time.time() + 5, time.time() + 5))                          # 同一個檔名重新轉了一次 → 再出現
    assert len(CL.list_all()) == 3 and CL.hidden_count() == 0
    assert CL.remove(exp) and CL.remove(os.path.join(CL.V.export_dir(), 'pic_pose.anim')) and CL.hidden_count() == 2
    assert [i['kind'] for i in CL.list_all()] == ['added']
    assert CL.restore_hidden() and len(CL.list_all()) == 3 and CL.hidden_count() == 0
    assert not CL.remove(src) and os.path.isfile(src)                          # 不在這兩個資料夾裡的檔案不碰
    print('  清單：加入、重名、移除、隱藏轉換結果、還原')


def test_engine():
    """引擎：點一個動作 → 60 Hz 送出；改速度／循環馬上生效；不循環播完自己停；讀不了的檔案回報錯誤。"""
    import engine as E
    sent = []

    class FakeOSC:
        def __init__(self, host, port):
            self.n = 0

        def send(self, p):
            self.n += 1
            sent.append((time.perf_counter(), dict(p)))

        def close(self):
            pass
    DK.AvatarOSC = FakeOSC
    DK.stale_avatar = lambda: None
    eng = E.Engine()

    def wait(cond, timeout=8):
        t = time.time()
        while time.time() - t < timeout:
            if cond(eng.get_state()):
                return True
            time.sleep(0.02)
        return False
    path = A.write(os.path.join(TMP, 'play.anim'), walk(45), FPS, 'play')         # 1.47 秒
    assert eng.cfg['motion'] == {'loop': True, 'speed': 1.0}
    eng.start_motion(path)
    assert wait(lambda s: s['phase'] == 'running' and s['osc'] and s['osc']['sent'] > 30)
    st = eng.get_state()
    assert st['mode'] == 'live' and st['job']['kind'] == 'motion' and st['job']['name'] == 'play' and st['job']['path'] == path
    assert st['preview'] and st['preview'].startswith('data:image/jpeg') and abs(st['job']['duration'] - 44 / FPS) < 1e-3
    time.sleep(2.0)                                                                # 循環：超過長度還在送
    st = eng.get_state()
    assert st['mode'] == 'live' and 0 <= st['job']['position'] <= st['job']['duration']
    ts = [t for t, _ in sent[-90:]]
    hz = (len(ts) - 1) / (ts[-1] - ts[0])                                          # 平均（Windows 的計時器一格 15.6 ms，個別間隔會不均）
    assert 45 < hz < 75, hz
    n0, t_off = len(sent), time.perf_counter()
    eng.update_config({'motion': {'loop': False, 'speed': 2.0}})                   # 播了好幾圈之後關掉循環：把這一圈播完才停
    assert wait(lambda s: s['mode'] == 'idle'), eng.get_state(False)
    tail = [p for t, p in sent[n0:] if t > t_off + 0.05]
    k = max(tail[0], key=lambda m: np.ptp(CL.Clip(path).P[m]) if m.startswith('m') else 0)
    jump = max(abs(a[k] - b[k]) for a, b in zip(tail, tail[1:])) if len(tail) > 1 else 0.0
    assert jump < 0.3, jump                                                        # 沒有直接跳到最後一幀
    assert eng.get_state(False)['error'] is None
    last = CL.Clip(path).sample(99, False)
    assert max(abs(sent[-1][1][k] - last[k]) for k in last) < 1e-6                 # Avatar 留在最後的姿勢
    pose = A.write(os.path.join(TMP, 'hold_pose.anim'), np.repeat(walk(1), 2, 0), FPS, 'hold_pose')
    n = len(sent)
    eng.start_motion(pose)                                                         # 姿勢：不循環也一直送到按停止
    assert wait(lambda s: s['phase'] == 'running' and s['job'] and s['job'].get('pose'))
    time.sleep(0.5)
    assert eng.get_state(False)['mode'] == 'live' and len(sent) - n > 15
    key = 'Y'
    first, target = sent[n][1], CL.Clip(pose).sample(0)                            # 換動作：從上一個動作最後的姿勢接過來，0.25 秒後到位
    assert max(abs(first[m] - last[m]) for m in last) < 0.1, max(abs(first[m] - last[m]) for m in last)
    assert max(abs(sent[-1][1][m] - target[m]) for m in target) < 1e-6 and key in first
    eng.stop()
    assert eng.get_state(False)['mode'] == 'idle'
    for bad, want in ((os.path.join(TMP, 'binary.anim'), 'ERR_ANIM_BINARY'), (os.path.join(TMP, 'gone.anim'), 'ERR_MOTION_MISSING')):
        eng.start_motion(bad)
        assert wait(lambda s: s['mode'] == 'idle' and s['error'])
        assert eng.get_state(False)['error'] == want, eng.get_state(False)['error']
    r = eng.add_motions([path])
    lm = eng.list_motions()
    assert len(r['added']) == 1 and any(i['path'] == r['added'][0] for i in lm['items']) and lm['hidden'] == 0
    assert eng.remove_motion(r['added'][0])
    eng.shutdown()
    print(f'  引擎：送出 {len(sent)} 次，約 {hz:.0f} Hz')


def main():
    for f in (test_inverse, test_roundtrip, test_hermite_and_stepped, test_real_world_quirks, test_file_variants, test_new_unity_layout, test_fit_range, test_upper_chest_and_transforms, test_pose_and_loop, test_bad_files, test_library, test_engine):
        print(f.__name__)
        f()
    print('OK')


if __name__ == '__main__':
    main()
