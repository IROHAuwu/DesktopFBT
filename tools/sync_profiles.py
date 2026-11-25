"""決定 sync.PROFILES 的分配表：在一組姿勢上，一次加一個 bit 給「讓別人看到的姿勢最接近」的那個值（貪婪法）。
執行：python tools/sync_profiles.py 動畫1.anim [動畫2.anim …]   （另外一定會用 tests/motion.py 的模擬動作）
量的是：頭、手肘、手、膝、腳在空間中的位置差（含身體的傾斜、朝向、高度），加上頭、手、腳的方向差。"""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import anim as A  # noqa: E402
import clips as CL  # noqa: E402
import desktop as DK  # noqa: E402
import motion  # noqa: E402
import sync as SY  # noqa: E402

BUDGETS = (64, 96, 120, 152, 184, 216)
FLOOR = {'m11': (96, 3)}             # 轉頭（脖子＋頭的左右轉）：測試動作裡很少出現，但實際上很顯眼 → 96 bit 以上至少 3 bit
POINTS = {'Head': 2.0, 'LeftLowerArm': 1.0, 'RightLowerArm': 1.0, 'LeftHand': 2.0, 'RightHand': 2.0, 'LeftLowerLeg': 1.0,
          'RightLowerLeg': 1.0, 'LeftFoot': 2.0, 'RightFoot': 2.0, 'LeftToes': 1.0, 'RightToes': 1.0, 'Chest': 1.0}
FACING = {'Head': 0.25, 'LeftHand': 0.15, 'RightHand': 0.15, 'LeftFoot': 0.15, 'RightFoot': 0.15}      # 公分／度
TIP = {'Head': (0.0, 0.17, 0.0), 'LeftHand': (-0.09, 0.0, 0.0), 'RightHand': (0.09, 0.0, 0.0)}
OFF = {n: tuple(float(x) for x in off) for n, (p, off) in DK.RIG.items()}
LEADERS = [n for n in SY.NAMES if not (n[0] == 'm' and int(n[1:]) in SY.FOLLOW)]


def rot(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx)


def pose(vals):
    """{名稱: 值} → ({骨頭: 世界位置（公尺）}, {骨頭: 世界旋轉})，含身體的傾斜、朝向、高度。"""
    yaw = math.degrees(math.atan2(vals['YawS'], vals['YawC']))
    body = tuple(float(x) for x in DK.tilt_quat(yaw, np.array([vals['TiltX'], vals['TiltZ']])))
    W, P = {'Hips': body}, {'Hips': (0.0, (vals['Y'] + 1.0) * 0.75, 0.0)}
    for name, parent in SY._CHAIN:
        W[name] = SY._qmul(W[parent], SY._local(name, lambda i: vals.get(f'm{i}', 0.0))) if name in SY._M else W[parent]
        d = rot(W[parent], OFF[name])
        P[name] = tuple(a + b for a, b in zip(P[parent], d))
    for name, tip in TIP.items():
        P[name] = tuple(a + b for a, b in zip(P[name], rot(W[name], tip)))
    return P, W


def cost(frames, refs, profile):
    total = 0.0
    for f, (P0, W0) in zip(frames, refs):
        P1, W1 = pose(SY.remote(f, profile)[1])
        for n, w in POINTS.items():
            total += w * 100.0 * math.dist(P0[n], P1[n])
        for n, w in FACING.items():
            total += w * math.degrees(2 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(W0[n], W1[n]))))))
    return total / len(frames) / sum(POINTS.values())


def frames_of(paths, per=60):
    out = []
    X = np.stack([motion.person(i / 30.0) for i in range(240)])
    paths = list(paths) + [A.write(os.path.join(tempfile.mkdtemp(), 'sim.anim'), X, 30.0, 'sim')]
    for p in paths:
        c = CL.Clip(p)
        idx = np.unique(np.linspace(0, c.n - 1, per).astype(int))
        out += [c.frame(int(i)) for i in idx]
        print(f'  {os.path.basename(p)}: {len(idx)} 幀', file=sys.stderr)
    return out


def groups():
    """一起增減的值：左右手腳用同樣的 bit 數。"""
    g = []
    for n in LEADERS:
        i = int(n[1:]) if n[0] == 'm' else -1
        if i in SY._MIRROR:
            g.append((n, f'm{SY._MIRROR[i]}'))
        elif i not in SY._MIRROR.values():
            g.append((n,))
    return g


def main():
    frames = frames_of(sys.argv[1:], per=40)
    refs = [pose(f) for f in frames]
    prof = {n: 6 for n in LEADERS}                              # 從全部 6 bit 開始，每次拿掉「拿掉後變差最少」的那個 bit
    tables = {}
    for budget in sorted(BUDGETS, reverse=True):
        while sum(prof.values()) > budget:
            base, best = cost(frames, refs, prof), None
            for g in groups():
                have = prof.get(g[0], 0)
                if have == 0 or any(n in FLOOR and budget >= FLOOR[n][0] and have <= FLOOR[n][1] for n in g):
                    continue
                to = have - 1 if have > 2 else 0                # 有 0 那一格的量化最少要 2 bit（-1、0、+1）
                trial = dict(prof)
                for n in g:
                    trial[n] = to
                loss = (cost(frames, refs, trial) - base) / ((have - to) * len(g))
                if best is None or loss < best[0]:
                    best = (loss, g, to)
            for n in best[1]:
                prof[n] = best[2]
        tables[budget] = {n: v for n, v in prof.items() if v}
        print(f'# {sum(prof.values())} bit：平均誤差 {cost(frames, refs, prof):.2f} 公分', file=sys.stderr)
    print('PROFILES = [')
    for budget in sorted(tables):
        t = tables[budget]
        print('    {' + ', '.join(f"'{n}': {t[n]}" for n in SY.NAMES if t.get(n)) + '},      # ' + str(sum(t.values())))
    print(']')


if __name__ == '__main__':
    main()
