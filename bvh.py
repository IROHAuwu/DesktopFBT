"""把 3D 關節點序列轉成 BVH 動作檔（Blender、MotionBuilder、Unity 外掛等都能讀）。

骨架：Hips → Spine → Chest → Neck → Head，雙臂 Shoulder/Arm/ForeArm/Hand，雙腳 UpLeg/Leg/Foot/ToeBase。
靜止姿勢是 T-pose：Y 上、角色面向 +Z、角色的左邊是 +X；單位公分；旋轉順序 Z X Y。
"""
import numpy as np

import pose as P

# (名稱, 父, 靜止方向, 長度的計算方式)
#  長度：('pts', a, b) = 兩關節的中位數距離；('fixed', 公尺)
SKELETON = [
    ('Hips', None, None, None),
    ('Spine', 'Hips', (0, 1, 0), ('frac_torso', 0.0)),
    ('Chest', 'Spine', (0, 1, 0), ('frac_torso', 0.5)),
    ('Neck', 'Chest', (0, 1, 0), ('frac_torso', 0.5)),
    ('Head', 'Neck', (0, 1, 0), ('fixed', 0.10)),
    ('LeftShoulder', 'Chest', (1, 0, 0), ('fixed', 0.04)),
    ('LeftArm', 'LeftShoulder', (1, 0, 0), ('half_sho', P.L_SHO)),
    ('LeftForeArm', 'LeftArm', (1, 0, 0), ('pts', P.L_SHO, P.L_ELB)),
    ('LeftHand', 'LeftForeArm', (1, 0, 0), ('pts', P.L_ELB, P.L_WRI)),
    ('RightShoulder', 'Chest', (-1, 0, 0), ('fixed', 0.04)),
    ('RightArm', 'RightShoulder', (-1, 0, 0), ('half_sho', P.R_SHO)),
    ('RightForeArm', 'RightArm', (-1, 0, 0), ('pts', P.R_SHO, P.R_ELB)),
    ('RightHand', 'RightForeArm', (-1, 0, 0), ('pts', P.R_ELB, P.R_WRI)),
    ('LeftUpLeg', 'Hips', (1, 0, 0), ('half_hip', P.L_HIP)),
    ('LeftLeg', 'LeftUpLeg', (0, -1, 0), ('pts', P.L_HIP, P.L_KNEE)),
    ('LeftFoot', 'LeftLeg', (0, -1, 0), ('pts', P.L_KNEE, P.L_ANK)),
    ('LeftToeBase', 'LeftFoot', (0, -0.45, 1), ('pts', P.L_ANK, P.L_BIGTOE)),
    ('RightUpLeg', 'Hips', (-1, 0, 0), ('half_hip', P.R_HIP)),
    ('RightLeg', 'RightUpLeg', (0, -1, 0), ('pts', P.R_HIP, P.R_KNEE)),
    ('RightFoot', 'RightLeg', (0, -1, 0), ('pts', P.R_KNEE, P.R_ANK)),
    ('RightToeBase', 'RightFoot', (0, -0.45, 1), ('pts', P.R_ANK, P.R_BIGTOE)),
]
END_SITES = {'Head': (0, 0.12, 0), 'LeftHand': (0.08, 0, 0), 'RightHand': (-0.08, 0, 0),
             'LeftToeBase': (0, 0, 0.05), 'RightToeBase': (0, 0, 0.05)}
NAMES = [s[0] for s in SKELETON]
PARENT = {s[0]: s[1] for s in SKELETON}


# ------------------------------------------------------------------ 數學
def _n(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def frame_from(primary, secondary):
    """以 primary 為第一軸、secondary 決定扭轉，回傳正交基底（欄 = e1, e2, e3）。"""
    a = _n(np.asarray(primary, float))
    b = np.asarray(secondary, float)
    b = _n(b - np.dot(b, a) * a) if a is not None else None
    if a is None or b is None:
        return None
    return np.column_stack([a, b, np.cross(a, b)])


def align(rest_p, rest_s, obs_p, obs_s):
    """旋轉 R 使 R @ rest_p ∥ obs_p，且扭轉讓 rest_s 對到 obs_s。"""
    F0 = frame_from(rest_p, rest_s)
    F1 = frame_from(obs_p, obs_s)
    if F0 is None or F1 is None:
        return None
    return F1 @ F0.T


def swing(a, b):
    """最短的旋轉（不帶扭轉），把方向 a 轉到 b。"""
    a, b = _n(np.asarray(a, float)), _n(np.asarray(b, float))
    if a is None or b is None:
        return np.eye(3)
    c = float(a @ b)
    if c < -0.9999:                                   # 正好反向：繞任一條垂直的軸轉半圈
        ax = _n(np.cross(a, (0.0, 1.0, 0.0) if abs(a[1]) < 0.9 else (1.0, 0.0, 0.0)))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2.0 * K @ K
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


BEND_S0 = 0.35      # 彎曲角度的 sin 等於這個值時（約 20°），量到的彎曲平面和「不扭轉」的預設各信一半


def bend_axis(upper, lower, parent, rest_d, rest_n, forward_only=False):
    """兩節肢體（upper：上段、lower：下段的向量）彎曲的軸，決定這兩節骨頭的扭轉。

    量到的軸 = upper × lower，誤差大約是「骨頭方向的誤差 ÷ sin(彎曲角度)」：快打直時幾乎是亂的。
    所以和預設值加權：預設 = 從靜止姿勢用最短的旋轉擺到現在方向時的軸（相對 parent，沒有扭轉，自然的手腳大多接近這樣）。
    彎得越多越信量到的。forward_only：量到的軸和預設反向（膝蓋往後折，一定是雜訊）時不採用。"""
    upper, lower = np.asarray(upper, float), np.asarray(lower, float)
    lu, ll = np.linalg.norm(upper), np.linalg.norm(lower)
    prior = None
    if parent is not None and lu > 1e-9:
        prior = parent @ swing(rest_d, parent.T @ (upper / lu)) @ np.asarray(rest_n, float)
    n = np.cross(upper, lower)
    ln = np.linalg.norm(n)
    if ln < 1e-9 or lu < 1e-9 or ll < 1e-9:
        return prior
    n = n / ln
    if prior is None:
        return n
    sn = ln / (lu * ll)
    w = sn * sn / (sn * sn + BEND_S0 * BEND_S0)
    if forward_only:
        w *= float(np.clip((n @ prior + 0.2) / 0.5, 0.0, 1.0))
    out = w * n + (1.0 - w) * prior
    return out if np.linalg.norm(out) > 0.05 else prior


def euler_zxy(R):
    """R = Rz(c) Rx(a) Ry(b) → (z, x, y) 度。"""
    a = np.arcsin(np.clip(R[2, 1], -1, 1))
    if abs(R[2, 1]) < 0.99999:
        c = np.arctan2(-R[0, 1], R[1, 1])
        b = np.arctan2(-R[2, 0], R[2, 2])
    else:
        c = np.arctan2(R[1, 0], R[0, 0])
        b = 0.0
    return np.degrees([c, a, b])


def from_euler_zxy(z, x, y):
    z, x, y = np.radians([z, x, y])
    cz, sz, cx, sx, cy, sy = np.cos(z), np.sin(z), np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


# ------------------------------------------------------------------ 前處理
def fill_and_smooth(X, ok, fps, sigma_s=0.05):
    """補上短暫缺漏（線性內插），再做零相位高斯平滑（不會延遲）。X: (T,26,3)。"""
    T = len(X)
    X = X.copy()
    t = np.arange(T)
    for j in range(X.shape[1]):
        m = ok[:, j]
        if m.sum() == 0:
            continue
        for c in range(3):
            X[:, j, c] = np.interp(t, t[m], X[m, j, c])
    sig = max(sigma_s * fps, 0.5)
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.pad(X, ((r, r), (0, 0), (0, 0)), mode='edge')
    out = np.zeros_like(X)
    for i, w in enumerate(k):
        out += w * pad[i:i + T]
    return out


def lock_feet(X, fps, floor=0.0):
    """腳踩地時把腳固定住，去掉單鏡頭常見的「溜冰」。"""
    X = X.copy()
    for ank, toe, heel in ((P.L_ANK, P.L_BIGTOE, P.L_HEEL), (P.R_ANK, P.R_BIGTOE, P.R_HEEL)):
        low = np.minimum(X[:, toe, 1], X[:, heel, 1]) - floor
        v = np.zeros(len(X))
        v[1:] = np.linalg.norm(np.diff(X[:, ank, [0, 2]], axis=0), axis=1) * fps
        contact = (low < 0.05) & (v < 0.35)
        i = 0
        while i < len(X):
            if not contact[i]:
                i += 1
                continue
            j = i
            while j < len(X) and contact[j]:
                j += 1
            if j - i >= 3:
                for jj in (ank, toe, heel):
                    anchor = np.median(X[i:j, jj, [0, 2]], axis=0)
                    seg = X[i:j, jj, [0, 2]]
                    n = j - i
                    w = np.ones(n)                      # 進出接觸時漸變，避免跳動
                    ramp = min(3, n // 2)
                    if ramp:
                        w[:ramp] = np.linspace(0.3, 1, ramp)
                        w[-ramp:] = np.linspace(1, 0.3, ramp)
                    X[i:j, jj, 0] = seg[:, 0] + w * (anchor[0] - seg[:, 0])
                    X[i:j, jj, 2] = seg[:, 1] + w * (anchor[1] - seg[:, 1])
            i = j
    return X


# ------------------------------------------------------------------ 骨架與旋轉
def _bone_lengths(X):
    med = lambda a, b: float(np.median(np.linalg.norm(X[:, a] - X[:, b], axis=1)))  # noqa: E731
    torso = med(P.HIP, P.NECK)
    hipw = med(P.L_HIP, P.R_HIP)
    show = med(P.L_SHO, P.R_SHO)
    L = {}
    for name, parent, d, spec in SKELETON:
        if spec is None:
            continue
        kind = spec[0]
        if kind == 'pts':
            L[name] = med(spec[1], spec[2])
        elif kind == 'fixed':
            L[name] = spec[1]
        elif kind == 'frac_torso':
            L[name] = spec[1] * torso
        elif kind == 'half_hip':
            L[name] = 0.5 * hipw
        elif kind == 'half_sho':
            L[name] = max(0.5 * show - 0.03, 0.02)
    # 頭的長度：脖子到耳朵高度
    return L, torso


def offsets(L, torso):
    off = {'Hips': np.zeros(3)}
    for name, parent, d, spec in SKELETON:
        if parent is None:
            continue
        off[name] = np.asarray(d, float) / np.linalg.norm(d) * L[name]
    # 肩膀關節在胸口上方（接近脖子的高度）
    off['LeftShoulder'] = np.array([0.03, 0.46 * torso, 0.0])
    off['RightShoulder'] = np.array([-0.03, 0.46 * torso, 0.0])
    return off


def global_rotations(J):
    """單幀：Halpe26 關節點 (26,3) → {骨頭: 世界旋轉}（相對 T-pose）。"""
    G = {}
    hip = J[P.HIP]
    x_h = J[P.L_HIP] - J[P.R_HIP]                 # 角色的左邊
    chest_up = J[P.NECK] - hip
    G['Hips'] = align((0, 1, 0), (1, 0, 0), chest_up, x_h)
    x_c = J[P.L_SHO] - J[P.R_SHO]
    G['Chest'] = align((0, 1, 0), (1, 0, 0), chest_up, x_c)
    # 脊椎：骨盆和胸口的中間
    nh, nc = _n(x_h), _n(x_c)
    G['Spine'] = None if nh is None or nc is None else align((0, 1, 0), (1, 0, 0), chest_up, nh + nc)
    head_up = 0.5 * (J[P.L_EAR] + J[P.R_EAR]) - J[P.NECK]
    x_e = J[P.L_EAR] - J[P.R_EAR]
    G['Neck'] = align((0, 1, 0), (1, 0, 0), head_up, x_c + x_e)
    G['Head'] = align((0, 1, 0), (1, 0, 0), head_up, x_e)
    G['LeftShoulder'] = G['Chest']
    G['RightShoulder'] = G['Chest']
    for side, sho, elb, wri, sgn in (('Left', P.L_SHO, P.L_ELB, P.L_WRI, 1), ('Right', P.R_SHO, P.R_ELB, P.R_WRI, -1)):
        up_a, fo_a = J[elb] - J[sho], J[wri] - J[elb]
        rest_n = (0, -sgn, 0)                       # 手肘往前彎時的軸
        n = bend_axis(up_a, fo_a, G['Chest'], (sgn, 0, 0), rest_n)
        G[side + 'Arm'] = None if n is None else align((sgn, 0, 0), rest_n, up_a, n)
        G[side + 'ForeArm'] = None if n is None else align((sgn, 0, 0), rest_n, fo_a, n)
        G[side + 'Hand'] = G[side + 'ForeArm']
    for side, hp, kn, an, toe, heel in (('Left', P.L_HIP, P.L_KNEE, P.L_ANK, P.L_BIGTOE, P.L_HEEL),
                                        ('Right', P.R_HIP, P.R_KNEE, P.R_ANK, P.R_BIGTOE, P.R_HEEL)):
        th, sh = J[kn] - J[hp], J[an] - J[kn]
        n = bend_axis(th, sh, G['Hips'], (0, -1, 0), (1, 0, 0), forward_only=True)   # 膝蓋往前彎時 = 角色左邊 (+X)
        if n is None:
            n = np.array([1.0, 0.0, 0.0])
        G[side + 'UpLeg'] = align((0, -1, 0), (1, 0, 0), th, n)
        G[side + 'Leg'] = align((0, -1, 0), (1, 0, 0), sh, n)
        foot = J[toe] - J[an]
        G[side + 'Foot'] = align((0, -0.45, 1), (1, 0, 0), foot, n)
        G[side + 'ToeBase'] = G[side + 'Foot']
    return G


def rotation_frames(X):
    """整段動作逐幀的骨頭世界旋轉。某一幀算不出來的骨頭（關節重疊在一起、整段都沒拍到）沿用上一幀的，所以每根骨頭一定有值。"""
    last = {n: np.eye(3) for n in NAMES}
    for J in X:
        G = global_rotations(J)
        for n in NAMES:
            R = G.get(n)
            if R is None or not np.all(np.isfinite(R)):
                G[n] = last[n]
        last = G
        yield G


def to_bvh(X, fps, floor=0.0):
    """X: (T,26,3) 世界座標（y 上、公尺、已平滑）。回傳 BVH 文字。"""
    L, torso = _bone_lengths(X)
    off = offsets(L, torso)
    lines = ['HIERARCHY']

    def write(name, depth):
        ind = '  ' * depth
        o = off[name] * 100
        if PARENT[name] is None:
            lines.append(f'{ind}ROOT {name}')
        else:
            lines.append(f'{ind}JOINT {name}')
        lines.append(ind + '{')
        lines.append(f'{ind}  OFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}')
        if PARENT[name] is None:
            lines.append(f'{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation')
        else:
            lines.append(f'{ind}  CHANNELS 3 Zrotation Xrotation Yrotation')
        kids = [n for n in NAMES if PARENT[n] == name]
        for k in kids:
            write(k, depth + 1)
        if not kids:
            e = np.asarray(END_SITES.get(name, (0, 0.05, 0))) * 100
            lines.append(f'{ind}  End Site')
            lines.append(ind + '  {')
            lines.append(f'{ind}    OFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}')
            lines.append(ind + '  }')
        lines.append(ind + '}')
    write('Hips', 0)
    lines += ['MOTION', f'Frames: {len(X)}', f'Frame Time: {1.0 / fps:.6f}']
    prev_e = {}
    for J, G in zip(X, rotation_frames(X)):
        row = list(J[P.HIP] * 100 - np.array([0, floor * 100, 0]))
        for n in NAMES:
            p = PARENT[n]
            Rl = G[n] if p is None else G[p].T @ G[n]
            e = euler_zxy(Rl)
            if n in prev_e:                          # 角度連續（避免 179° → -179° 的跳動）
                e = prev_e[n] + (e - prev_e[n] + 180) % 360 - 180
            prev_e[n] = e
            row += list(e)
        lines.append(' '.join(f'{v:.4f}' for v in row))
    return '\n'.join(lines) + '\n'


def forward_kinematics(text):
    """讀回 BVH，回傳每幀各骨頭的世界位置 {name: (T,3)}（公尺）。測試用。"""
    toks = text.split()
    i = 0
    off, order, stack, parent = {}, [], [], {}
    while toks[i] != 'MOTION':
        if toks[i] in ('ROOT', 'JOINT'):
            name = toks[i + 1]
            parent[name] = stack[-1] if stack else None
            order.append(name)
            stack.append(name)
            i += 2
        elif toks[i] == 'End':
            depth = 0
            i += 2
            while True:
                if toks[i] == '{':
                    depth += 1
                elif toks[i] == '}':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            continue
        elif toks[i] == 'OFFSET':
            off[stack[-1]] = np.array([float(v) for v in toks[i + 1:i + 4]]) / 100
            i += 4
        elif toks[i] == '}':
            stack.pop()
            i += 1
        else:
            i += 1
    nf = int(toks[i + 2])
    data = np.array([float(v) for v in toks[i + 6:]]).reshape(nf, -1)
    out = {n: np.zeros((nf, 3)) for n in order}
    for f in range(nf):
        c = 3
        G, Pos = {}, {}
        for n in order:
            Rl = from_euler_zxy(*data[f, c:c + 3])
            c += 3
            if parent[n] is None:
                Pos[n] = data[f, :3] / 100
                G[n] = Rl
            else:
                p = parent[n]
                Pos[n] = Pos[p] + G[p] @ off[n]
                G[n] = G[p] @ Rl
            out[n][f] = Pos[n]
    return out
