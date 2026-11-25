"""給別人看的同步（第 2 版）：把姿勢壓成幾十到兩百個同步 Bool。

VRChat 一個 Avatar 的同步參數總共只有 256 bit，全身 49 個值每個只能分到幾個 bit。第 1 版（desktop.encode）每個值各自量化：
  ・還原的值是每一格的中間，0 表示不出來：站直不動時別人看到的身體是歪的（4 bit 時側傾 11°）。
  ・bit 不夠時排在後面的值完全不同步（脖子、肩膀、各種扭轉），別人看到的頭不會動、手的方向偏掉。
  ・每個值的誤差沿著手臂、腿一路累加。
第 2 版：
  ・固定幾種分配表（PROFILES），Unity 工具依 Avatar 剩下的 bit 選最大的一種；這邊每一種都算、都送（Avatar 上沒有的參數 VRChat 會忽略），
    不需要知道 Avatar 用的是哪一種。
  ・身體的傾斜、朝向用「有 0 這一格」的量化；傾斜只同步 ±90°、高度只同步用得到的範圍，同樣的 bit 比較細。
  ・從骨盆往外一根一根算：上一根骨頭量化後的誤差、沒有同步的自由度（肩膀…），由下一根補回來，手腳和頭在空間中的方向盡量對。
  ・脊椎＋胸、脖子＋頭各用一組值（兩根骨頭轉一樣多），省下來的 bit 給手腳。
Unity 那邊的還原：值 = lo + Σ bit_k · step · 2^(n-1-k)（k = 0 是最高位）。數字要和 unity/Desktop/DesktopFBTDesktop.cs 一致（tests/test_sync.py 會比對）。
"""
import math

import desktop as DK

BODY = ('YawS', 'YawC', 'TiltX', 'TiltZ', 'Y')
NAMES = tuple(f'm{i}' for i in DK.MUSCLES) + BODY          # 和 Unity 的 FloatParams() 同一個順序
# 名稱: (最小, 最大, 有沒有 0 這一格)。沒列的（肌肉）= (-1, 1, True)：肌肉的 0、±1 都剛好有一格（扭轉平常是 0，手腳伸直是 1）
RANGE = {'YawS': (-1.0, 1.0, True), 'YawC': (-1.0, 1.0, True), 'TiltX': (-0.5, 0.5, True), 'TiltZ': (-0.5, 0.5, True),
         'Y': (-0.9, 0.5, False)}
FOLLOW = {3: 0, 4: 1, 5: 2, 12: 9, 13: 10, 14: 11}          # 跟著別人的肌肉 → 它用誰的值（胸 ← 脊椎、頭 ← 脖子）
PAIR = {'Spine': 'Chest', 'Neck': 'Head'}                    # 帶頭的骨頭 → 跟著它的骨頭
_FOLLOWER_BONES = set(PAIR.values())
_MIRROR = {21: 29, 22: 30, 23: 31, 24: 32, 25: 33, 26: 34, 27: 35, 37: 46, 38: 47, 39: 48, 40: 49, 41: 50, 42: 51, 43: 52, 44: 53, 45: 54}


# 分配表（名稱 → bit 數，沒列的不同步）：tools/sync_profiles.py 在幾段動作上算出來的——從全部 6 bit 開始，
# 每次拿掉「拿掉後別人看到的姿勢變差最少」的那個 bit，左右手腳一起。改了要重新產生 Unity 那邊的表（tests/test_sync.py 會比對）。
PROFILES = [
    {'m21': 4, 'm22': 3, 'm24': 3, 'm29': 4, 'm30': 3, 'm32': 3, 'm39': 3, 'm40': 3, 'm41': 3, 'm42': 3, 'm48': 3, 'm49': 3, 'm50': 3, 'm51': 3, 'YawS': 4, 'YawC': 3, 'TiltX': 5, 'TiltZ': 4, 'Y': 4},      # 64
    {'m0': 4, 'm1': 3, 'm9': 3, 'm10': 3, 'm11': 3, 'm21': 5, 'm22': 4, 'm24': 4, 'm29': 5, 'm30': 4, 'm32': 4, 'm39': 4, 'm40': 4, 'm41': 4, 'm42': 3, 'm48': 4, 'm49': 4, 'm50': 4, 'm51': 3, 'YawS': 4, 'YawC': 4, 'TiltX': 5, 'TiltZ': 5, 'Y': 6},      # 96
    {'m0': 4, 'm1': 4, 'm9': 3, 'm10': 3, 'm11': 3, 'm21': 5, 'm22': 5, 'm23': 3, 'm24': 4, 'm25': 3, 'm26': 2, 'm29': 5, 'm30': 5, 'm31': 3, 'm32': 4, 'm33': 3, 'm34': 2, 'm39': 4, 'm40': 4, 'm41': 4, 'm42': 4, 'm48': 4, 'm49': 4, 'm50': 4, 'm51': 4, 'YawS': 4, 'YawC': 4, 'TiltX': 6, 'TiltZ': 5, 'Y': 6},      # 118
    {'m0': 4, 'm1': 5, 'm2': 3, 'm9': 3, 'm10': 3, 'm11': 3, 'm21': 5, 'm22': 6, 'm23': 3, 'm24': 5, 'm25': 4, 'm26': 3, 'm29': 5, 'm30': 6, 'm31': 3, 'm32': 5, 'm33': 4, 'm34': 3, 'm39': 5, 'm40': 5, 'm41': 4, 'm42': 5, 'm43': 3, 'm44': 2, 'm48': 5, 'm49': 5, 'm50': 4, 'm51': 5, 'm52': 3, 'm53': 2, 'YawS': 5, 'YawC': 5, 'TiltX': 6, 'TiltZ': 6, 'Y': 6},      # 149
    {'m0': 4, 'm1': 5, 'm2': 3, 'm9': 4, 'm10': 3, 'm11': 4, 'm21': 5, 'm22': 6, 'm23': 4, 'm24': 6, 'm25': 5, 'm26': 3, 'm29': 5, 'm30': 6, 'm31': 4, 'm32': 6, 'm33': 5, 'm34': 3, 'm37': 3, 'm38': 2, 'm39': 5, 'm40': 6, 'm41': 5, 'm42': 5, 'm43': 3, 'm44': 4, 'm45': 3, 'm46': 3, 'm47': 2, 'm48': 5, 'm49': 6, 'm50': 5, 'm51': 5, 'm52': 3, 'm53': 4, 'm54': 3, 'YawS': 6, 'YawC': 6, 'TiltX': 6, 'TiltZ': 6, 'Y': 6},      # 183
    {'m0': 6, 'm1': 6, 'm2': 3, 'm9': 5, 'm10': 5, 'm11': 5, 'm21': 6, 'm22': 6, 'm23': 6, 'm24': 6, 'm25': 5, 'm26': 4, 'm27': 3, 'm29': 6, 'm30': 6, 'm31': 6, 'm32': 6, 'm33': 5, 'm34': 4, 'm35': 3, 'm37': 3, 'm38': 2, 'm39': 6, 'm40': 6, 'm41': 6, 'm42': 6, 'm43': 4, 'm44': 5, 'm45': 4, 'm46': 3, 'm47': 2, 'm48': 6, 'm49': 6, 'm50': 6, 'm51': 6, 'm52': 4, 'm53': 5, 'm54': 4, 'YawS': 6, 'YawC': 6, 'TiltX': 6, 'TiltZ': 6, 'Y': 6},      # 216
]
COST = [sum(p.values()) for p in PROFILES]


def bit_name(profile, name, k):
    return f'P{profile}/{name}_{k}'


def levels(name, n):
    """n 個 bit 時：(最小值, 每一格的大小, 格數)。"""
    lo, hi, zero = RANGE.get(name, (-1.0, 1.0, True))
    count = (1 << n) - (1 if zero else 0)
    return lo, (hi - lo) / (count - 1), count


def quantize(name, n, v):
    lo, step, count = levels(name, n)
    return min(max(int(math.floor((v - lo) / step + 0.5)), 0), count - 1)


def value(name, n, q):
    lo, hi, zero = RANGE.get(name, (-1.0, 1.0, True))
    return lo + (hi - lo) * q / ((1 << n) - (2 if zero else 1))      # 這樣算 0、最大、最小剛好沒有捨入誤差


def bits(q, n):
    return [bool(q >> (n - 1 - k) & 1) for k in range(n)]


def unbits(b):
    return sum(1 << (len(b) - 1 - k) for k, x in enumerate(b) if x)


# ------------------------------------------------------------------ 四元數（純 Python：每秒要算上百次，比 numpy 快）
def _qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def _qinv(q):
    return (-q[0], -q[1], -q[2], q[3])


def _half(q):
    """轉一半：q = h · h。"""
    x, y, z, w = q if q[3] >= 0 else (-q[0], -q[1], -q[2], -q[3])
    n = math.sqrt(x * x + y * y + z * z + (w + 1.0) ** 2)
    return (x / n, y / n, z / n, (w + 1.0) / n) if n > 1e-9 else (0.0, 0.0, 0.0, 1.0)


_IDENT = (0.0, 0.0, 0.0, 1.0)
_M = {n: (tuple(float(x) for x in pre), tuple(float(x) for x in post), sign, dofs) for n, (pre, post, sign, dofs) in DK.MODEL.items()}
_CHAIN = [(n, p) for n, (p, _) in DK.RIG.items() if p is not None]


def _local(name, get):
    """desktop.muscles_of 的反運算：get(肌肉編號) → 這根骨頭的區域旋轉。"""
    pre, post, sign, dofs = _M[name]
    t = [0.0, 0.0, 0.0]
    for d in range(3):
        if dofs[d] is None:
            continue
        idx, mn, mx = dofs[d]
        v = get(idx)
        t[d] = math.tan(math.radians((v * mx if v >= 0 else -v * mn) * sign[d]) / 2.0)
    q = (t[0], t[1] + t[0] * t[2], t[2] - t[0] * t[1], 1.0)
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + 1.0)
    return _qmul(_qmul(pre, (q[0] / n, q[1] / n, q[2] / n, 1.0 / n)), _qinv(post))


def _muscles(name, L):
    """和 desktop.muscles_of 一樣：區域旋轉 → {肌肉編號: 值}（夾在 ±1）。"""
    pre, post, sign, dofs = _M[name]
    q = _qmul(_qmul(_qinv(pre), L), post)
    if q[3] < 0:
        q = (-q[0], -q[1], -q[2], -q[3])
    w = max(q[3], 1e-6)
    tx = q[0] / w if dofs[0] else 0.0
    a, b = q[1] / w, q[2] / w
    k = 1.0 + tx * tx
    t = (tx, (a - tx * b) / k, (b + tx * a) / k)
    out = {}
    for d in range(3):
        if dofs[d] is None:
            continue
        idx, mn, mx = dofs[d]
        ang = math.degrees(2.0 * math.atan(t[d])) * sign[d]
        out[idx] = min(max(ang / mx if ang >= 0 else -ang / mn, -1.0), 1.0)
    return out


def wanted(params):
    """自己這邊各骨頭在空間中的方向（相對骨盆）。每一種分配表都用同一份，先算好一次。"""
    mus = {i: float(params.get(f'm{i}', 0.0)) for i in DK.MUSCLES}
    want = {'Hips': _IDENT}
    for name, parent in _CHAIN:
        want[name] = _qmul(want[parent], _local(name, mus.__getitem__)) if name in _M else want[parent]
    return want


def remote(params, profile, want=None):
    """這一幀的參數 → (要送的 {名稱: (格子編號, bit 數)}, 別人那邊還原出來的 {名稱: 值}，49 個都有)。"""
    codes, vals = {}, {}
    for b in BODY:
        n = profile.get(b, 0)
        if n:
            q = quantize(b, n, float(params[b]))
            codes[b] = (q, n)
            vals[b] = value(b, n, q)
        else:
            vals[b] = 0.0
    if want is None:
        want = wanted(params)
    got = {'Hips': _IDENT}
    for name, parent in _CHAIN:
        if name not in _M:
            got[name] = got[parent]
            continue
        if name in _FOLLOWER_BONES:                             # 胸、頭：和脊椎、脖子轉一樣多
            m = {d[0]: vals[f'm{FOLLOW[d[0]]}'] for d in _M[name][3] if d is not None}
        else:
            back = _qinv(got[parent])
            L = _half(_qmul(back, want[PAIR[name]])) if name in PAIR else _qmul(back, want[name])
            m = _muscles(name, L)                               # 接在（量化過的）上一根後面要轉多少
            for i in m:
                key = f'm{i}'
                n = profile.get(key, 0)
                if n:
                    q = quantize(key, n, m[i])
                    codes[key] = (q, n)
                    m[i] = value(key, n, q)
                else:
                    m[i] = 0.0                                  # 沒有同步的：別人那邊是 0，少轉的由下一根補
        for i, v in m.items():
            vals[f'm{i}'] = v
        got[name] = _qmul(got[parent], _local(name, m.__getitem__))
    return codes, vals
