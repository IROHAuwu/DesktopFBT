"""桌面模式：3D 關節點 → Unity Humanoid 肌肉值與身體旋轉 → VRChat Avatar 參數（OSC）。

Avatar 上需要先用 Unity 工具（DesktopFBTDesktop.cs）加上對應的 Action 層。
肌肉模型取自 Unity 的 Humanoid（標準骨架的 pre/post 旋轉、限制符號、預設肌肉範圍），已用 Unity 本身驗證。
"""
import collections

import numpy as np

import bvh
import pose as P

# 標準骨架（Unity 座標：+X = 角色右邊、+Z = 前方；靜止時各骨頭的區域旋轉都是單位旋轉）
#   名稱: (父, 相對父骨頭的位置)
RIG = {
    'Hips': (None, (0, 0, 0)), 'Spine': ('Hips', (0, 0.0001, 0)), 'Chest': ('Spine', (0, 0.25, 0)),
    'Neck': ('Chest', (0, 0.25, 0)), 'Head': ('Neck', (0, 0.10, 0)),
    'LeftShoulder': ('Chest', (-0.03, 0.23, 0)), 'LeftUpperArm': ('LeftShoulder', (-0.15, 0, 0)),
    'LeftLowerArm': ('LeftUpperArm', (-0.28, 0, 0)), 'LeftHand': ('LeftLowerArm', (-0.25, 0, 0)),
    'RightShoulder': ('Chest', (0.03, 0.23, 0)), 'RightUpperArm': ('RightShoulder', (0.15, 0, 0)),
    'RightLowerArm': ('RightUpperArm', (0.28, 0, 0)), 'RightHand': ('RightLowerArm', (0.25, 0, 0)),
    'LeftUpperLeg': ('Hips', (-0.09, 0, 0)), 'LeftLowerLeg': ('LeftUpperLeg', (0, -0.42, 0)),
    'LeftFoot': ('LeftLowerLeg', (0, -0.42, 0)), 'LeftToes': ('LeftFoot', (0, -0.061, 0.137)),
    'RightUpperLeg': ('Hips', (0.09, 0, 0)), 'RightLowerLeg': ('RightUpperLeg', (0, -0.42, 0)),
    'RightFoot': ('RightLowerLeg', (0, -0.42, 0)), 'RightToes': ('RightFoot', (0, -0.061, 0.137)),
}
RIG_HIP_Y = 0.92
RIG_LEG = 0.84
HUMAN_SCALE = 1.0036673

# 名稱: (preQ, postQ, 限制符號, [(肌肉編號, min, max) | None] × (x, y, z))，四元數 (x, y, z, w)
_A = [-0.5, -0.5, 0.5, 0.5]
_L = [-0.5, 0.5, -0.5, 0.5]
MODEL = {
    'Spine': (_A, _A, (1, 1, 1), [(2, -40, 40), (1, -40, 40), (0, -40, 40)]),
    'Chest': (_A, _A, (1, 1, 1), [(5, -40, 40), (4, -40, 40), (3, -40, 40)]),
    'Neck': (_A, _A, (1, 1, 1), [(11, -40, 40), (10, -40, 40), (9, -40, 40)]),
    'Head': (_A, _A, (1, 1, 1), [(14, -40, 40), (13, -40, 40), (12, -40, 40)]),
    'LeftShoulder': ([0, 0, 1, 0], [0, 0, 1, 0], (1, 1, -1), [None, (38, -15, 15), (37, -15, 30)]),
    'LeftUpperArm': ([0.2442102, 0, 0.9112322, -0.3316885], [0, 0, 1, 0], (1, 1, -1),
                     [(41, -90, 90), (40, -100, 100), (39, -60, 100)]),
    'LeftLowerArm': ([0.4544878, 0.5417018, 0.5417018, -0.4544878], [0, 0.7071068, 0.7071068, 0], (1, 1, -1),
                     [(43, -90, 90), None, (42, -80, 80)]),
    'RightShoulder': ([0, 0, 0, 1], [0, 0, 0, 1], (1, 1, 1), [None, (47, -15, 15), (46, -15, 30)]),
    'RightUpperArm': ([0, -0.2442102, -0.3316885, 0.9112322], [0, 0, 0, 1], (-1, 1, 1),
                      [(50, -90, 90), (49, -100, 100), (48, -60, 100)]),
    'RightLowerArm': ([-0.5417018, -0.4544878, -0.4544878, 0.5417018], [-0.7071068, 0, 0, 0.7071068], (-1, 1, 1),
                      [(52, -90, 90), None, (51, -80, 80)]),
    'LeftHand': ([0, 0, 1, 0], [0, 0, 1, 0], (1, 1, -1), [None, (45, -40, 40), (44, -80, 80)]),
    'RightHand': ([0, 0, 0, 1], [0, 0, 0, 1], (1, 1, 1), [None, (54, -40, 40), (53, -80, 80)]),
    'LeftUpperLeg': ([-0.6123892, 0.3535244, -0.6123892, 0.3535244], _L, (1, 1, 1),
                     [(23, -60, 60), (22, -60, 60), (21, -90, 50)]),
    'LeftLowerLeg': ([-0.0616696, 0.7044125, -0.0616696, 0.7044125], _L, (1, 1, -1),
                     [(25, -90, 90), None, (24, -80, 80)]),
    'LeftFoot': (_L, _L, (1, 1, 1), [None, (27, -30, 30), (26, -50, 50)]),
    'RightUpperLeg': ([-0.6123892, 0.3535244, -0.6123892, 0.3535244], _L, (-1, -1, 1),
                      [(31, -60, 60), (30, -60, 60), (29, -90, 50)]),
    'RightLowerLeg': ([-0.0616696, 0.7044125, -0.0616696, 0.7044125], _L, (-1, 1, -1),
                      [(33, -90, 90), None, (32, -80, 80)]),
    'RightFoot': (_L, _L, (1, -1, 1), [None, (35, -30, 30), (34, -50, 50)]),
}
ORDER = list(MODEL)
MUSCLES = sorted(m[0] for b in MODEL.values() for m in b[3] if m is not None)    # 44 個（含手腕）
# 「所有人（腿與身體）」時同步的部分：其他維持本機
SYNC_LITE = sorted(m[0] for n in ('Spine', 'Chest', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
                                  'RightUpperLeg', 'RightLowerLeg', 'RightFoot')
                   for m in MODEL[n][3] if m is not None)
BODY_PARAMS = ('YawS', 'YawC', 'TiltX', 'TiltZ', 'Y')
PREFIX = 'YFBT/'
OLD_PREFIX = 'P' + 'FBT/'

# 「所有人（全身，不壓縮）」：每個值另外拆成 6 個 Bool（_0 是最高位）。Avatar 依可用的 bit 數只收前面幾個，
#   Unity 工具裡決定；這裡一律送 6 個。自己這邊仍用完整的 Float。
MAX_BITS = 6


def _recent_avatar(root=None):
    """VRChat 的 OSC 設定檔裡，最近用過、有 DesktopFBT 參數的 Avatar：(名字, 參數名稱的集合)。找不到 → None。"""
    import glob
    import json
    import os
    root = root or os.path.join(os.path.expanduser('~'), 'AppData', 'LocalLow', 'VRChat', 'VRChat', 'OSC')
    try:
        files = sorted(glob.glob(os.path.join(root, '*', 'Avatars', '*.json')), key=os.path.getmtime, reverse=True)
    except OSError:
        return None
    for f in files[:30]:
        try:
            with open(f, encoding='utf-8-sig') as fp:
                d = json.load(fp)
        except (OSError, ValueError):
            continue
        names = {p.get('name') for p in d.get('parameters', []) if isinstance(p, dict)}
        names = {n for n in names if isinstance(n, str)}
        if any(n.startswith(PREFIX) or n.startswith(OLD_PREFIX) for n in names):     # OLD_PREFIX：改名前（Phone FBT）的控制器
            return (d.get('name') or os.path.basename(f)), names
    return None


def stale_avatar(root=None):
    """最近用過、有 DesktopFBT 參數的 Avatar 如果還是舊版控制器（沒有 TiltX），回傳它的名字。"""
    r = _recent_avatar(root)
    return None if r is None or PREFIX + 'TiltX' in r[1] else r[0]


def old_sync_avatar(root=None):
    """最近用過的 Avatar 如果用的是第 1 版的同步（每個值各自拆成 Bool：別人看到的身體會歪、細節少），回傳它的名字。
    第 2 版的同步參數名稱是 YFBT/P<表的編號>/…（sync.py）。VRChat 的設定檔在重新上傳後不一定會更新，所以只當提醒用。"""
    r = _recent_avatar(root)
    if r is None or PREFIX + 'TiltX_0' not in r[1]:
        return None
    return None if any(n.startswith(PREFIX + 'P') and '/' in n[len(PREFIX):] for n in r[1]) else r[0]


def encode(v, n=MAX_BITS):
    q = min(int((float(np.clip(v, -1.0, 1.0)) + 1.0) * 0.5 * (1 << n)), (1 << n) - 1)
    return [bool(q >> (n - 1 - k) & 1) for k in range(n)]


def decode(b):
    """只收前 n 個 bit 時 Avatar 還原出的值：-1 + 2^-n + Σ b_k 2^-k"""
    n = len(b)
    return -1.0 + 2.0 ** -n + sum(2.0 ** -k for k, x in enumerate(b) if x)


# 質心 = 各骨頭位置的加權和（以這個標準骨架擬合，誤差 < 1e-6 m）
COM_W = {'Hips': -1.883343, 'Spine': 0.064707, 'Chest': 0.081525, 'Neck': 0.005721, 'Head': 0.054546,
         'LeftShoulder': 0.115466, 'LeftUpperArm': 0.024243, 'LeftLowerArm': 0.012121, 'LeftHand': 0.015152,
         'RightShoulder': 0.11547, 'RightUpperArm': 0.015151, 'RightLowerArm': 0.021212, 'RightHand': 0.015152,
         'LeftUpperLeg': 1.050227, 'LeftLowerLeg': 0.084849, 'LeftFoot': 0.033939, 'LeftToes': 0.002424,
         'RightUpperLeg': 1.050227, 'RightLowerLeg': 0.084848, 'RightFoot': 0.033939, 'RightToes': 0.002424}

BVH_NAME = {'Hips': 'Hips', 'Spine': 'Spine', 'Chest': 'Chest', 'Neck': 'Neck', 'Head': 'Head',
            'LeftShoulder': 'LeftShoulder', 'LeftUpperArm': 'LeftArm', 'LeftLowerArm': 'LeftForeArm',
            'LeftHand': 'LeftHand', 'RightShoulder': 'RightShoulder', 'RightUpperArm': 'RightArm',
            'RightLowerArm': 'RightForeArm', 'RightHand': 'RightHand', 'LeftUpperLeg': 'LeftUpLeg',
            'LeftLowerLeg': 'LeftLeg', 'LeftFoot': 'LeftFoot', 'LeftToes': 'LeftToeBase',
            'RightUpperLeg': 'RightUpLeg', 'RightLowerLeg': 'RightLeg', 'RightFoot': 'RightFoot',
            'RightToes': 'RightToeBase'}

_M = np.diag([-1.0, 1.0, 1.0])


# ------------------------------------------------------------------ 四元數 (x, y, z, w)
def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def mat_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def unity_euler(q):
    """Unity 的 eulerAngles（Quaternion.Euler(x, y, z) = Ry·Rx·Rz），度：(x, y, z)。x 在 ±90 之間。"""
    R = quat_mat(q)
    x = np.arcsin(np.clip(-R[1, 2], -1, 1))
    if abs(R[1, 2]) < 0.9999:
        y = np.arctan2(R[0, 2], R[2, 2])
        z = np.arctan2(R[1, 0], R[1, 1])
    else:
        y = np.arctan2(-R[2, 0], R[0, 0])
        z = 0.0
    return np.degrees([x, y, z])


TILT_MAX = 150.0       # 身體傾斜送到幾度（彎腰摸地板約 110°）


def _axis_angle(axis, a):
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def body_tilt(q):
    """身體旋轉 → (偏航角（度）, 傾斜 (x, z))：q = Ry(偏航) · 繞水平軸 (x, 0, z) 轉 |(x, z)|·180°。
    尤拉角在彎腰 90° 時會萬向鎖（偏航、側傾亂跳），這種拆法除了倒立以外都連續。"""
    R = quat_mat(q)
    up = R[:, 1]
    ang = float(np.arccos(np.clip(up[1], -1.0, 1.0)))
    ax = np.array([up[2], 0.0, -up[0]])                  # (0,1,0) × up
    n = np.linalg.norm(ax)
    if n < 1e-9:
        ax, ang = np.array([1.0, 0.0, 0.0]), (0.0 if up[1] > 0 else np.pi)
    else:
        ax = ax / n
    Rt = _axis_angle(ax, ang).T @ R                      # 剩下的 = 繞 y 轉
    yaw = float(np.arctan2(Rt[0, 2], Rt[2, 2]))
    c, s_ = np.cos(yaw), np.sin(yaw)
    v = ax * ang
    vx, vz = c * v[0] - s_ * v[2], s_ * v[0] + c * v[2]   # Ry(-yaw) · v
    t = np.array([vx, vz]) / np.pi
    r = np.linalg.norm(t)
    if r > TILT_MAX / 180.0:
        t *= TILT_MAX / 180.0 / r
    return np.degrees(yaw), t


def tilt_quat(yaw_deg, t):
    """body_tilt 的反運算（測試、Unity 的格點用同一個定義）。"""
    yaw = np.radians(yaw_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    r = float(np.linalg.norm(t))
    S = np.eye(3) if r < 1e-9 else _axis_angle(np.array([t[0], 0.0, t[1]]) / r, r * np.pi)
    return mat_quat(Ry @ S)


# ------------------------------------------------------------------ 肌肉
def muscles_of(name, L):
    """區域旋轉 L（Unity 四元數）→ {肌肉編號: 值}。沒有的自由度直接捨去（例如手肘不能側彎）。"""
    pre, post, sign, dofs = MODEL[name]
    q = qmul(qmul(qinv(np.asarray(pre, float)), L), np.asarray(post, float))
    if q[3] < 0:
        q = -q
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
        ang = np.degrees(2.0 * np.arctan(t[d])) * sign[d]
        v = ang / mx if ang >= 0 else -ang / mn
        out[idx] = float(np.clip(v, -1.0, 1.0))
    return out


def _world(G):
    """BVH 骨架的世界旋轉（右手、+X = 角色左邊）→ Unity 標準骨架的世界旋轉（四元數）。"""
    W = {}
    for un, bn in BVH_NAME.items():
        R = G.get(bn)
        if R is not None:
            W[un] = mat_quat(_M @ R @ _M)
    return W


def _positions(W):
    pos = {}
    for n, (p, off) in RIG.items():
        if p is None:
            pos[n] = np.zeros(3)
        else:
            pos[n] = pos[p] + quat_mat(W[p]) @ np.asarray(off, float)
    return pos


def body_rotation(pos):
    x = (pos['RightUpperLeg'] - pos['LeftUpperLeg']) + (pos['RightUpperArm'] - pos['LeftUpperArm'])
    y = 0.5 * (pos['LeftUpperArm'] + pos['RightUpperArm']) - 0.5 * (pos['LeftUpperLeg'] + pos['RightUpperLeg'])
    y = y / np.linalg.norm(y)
    x = x - np.dot(x, y) * y
    x /= np.linalg.norm(x)
    return mat_quat(np.column_stack([x, y, np.cross(x, y)]))


# 標準骨架腳底的點（相對腳踝、腳尖骨頭，靜止時剛好在地上）
_HEEL = np.array([0.0, -0.08, -0.05])
_TIP = np.array([0.0, -0.019, 0.05])


def ground_height(W, pos):
    """貼地時骨盆要多高：全身最低的點（腳跟、腳尖、手指尖、頭頂、各關節）碰到地板。"""
    pts = list(pos.values())
    for side, sx in (('Left', -1.0), ('Right', 1.0)):
        pts.append(pos[side + 'Foot'] + quat_mat(W[side + 'Foot']) @ _HEEL)
        pts.append(pos[side + 'Toes'] + quat_mat(W[side + 'Toes']) @ _TIP)
        pts.append(pos[side + 'Hand'] + quat_mat(W[side + 'Hand']) @ np.array([0.09 * sx, 0.0, 0.0]))
    pts.append(pos['Head'] + quat_mat(W['Head']) @ np.array([0.0, 0.17, 0.0]))
    return -min(p[1] for p in pts)


def human_pose(G, hip_height=None):
    """骨頭世界旋轉 → (肌肉 {編號: 值}, 身體旋轉四元數, 身體高度（Unity bodyPosition.y）)。
    G：bvh.global_rotations 的結果（全部骨頭都要有值）。hip_height：標準骨架尺度的骨盆高度（公尺）；
    None 時讓最低的腳貼地（影片裡算出來的高度不準，常常整個人浮在空中）。"""
    W = _world(G)
    mus = {}
    for n in ORDER:
        L = qmul(qinv(W[RIG[n][0]]), W[n])
        mus.update(muscles_of(n, L))
    pos = _positions(W)
    qb = body_rotation(pos)
    com = sum(COM_W[n] * pos[n] for n in RIG)
    if hip_height is None:
        hip_height = ground_height(W, pos)
    by = (hip_height + com[1]) / HUMAN_SCALE
    return mus, qb, float(by)


# COCO-WholeBody 的手（每隻 21 點）：手腕 0、食指根 5、中指根 9、無名指根 13、小指根 17
HAND_L, HAND_R = 91, 112
_KNUCKLES = (5, 9, 13, 17)


def hand_rotation(H, side):
    """手的 21 個 3D 點 → BVH 骨架的世界旋轉（T-pose 時手心朝下）。"""
    sgn = 1 if side == 'Left' else -1
    fwd = H[list(_KNUCKLES)].mean(0) - H[0]
    across = np.cross(H[5] - H[0], H[17] - H[0])
    palm = -across if side == 'Left' else across          # 從手心往外
    if np.linalg.norm(fwd) < 1e-4 or np.linalg.norm(palm) < 1e-8:
        return None
    return bvh.align((sgn, 0, 0), (0, -1, 0), fwd, palm)


def apply_hands(G, hands):
    """hands = {'Left': (21,3) | None, 'Right': ...}：手腕方向由手的點決定，前臂的扭轉跟著手。"""
    for side in ('Left', 'Right'):
        H = hands.get(side) if hands else None
        if H is None:
            continue
        Rh = hand_rotation(H, side)
        F = G.get(side + 'ForeArm')
        if Rh is None or F is None:
            continue
        sgn = 1 if side == 'Left' else -1
        rest_n = np.array((0.0, -sgn, 0.0))
        d = F @ np.array((sgn, 0.0, 0.0))
        n = Rh @ np.array((0.0, -1.0, 0.0)) * (1 if side == 'Left' else -1)
        Fn = bvh.align((sgn, 0, 0), rest_n, d, n)
        if Fn is not None:
            G[side + 'ForeArm'] = Fn
        G[side + 'Hand'] = Rh


class DesktopPose:
    """逐幀：Halpe26 3D 點（影片模式的世界座標：x = 相機右、y = 上、z = 朝向相機）→ Avatar 參數。"""

    def __init__(self, grounded=True):
        self.grounded = grounded
        self.last_G = {n: np.eye(3) for n in bvh.NAMES}
        self.leg = None
        self.qb = None

    def __call__(self, X, ok, hands=None, t=None):
        need = (P.HIP, P.NECK, P.L_HIP, P.R_HIP, P.L_SHO, P.R_SHO, P.L_KNEE, P.R_KNEE, P.L_ANK, P.R_ANK)
        if X is None or not all(ok[j] for j in need):
            return None
        X = np.asarray(X, float)
        G = bvh.global_rotations(X)
        for n in bvh.NAMES:
            if G.get(n) is None:
                G[n] = self.last_G[n]
        apply_hands(G, hands)
        self.last_G = G
        leg = 0.5 * sum(np.linalg.norm(X[a] - X[b]) + np.linalg.norm(X[b] - X[c])
                        for a, b, c in ((P.L_HIP, P.L_KNEE, P.L_ANK), (P.R_HIP, P.R_KNEE, P.R_ANK)))
        self.leg = leg if self.leg is None else 0.98 * self.leg + 0.02 * leg
        hy = float(X[P.HIP][1]) * RIG_LEG / max(self.leg, 0.3)
        mus, qb, by = human_pose(G, None if self.grounded else hy)
        if self.qb is not None and np.dot(self.qb, qb) < 0:      # 四元數連續（同一個旋轉的兩種寫法不要跳來跳去）
            qb = -qb
        self.qb = qb
        ey, tilt = body_tilt(qb)
        out = {f'm{i}': float(mus[i]) for i in MUSCLES}
        out['YawS'] = float(np.sin(np.radians(ey)))
        out['YawC'] = float(np.cos(np.radians(ey)))
        out['TiltX'] = float(tilt[0])
        out['TiltZ'] = float(tilt[1])
        out['Y'] = float(np.clip(by / 0.75 - 1.0, -1, 1))
        return out


class AvatarOSC:
    """把參數送到 VRChat：/avatar/parameters/YFBT/...（Float 給自己用，拆好的 Bool 給同步用，只在有變時送）"""

    REFRESH_S = 3.0         # 同步的 Bool 平常只送有變的；每隔這麼久全部重送一次（VRChat 換 Avatar 後也會跟上）
    RATE = 60.0             # send() 每秒被呼叫幾次（Playback、動作庫都是 60）

    def __init__(self, host='127.0.0.1', port=9000):
        from pythonosc.udp_client import SimpleUDPClient
        import sync
        self.client = SimpleUDPClient(host, port)
        self.sync = sync
        self.sent = {}
        self.last = {}
        self.n = 0

    def _bool(self, key, b, full):
        if full or self.sent.get(key) is not b:
            self.sent[key] = b
            self.client.send_message(f'/avatar/parameters/{PREFIX}{key}', b)

    def send(self, params):
        clean = {}
        for k, v in params.items():                # 自己看到的：每次都送完整的數值
            v = float(v)
            if v != v or v in (float('inf'), float('-inf')):    # 壞掉的數字（NaN／無限大）不送：沿用上一次的。送出去 VRChat 那邊的姿勢會壞掉
                v = self.last.get(k, 0.0)
            else:
                self.last[k] = v
            clean[k] = v
            self.client.send_message(f'/avatar/parameters/{PREFIX}{k}', v)
        params = clean
        # 給別人看的同步 Bool：VRChat 每秒只同步幾次，不用跟著 60 Hz。一次只算一種（輪流），不要某一幀突然算很久。
        SY = self.sync
        slots = len(SY.PROFILES) + 1
        slot, turn = self.n % slots, self.n // slots
        self.n += 1
        full = turn % max(1, int(self.REFRESH_S * self.RATE / slots)) == 0
        if slot == 0:                              # 第 1 版的同步：還沒重新上傳的 Avatar 用
            top = (1 << MAX_BITS) - 1
            for k, v in params.items():
                q = min(int((min(max(float(v), -1.0), 1.0) + 1.0) * 0.5 * (top + 1)), top)
                for i in range(MAX_BITS):
                    self._bool(f'{k}_{i}', bool(q >> (MAX_BITS - 1 - i) & 1), full)
            return
        try:                                       # 第 2 版：每一種分配表都送，Avatar 上沒有的參數 VRChat 會忽略
            for name, (q, n) in SY.remote(params, SY.PROFILES[slot - 1])[0].items():
                for k in range(n):
                    self._bool(SY.bit_name(slot - 1, name, k), bool(q >> (n - 1 - k) & 1), full)
        except (KeyError, ValueError, ZeroDivisionError):
            pass                                   # 參數不齊（不應該發生）：這一次不送同步的部分

    def close(self):
        pass


class Playback:
    """固定延遲、60 Hz 送出：在收到的姿勢之間內插，處理偶爾變慢時動作也不會卡住。

    add(t, params)：t = 這一幀被擷取的時間（time.perf_counter）。送出的是 now - delay 那一刻的姿勢。"""

    def __init__(self, out, delay, hz=60.0):
        import threading
        self.out = out
        self.delay = float(delay)
        self.dt = 1.0 / float(hz)
        self.buf = collections.deque(maxlen=600)
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def add(self, t, params):
        with self.lock:
            if self.buf and t <= self.buf[-1][0]:
                return
            self.buf.append((float(t), params))

    def sample(self, t):
        with self.lock:
            buf = list(self.buf)
        if not buf:
            return None
        if t <= buf[0][0]:
            return buf[0][1]
        if t >= buf[-1][0]:
            return buf[-1][1]                    # 還沒收到更新的：停在最後一個（不往外推）
        lo, hi = 0, len(buf) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if buf[m][0] <= t:
                lo = m
            else:
                hi = m
        (t0, a), (t1, b) = buf[lo], buf[hi]
        u = (t - t0) / max(t1 - t0, 1e-6)
        out = {k: a[k] + u * (b[k] - a[k]) for k in a if k in b}
        n = np.hypot(out.get('YawS', 0.0), out.get('YawC', 1.0))
        if n > 1e-6 and 'YawS' in out:
            out['YawS'] /= n
            out['YawC'] /= n
        return out

    def _run(self):
        import time
        nxt = time.perf_counter()
        while not self.stop_ev.is_set():
            now = time.perf_counter()
            p = self.sample(now - self.delay)
            if p is not None:
                try:
                    self.out.send(p)
                except (OSError, ValueError, OverflowError, TypeError):
                    pass                    # 這一幀送不出去（網路或數字壞掉）就跳過，不能讓這條執行緒停掉：停了 Avatar 就不動了
            with self.lock:
                while len(self.buf) > 2 and self.buf[1][0] < now - self.delay - 1.0:
                    self.buf.popleft()
            nxt += self.dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if time.perf_counter() - nxt > 0.5:
                nxt = time.perf_counter()

    def stop(self):
        self.stop_ev.set()
        self.th.join(timeout=1.0)
