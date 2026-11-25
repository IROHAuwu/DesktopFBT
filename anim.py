"""動作 → Unity 的 Humanoid 動畫檔（.anim）。

不需要 Unity 先轉換：肌肉值用和桌面模式同一套換算（desktop.human_pose，已和 Unity 的 HumanPoseHandler 對過），
曲線直接寫成 Animator 的肌肉曲線（RootT、RootQ、各肌肉），任何 Humanoid Avatar 都能直接播放。
"""
import numpy as np

import bvh
import desktop as DK
import pose as P

MUSCLE_NAMES = [
    'Spine Front-Back', 'Spine Left-Right', 'Spine Twist Left-Right',
    'Chest Front-Back', 'Chest Left-Right', 'Chest Twist Left-Right',
    'UpperChest Front-Back', 'UpperChest Left-Right', 'UpperChest Twist Left-Right',
    'Neck Nod Down-Up', 'Neck Tilt Left-Right', 'Neck Turn Left-Right',
    'Head Nod Down-Up', 'Head Tilt Left-Right', 'Head Turn Left-Right',
    'Left Eye Down-Up', 'Left Eye In-Out', 'Right Eye Down-Up', 'Right Eye In-Out', 'Jaw Close', 'Jaw Left-Right',
    'Left Upper Leg Front-Back', 'Left Upper Leg In-Out', 'Left Upper Leg Twist In-Out', 'Left Lower Leg Stretch',
    'Left Lower Leg Twist In-Out', 'Left Foot Up-Down', 'Left Foot Twist In-Out', 'Left Toes Up-Down',
    'Right Upper Leg Front-Back', 'Right Upper Leg In-Out', 'Right Upper Leg Twist In-Out', 'Right Lower Leg Stretch',
    'Right Lower Leg Twist In-Out', 'Right Foot Up-Down', 'Right Foot Twist In-Out', 'Right Toes Up-Down',
    'Left Shoulder Down-Up', 'Left Shoulder Front-Back', 'Left Arm Down-Up', 'Left Arm Front-Back',
    'Left Arm Twist In-Out', 'Left Forearm Stretch', 'Left Forearm Twist In-Out', 'Left Hand Down-Up',
    'Left Hand In-Out',
    'Right Shoulder Down-Up', 'Right Shoulder Front-Back', 'Right Arm Down-Up', 'Right Arm Front-Back',
    'Right Arm Twist In-Out', 'Right Forearm Stretch', 'Right Forearm Twist In-Out', 'Right Hand Down-Up',
    'Right Hand In-Out',
]


def curves(X, fps):
    """X：(T,26,3) 世界座標（公尺、Y 朝上、已平滑）→ {屬性名: 每幀的值}。"""
    T = len(X)
    legs = []
    for side in ((P.L_HIP, P.L_KNEE, P.L_ANK), (P.R_HIP, P.R_KNEE, P.R_ANK)):
        legs.append(np.linalg.norm(X[:, side[0]] - X[:, side[1]], axis=1) + np.linalg.norm(X[:, side[1]] - X[:, side[2]], axis=1))
    k = DK.RIG_LEG / max(float(np.median(np.concatenate(legs))), 1e-3)    # 人的大小 → 標準骨架
    out = {n: np.zeros(T) for n in ('RootT.x', 'RootT.y', 'RootT.z', 'RootQ.x', 'RootQ.y', 'RootQ.z', 'RootQ.w')}
    mus_all = {}
    last_q = None
    frames = []
    for G in bvh.rotation_frames(X):
        W = DK._world(G)
        pos = DK._positions(W)
        frames.append((G, W, pos))
    # 高度：腳踩在地上的幀，讓標準骨架最低的點剛好貼地（人的比例和標準骨架不同，直接換算會浮起來或陷下去）
    foot = X[:, [P.L_ANK, P.R_ANK, P.L_HEEL, P.R_HEEL], 1].min(axis=1)
    on = foot < np.percentile(foot, 30) + 0.03
    dy = np.median([DK.ground_height(W, pos) - X[t, P.HIP, 1] * k
                    for t, (G, W, pos) in enumerate(frames) if on[t]]) if on.any() else 0.0
    for t in range(T):
        G, W, pos = frames[t]
        mus, qb, _ = DK.human_pose(G, 0.0)
        com = sum(DK.COM_W[n] * pos[n] for n in DK.RIG)
        hip = X[t, P.HIP] * k + np.array([0.0, dy, 0.0])
        hip_u = np.array([-hip[0], hip[1], hip[2]])                # 右手系 → Unity
        root = (hip_u + com) / DK.HUMAN_SCALE
        if last_q is not None and np.dot(last_q, qb) < 0:          # 四元數連續
            qb = -qb
        last_q = qb
        for i, a in enumerate('xyz'):
            out['RootT.' + a][t] = root[i]
        for i, a in enumerate('xyzw'):
            out['RootQ.' + a][t] = qb[i]
        for idx, v in mus.items():
            mus_all.setdefault(idx, np.zeros(T))[t] = v
    for idx, v in sorted(mus_all.items()):
        out[MUSCLE_NAMES[idx]] = v
    return out


def reduce(times, vals, tol):
    """去掉可以內插回來的關鍵影格（Douglas–Peucker，誤差 ≤ tol）。回傳保留的索引。"""
    n = len(vals)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        u = (times[a + 1:b] - times[a]) / (times[b] - times[a])
        err = np.abs(vals[a + 1:b] - (vals[a] + u * (vals[b] - vals[a])))
        i = int(np.argmax(err))
        if err[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def _f(v):
    return f'{float(v):.6g}'


def write(path, X, fps, name):
    """寫出 .anim（Unity 的文字格式）。"""
    cv = curves(X, fps)
    T = len(X)
    times = np.arange(T) / float(fps)
    dur = float(times[-1]) if T > 1 else 1.0 / fps
    lines = []
    for attr, vals in cv.items():
        keep = reduce(times, vals, 0.001 if attr.startswith('Root') else 0.003)
        # 斜率 = 到相鄰關鍵影格的直線（進、出各一個）：Unity 用斜率做三次內插，這樣兩個關鍵影格之間剛好是直線，
        # 和 reduce() 保證的誤差一致。以前寫的是原始訊號的斜率，關鍵影格稀疏、動作突然變快的地方會衝過頭（肌肉值差到 0.2）。
        chord = np.diff(vals[keep]) / np.diff(times[keep]) if len(keep) > 1 else np.zeros(0)
        lines += ['  - curve:', '      serializedVersion: 2', '      m_Curve:']
        for j, t in enumerate(keep):                 # 時間寫到 9 位：6 位的話，10 分鐘以後的關鍵影格只準到 1 毫秒
            s_in = _f(chord[j - 1] if j > 0 else (chord[0] if len(chord) else 0.0))
            s_out = _f(chord[j] if j < len(chord) else (chord[-1] if len(chord) else 0.0))
            lines += ['      - serializedVersion: 3', f'        time: {float(times[t]):.9g}', f'        value: {_f(vals[t])}',
                      f'        inSlope: {s_in}', f'        outSlope: {s_out}', '        tangentMode: 0', '        weightedMode: 0',
                      '        inWeight: 0.33333334', '        outWeight: 0.33333334']
        lines += ['      m_PreInfinity: 2', '      m_PostInfinity: 2', '      m_RotationOrder: 4',
                  f'    attribute: {attr}', '    path: ', '    classID: 95', '    script: {fileID: 0}']
    body = '\n'.join(lines)
    text = f'''%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!74 &7400000
AnimationClip:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {{fileID: 0}}
  m_PrefabInstance: {{fileID: 0}}
  m_PrefabAsset: {{fileID: 0}}
  m_Name: {name}
  serializedVersion: 7
  m_Legacy: 0
  m_Compressed: 0
  m_UseHighQualityCurve: 1
  m_RotationCurves: []
  m_CompressedRotationCurves: []
  m_EulerCurves: []
  m_PositionCurves: []
  m_ScaleCurves: []
  m_FloatCurves:
{body}
  m_PPtrCurves: []
  m_SampleRate: {_f(fps)}
  m_WrapMode: 0
  m_Bounds:
    m_Center: {{x: 0, y: 0, z: 0}}
    m_Extent: {{x: 0, y: 0, z: 0}}
  m_ClipBindingConstant:
    genericBindings: []
    pptrCurveMapping: []
  m_AnimationClipSettings:
    serializedVersion: 2
    m_AdditiveReferencePoseClip: {{fileID: 0}}
    m_AdditiveReferencePoseTime: 0
    m_StartTime: 0
    m_StopTime: {dur:.9g}
    m_OrientationOffsetY: 0
    m_Level: 0
    m_CycleOffset: 0
    m_HasAdditiveReferencePose: 0
    m_LoopTime: 0
    m_LoopBlend: 0
    m_LoopBlendOrientation: 0
    m_LoopBlendPositionY: 0
    m_LoopBlendPositionXZ: 0
    m_KeepOriginalOrientation: 0
    m_KeepOriginalPositionY: 1
    m_KeepOriginalPositionXZ: 0
    m_HeightFromFeet: 0
    m_Mirror: 0
  m_EditorCurves:
{body}
  m_EulerEditorCurves: []
  m_HasGenericRootTransform: 0
  m_HasMotionFloatCurves: 0
  m_Events: []
'''
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return path
