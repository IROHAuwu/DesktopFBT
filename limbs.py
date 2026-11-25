"""單相機的手腳前後修正（單張畫面就能判斷的情況）。

- 手在頭附近：畫面上手和頭重疊時，看「手蓋住的那一塊臉」的五官。五官看得到 → 手在頭後面；
  五官被擋住（或背對鏡頭、沒有臉）→ 手在鏡頭這一側。五官的資訊來自 133 點模型的 68 個臉部點（head_cue）；
  沒有這份資訊時退回舊的做法（臉和手腕的分數誰高）。
- 頭當成一顆球：前臂和手掌（手腕到指尖）都不會穿過頭。手肘不動，前臂往鏡頭或往後轉到整隻手都在球面外（前面或後面）。
- 膝蓋不會往後折：膝蓋在大腿到腳踝連線後面時，拉回到連線上（站直時模型常把膝蓋估到後面，看起來整個人站不直）。
- 手不會穿過軀幹：軀幹當成橢圓柱（左右寬、前後窄），手肘、手腕、前臂跑進去時，
  在軀幹左右中線前面的就推到前面的表面，在後面的就推到後面（移動時沿著身體的弧線繞過去）。
"""
import numpy as np

import pose as P

_ARMS = ((P.L_SHO, P.L_ELB, P.L_WRI, 'Left'), (P.R_SHO, P.R_ELB, P.R_WRI, 'Right'))
_LEGS = ((P.L_HIP, P.L_KNEE, P.L_ANK), (P.R_HIP, P.R_KNEE, P.R_ANK))
_FACE = (P.NOSE, P.L_EYE, P.R_EYE)
HEAD_R = 0.10                 # 頭的半徑（兩耳中點為球心）
NEAR_HEAD = 0.16              # 畫面上手掌離頭中心這麼近才處理
HOLD = 12                     # 判斷不出來時，沿用上一次的判斷幾幀（即時、影片）
FAR_FRONT = 0.30              # 模型把手腕放在頭前面超過這麼遠（公尺）時，不照「在頭後面」的判斷改
HAND_LEN = 0.72               # 手掌＋手指的長度（相對前臂）：匯出的動畫手掌是直的，沿著前臂延伸
R_BACK = HEAD_R + 0.03        # 手掌的中線離頭中心至少這麼遠（頭的半徑＋手的厚度）
R_FRONT = 0.14                # 臉的前面：鼻子比較突出
_DEBUG = None                 # 除錯用：設成 dict 時，_head 把每個候選的判斷結果放進去
FUNNEL = 2.0                  # 畫面上在頭的範圍外時，每離開 1cm 可以多往另一側 2cm
_SHIFTS = np.round(np.arange(-60, 61) * 0.01, 2)      # 手腕沿視線前後移動的候選（公尺）
_FORE = np.array([0.5, 0.75])                         # 前臂上檢查的位置（手肘 0 → 手腕 1）
_PALM = np.linspace(0.0, 1.0, 7)                      # 手掌上檢查的位置（手腕 0 → 指尖 1）
_R_BACK = np.array([HEAD_R + 0.01] * len(_FORE) + [R_BACK] * len(_PALM))      # 每個檢查點離頭中心至少多遠
_R_FRONT = np.array([HEAD_R + 0.01] * len(_FORE) + [R_FRONT] * len(_PALM))
_R_ANY = np.array([HEAD_R + 0.01] * len(_FORE) + [R_BACK] * len(_PALM))

# 133 點（COCO-WholeBody）：身體 0–16、腳 17–22、臉 23–90、左手 91–111、右手 112–132
_WB_FACE = slice(23, 91)
_WB_ARMS = (('Left', 9, slice(91, 112)), ('Right', 10, slice(112, 133)))


def head_cue(k, s, detail=None):
    """畫面上的判斷：手和頭重疊時，手在臉前面還是頭後面。
    k (133,2) 畫素、s (133,) 分數 → {'Left': c, 'Right': c}；c = +1 手在鏡頭這一側（五官被蓋住，或看不到臉）、
    -1 手在頭後面（手蓋住的地方五官看得到）、0 沒有重疊或判斷不出來。detail：傳入 dict 時填入判斷用的數字（記錄用）。"""
    k, s = np.asarray(k, float), np.asarray(s, float)
    out = {'Left': 0, 'Right': 0}
    if len(k) < 133:
        return out
    fk, fs = k[_WB_FACE], s[_WB_FACE]
    span = float(np.ptp(fk, axis=0).max())
    ears = float(np.linalg.norm(k[3] - k[4]))
    r = 0.5 * max(span, ears)                              # 頭在畫面上的半徑（臉的寬度的一半；背對時用兩耳）
    if r < 4:
        return out
    c = fk.mean(0) if span >= ears else 0.5 * (k[3] + k[4])
    base = float(np.percentile(fs, 75))                    # 臉上看得最清楚的那部分
    face_visible = base >= 0.5
    for side, wi, sl in _WB_ARMS:
        pts = np.vstack([k[wi][None], k[sl]])              # 手腕＋手的 21 點（被擋住時位置是模型推測的）
        if float(np.linalg.norm(pts - c, axis=1).min()) > 1.3 * r:
            continue
        hand = float(np.mean(s[sl]))
        dist = np.linalg.norm(fk[:, None] - pts[None], axis=2).min(1)
        under = dist < 0.35 * r                            # 手蓋住的臉部點
        local = ref = None
        v = 0
        if under.sum() >= 5:
            local = float(fs[under].mean())
            ref = float(np.percentile(fs[~under], 75)) if (~under).sum() >= 8 else base
            if local >= 0.5 and local >= 0.8 * ref:
                v = -1                                     # 手所在的地方五官看得到：手在頭後面
            elif local < 0.4 or local < 0.6 * ref:
                v = 1                                      # 五官被蓋住：手在臉前面
            elif hand - local > 0.15:
                v = 1
            elif local - hand > 0.15:
                v = -1
        elif face_visible:
            if hand < 0.35:
                v = -1                                     # 手和頭（頭髮、頭頂）重疊但手看不到：被頭擋住
        elif hand >= 0.5:
            v = 1                                          # 看不到臉（背對）而手看得到：手在鏡頭這一側
        out[side] = v
        if detail is not None:
            detail[side] = dict(cue=v, under=int(under.sum()), local=local, ref=ref, hand=round(hand, 2), base=round(base, 2))
    return out


def steady_cues(cues, fps=30.0, span_s=0.5):
    """整段影片（離線）：每一幀的判斷改用前後共 span_s 秒內的多數決，零星的相反判斷和空白不會讓手前後跳。
    cues：每幀 {'Left': c, 'Right': c} 或 None → 同樣長度的 list（每幀都是 dict）。"""
    n = len(cues)
    h = max(1, int(round(0.5 * span_s * fps)))
    out = [{} for _ in range(n)]
    for side in ('Left', 'Right'):
        v = np.array([int((c or {}).get(side, 0)) for c in cues], float)
        vote = np.convolve(v, np.ones(2 * h + 1), 'same') if n else v
        for i in range(n):
            out[i][side] = int(np.sign(vote[i])) if abs(vote[i]) >= 2 else int(v[i])
    return out


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _ray(p, cam):
    """從這個點指向相機的方向（沿著它移動，點在畫面上的位置不變）。"""
    if cam is None:
        return np.array([0.0, 0.0, 1.0])                    # 單相機的世界座標：相機在人的 +Z 方向
    d = _unit(np.asarray(cam, float) - p)
    return d if d is not None else np.array([0.0, 0.0, 1.0])


class Limbs:
    """有記憶的版本（即時、影片）：手在軀幹中線附近時，維持上一幀判斷的前／後，不會前後跳。"""

    def __init__(self):
        self.side = {}
        self.head_side = {}                                   # 手在頭的前面還是後面：{'Left': [判斷, 還能沿用幾幀]}

    def __call__(self, X, ok, sc, hands=None, cam=None, cue=None):
        X, hands = fix(X, ok, sc, hands, cam, cue=cue, memory=self.head_side)
        X, hands = self.torso(X, ok, hands)
        return X, hands

    def head(self, X, ok, sc, hands=None, cam=None, cue=None):
        """只做頭的部分（GEM-X 的結果用：膝蓋和軀幹交給模型自己）。"""
        X = np.array(X, float)
        hands = dict(hands or {})
        _head(X, ok, sc, hands, cam, cue, self.head_side, 0.1)
        return X, hands

    def torso(self, X, ok, hands):
        need = (P.NECK, P.HIP, P.L_SHO, P.R_SHO, P.L_HIP, P.R_HIP)
        if not all(ok[j] for j in need):
            return X, hands
        o = X[P.HIP]
        y = _unit(X[P.NECK] - o)
        side = (X[P.R_HIP] - X[P.L_HIP]) + (X[P.R_SHO] - X[P.L_SHO])
        if y is None:
            return X, hands
        z = _unit(np.cross(y, side))                               # 軀幹前方
        if z is None:
            return X, hands
        x = np.cross(y, z)
        h = float(np.linalg.norm(X[P.NECK] - o))
        a = max(0.42 * float(np.linalg.norm(X[P.R_SHO] - X[P.L_SHO])), 0.11)   # 左右半寬
        b = 0.105                                                  # 前後半厚（從中線）
        m = 0.02
        for s_, e, w, name in _ARMS:
            if not (ok[e] and ok[w]):
                continue
            pts = {'e': X[e], 'm': 0.5 * (X[e] + X[w]), 'w': X[w]}
            push = np.zeros(3)
            for key, p in pts.items():
                d = p - o
                lx, ly, lz = d @ x, d @ y, d @ z
                if not (-0.05 < ly < h) or (lx / a) ** 2 + (lz / b) ** 2 >= 1.0:
                    continue
                prev = self.side.get(name)
                front = lz >= 0 if prev is None or abs(lz) > 0.03 else prev     # 中線附近：照上一幀
                self.side[name] = front
                surf = b * np.sqrt(max(0.0, 1.0 - (lx / a) ** 2)) + m
                need_z = (surf if front else -surf) - lz
                if abs(need_z) > abs(push @ z):
                    push = need_z * z
            if np.linalg.norm(push) > 0:
                X = X.copy()
                X[e] = X[e] + push
                X[w] = X[w] + push
                if hands.get(name) is not None:
                    hands[name] = np.asarray(hands[name], float) + push
            else:
                d = X[w] - o
                if (d @ x / a) ** 2 + (d @ z / b) ** 2 > 1.6:          # 離軀幹夠遠：忘掉前／後
                    self.side.pop(name, None)
        return X, hands


GROW = 0.0                    # 設定裡「頭的大小」比預設（半徑 10cm）大多少：Avatar 的頭比較大時調大


def set_head_size(radius_m):
    """頭的半徑（公尺）。預設 0.10；Avatar 的頭比較大、手還是會穿過去時調大。"""
    global GROW
    GROW = float(np.clip(radius_m, 0.07, 0.18)) - HEAD_R


def _swing(E, W, Lf, d, palm, shifts, head):
    """前臂的候選方向：手肘不動，前臂朝「手腕沿視線 d 前後移動 δ」的方向轉（長度不變）。
    回傳 (手腕 (K,3)、檢查點相對頭中心：沿視線的前後 t (K,N)、畫面上離頭中心多遠 ρ (K,N)、距離 (K,N))。
    檢查點：前臂後半＋手腕到指尖（palm：手腕 → 指尖的向量；None 時手掌沿著前臂延伸）。"""
    fore = W[None] + shifts[:, None] * d[None] - E[None]
    fore /= np.maximum(np.linalg.norm(fore, axis=1, keepdims=True), 1e-9)
    Wc = E[None] + Lf * fore
    Tc = Wc + (palm[None] if palm is not None else HAND_LEN * Lf * fore)
    pts = np.concatenate([E[None, None] + _FORE[None, :, None] * (Wc - E[None])[:, None],
                          Wc[:, None] + _PALM[None, :, None] * (Tc - Wc)[:, None]], axis=1) - head
    t = pts @ d
    rho = np.linalg.norm(pts - t[..., None] * d, axis=2)
    return Wc, t, rho, np.linalg.norm(pts, axis=2)


def _miss(c, t, rho, dist):
    """每個候選離「可以接受」還差多少（≤ 0 就是可以）。c：-1 要在頭後面、+1 要在臉前面、0 只要不在頭裡面。
    前／後：畫面上在頭的範圍裡的點要在頭的弧面外側；範圍外的點可以越來越靠另一側（漏斗形：手滑過頭的邊緣時不會跳）。"""
    if c < 0:
        r = _R_BACK + GROW
        lim = np.where(rho < r, -np.sqrt(np.maximum(r ** 2 - rho ** 2, 0.0)), FUNNEL * (rho - r))
        return (t - lim).max(1)
    if c > 0:
        r = _R_FRONT + GROW
        lim = np.where(rho < r, np.sqrt(np.maximum(r ** 2 - rho ** 2, 0.0)), -FUNNEL * (rho - r))
        return (lim - t).max(1)
    return (_R_ANY + GROW - dist).max(1)


def _head(X, ok, sc, hands, cam, cue, memory, margin):
    """手在頭附近時決定前後，並且不讓「前臂＋手掌」穿過頭。直接改 X、hands。
    手掌：有手的點時用手腕 → 中指指尖；沒有時（匯出的動畫手掌是直的）沿著前臂延伸。
    做法：手肘不動，前臂往鏡頭或往後轉，找轉得最少、整隻手都在頭外面而且在該在的那一側的方向。
    動畫存的是骨頭的方向：上臂不變、只轉前臂，結果隨時間連續（不會有手肘突然翻到另一側的情況）。"""
    if not all(ok[j] for j in (P.L_EAR, P.R_EAR)):
        return
    head = 0.5 * (X[P.L_EAR] + X[P.R_EAR])
    face = float(np.mean([sc[j] for j in _FACE]))
    for s, e, w, side in _ARMS:
        if not (ok[s] and ok[e] and ok[w]):
            continue
        E, W = X[e], X[w]
        Lf = float(np.linalg.norm(W - E))
        if Lf < 1e-6:
            continue
        d = _ray(W, cam)
        H = hands.get(side)
        palm = None                                           # 手腕 → 指尖（有手的點時照實際的方向）
        if H is not None and len(H) >= 13 and np.all(np.isfinite(H[12])):
            v = np.asarray(H[12], float) - W
            n = float(np.linalg.norm(v))
            if n > 0.05:
                palm = v * (min(n, 0.22) / n)
        _, t, rho, dist = _swing(E, W, Lf, d, palm, _SHIFTS, head)
        i0 = len(_SHIFTS) // 2                                # δ = 0：現在的位置
        palm_rho = float(rho[i0, len(_FORE):].min())          # 畫面上手掌離頭中心最近多遠
        near = palm_rho <= NEAR_HEAD + GROW
        if not near and (dist[i0] >= _R_ANY + GROW).all():    # 畫面上手不在頭附近，前臂也沒有碰到頭
            if memory is not None and palm_rho > NEAR_HEAD + GROW + 0.05:
                memory.pop(side, None)
                memory.pop(side + '.dir', None)
            continue
        c = int(cue.get(side, 0)) if cue and near else 0
        if c == 0 and near:                                   # 沒有臉部點的資訊，或判斷不出來：臉和手腕的分數誰高
            gap = face - float(sc[w])
            c = -1 if gap > margin else (1 if gap < -margin else 0)
        if c < 0 and float((W - head) @ d) > FAR_FRONT:       # 手明顯伸向鏡頭（只是畫面上和臉重疊）：臉部的點常被「猜」出來，不要把手拉到頭後面
            c = 0
            if memory is not None:
                memory.pop(side, None)
        elif memory is not None and near:
            if c != 0:
                memory[side] = [c, HOLD]
            elif side in memory:
                c = memory[side][0]
                memory[side][1] -= 1
                if memory[side][1] <= 0:
                    del memory[side]
        miss = _miss(c, t, rho, dist)
        if _DEBUG is not None:
            _DEBUG[side] = dict(c=c, miss=miss, t=t, rho=rho, dist=dist)
        if miss[i0] <= 1e-9:                                  # 現在的位置就可以
            if memory is not None and c == 0 and miss[i0] < -0.03:
                memory.pop(side + '.dir', None)
            continue
        good = miss <= 1e-9
        cost = np.abs(_SHIFTS)
        if c == 0 and memory is not None and memory.get(side + '.dir'):
            cost = cost + np.where(np.sign(_SHIFTS) == -memory[side + '.dir'], 0.08, 0.0)     # 不知道前後：上一幀往哪邊轉，這一幀優先同一邊
        if good.any():
            k = int(np.argmin(np.where(good, cost, np.inf)))
            fine = _SHIFTS[k] - np.sign(_SHIFTS[k]) * np.linspace(0.01, 0.0, 11)              # 在這一格裡找剛好不碰到的位置（1mm）
        elif miss.min() < miss[i0] - 0.01:                    # 找不到完全符合的方向：盡量接近
            fine = _SHIFTS[[int(np.argmin(miss))]]
        else:
            continue
        Wf, tf, rf, df = _swing(E, W, Lf, d, palm, fine, head)
        mf = _miss(c, tf, rf, df)
        k = int(np.argmax(mf <= 1e-9)) if (mf <= 1e-9).any() else len(fine) - 1
        if memory is not None:
            if c == 0:
                memory[side + '.dir'] = float(np.sign(fine[k]))
            memory['moved'] = memory.get('moved', 0) + 1
        if H is not None:
            hands[side] = np.asarray(H, float) + (Wf[k] - W)
        X[w] = Wf[k]


def fix(X, ok, sc, hands=None, cam=None, margin=0.1, cue=None, memory=None):
    """cue：head_cue 的結果（沒有就只看分數）；memory：有記憶的呼叫端傳入的 dict（判斷不出來時沿用上一次）。"""
    X = np.array(X, float)
    hands = dict(hands or {})
    _head(X, ok, sc, hands, cam, cue, memory, margin)
    # 膝蓋不會往後折
    side = (X[P.R_HIP] - X[P.L_HIP]) + 0.3 * (X[P.R_SHO] - X[P.L_SHO])
    f = _unit(np.cross([0.0, 1.0, 0.0], side) * np.array([1.0, 0.0, 1.0]))
    if f is not None:
        for h, k, a in _LEGS:
            if not (ok[h] and ok[k] and ok[a]):
                continue
            u = _unit(X[a] - X[h])
            if u is None:
                continue
            on = X[h] + ((X[k] - X[h]) @ u) * u
            b = (X[k] - on) @ f
            if b < -0.01:
                X[k] = X[k] - b * f
    return X, hands
