"""平滑濾波：位置用 One Euro Filter，旋轉用四元數插值。"""
import math

import numpy as np


class OneEuro:
    """One Euro Filter（向量版）。慢動作時平滑、快動作時跟得上。"""

    def __init__(self, min_cutoff=1.5, beta=0.5, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None
        self.tau = 0.0          # 這次濾波造成的延遲（秒），給預測補償用

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-4)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.tau = 1.0 / (2 * math.pi * cutoff)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x

    def reset(self):
        self.x = None


class QuatSmooth:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.q = None

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        if self.q is None:
            self.q = q
            return q
        if np.dot(self.q, q) < 0:     # 走短的那條路
            q = -q
        self.q = (1 - self.alpha) * self.q + self.alpha * q
        self.q /= np.linalg.norm(self.q)
        return self.q


def _qangle(a, b):
    """兩個單位四元數之間的旋轉角度（度）。"""
    d = abs(float(np.dot(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


class RotStab:
    """旋轉的穩定化（實驗性）：取代固定比例的 QuatSmooth。全部用時間（秒）定義，不依賴固定幀率。

    - 依轉動速度調整：靜止時多平滑（alpha_rest 是 30 fps 時每幀的比例，依實際間隔換算），轉得快（>= fast_dps）時直接跟上。
      轉速用「低通後的原始旋轉」每秒的移動量估計（低通的係數也依實際間隔換算），雜訊會互相抵消，不會被誤認成轉動。
    - 異常的跳動：相鄰兩次原始旋轉的轉速超過 gate_dps（例如 30 fps 時一幀 45°）時先維持上一個輸出；
      新的方向持續 confirm_s 秒（而且至少 2 次）都差不多（agree_deg 內）才接受，直接跳過去。
      這只擋短暫的跳動；持續的錯誤方向在 confirm_s 後仍會通過——這不是修正左右認錯。真的很快的動作或掉幀也可能被擋一下（有延遲成本）。
    - 擋的時間有上限：連續擋超過 max_hold_s 秒（例如原始方向每幀來回亂跳、一直確認不了）就直接接受最新的原始方向，
      之後像原本的濾波一樣跟著走，不會把舊的方向凍結在畫面上。forced 記錄這種情況的次數。
    """

    def __init__(self, alpha_rest=0.3, slow_dps=30.0, fast_dps=300.0, gate_dps=1350.0, confirm_s=0.066, agree_deg=30.0, speed_lp=0.35, max_hold_s=0.3):
        self.alpha_rest, self.slow, self.fast, self.speed_lp = alpha_rest, slow_dps, fast_dps, speed_lp
        self.gate, self.confirm_s, self.agree, self.max_hold = gate_dps, confirm_s, agree_deg, max_hold_s
        self.reset()

    def reset(self):
        self.q = None
        self.q_raw = None
        self.q_lp = None
        self.t = None
        self.w = 0.0
        self.pend = None
        self.pend_t0 = None
        self.pend_n = 0
        self.held = 0           # 累計擋下的次數（診斷用）
        self.forced = 0         # 擋太久而強制接受的次數（診斷用）
        self.hold_t0 = None     # 這一段連續擋下從什麼時候開始

    def _accept(self, q):
        self.q, self.q_raw, self.q_lp, self.w = q, q, q, 0.0
        self.pend, self.pend_t0, self.pend_n = None, None, 0
        self.hold_t0 = None
        return self.q

    def __call__(self, q, t):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-9:
            return self.q           # 無效的輸入：不改變（沒有輸出過時是 None）
        q = q / n
        if self.q is None:
            self.t = t
            return self._accept(q)
        dt = max(t - self.t, 1e-3)
        self.t = t
        step = _qangle(self.q_raw, q)
        if step / dt > self.gate or (self.pend is not None and _qangle(self.pend, q) < self.agree):
            if self.pend is not None and _qangle(self.pend, q) < self.agree:
                self.pend_n += 1
            else:
                self.pend, self.pend_t0, self.pend_n = q, t, 1
            if self.pend_n >= 2 and t - self.pend_t0 >= self.confirm_s - 1e-6:
                return self._accept(q)
            if self.hold_t0 is None:
                self.hold_t0 = t
            if t - self.hold_t0 >= self.max_hold - 1e-6:
                self.forced += 1                      # 擋太久：直接接受最新的原始方向
                return self._accept(q)
            self.held += 1
            return self.q
        self.pend, self.pend_t0, self.pend_n = None, None, 0
        self.hold_t0 = None
        self.q_raw = q
        qq = q if np.dot(self.q_lp, q) >= 0 else -q
        b = 1.0 - (1.0 - self.speed_lp) ** (dt * 30.0)     # speed_lp 是 30 fps 時每幀的係數，依實際間隔換算
        lp = (1 - b) * self.q_lp + b * qq
        lp /= np.linalg.norm(lp)
        self.w = _qangle(self.q_lp, lp) / dt      # 穩定轉動時，低通後每幀的移動量 ≈ 真正的轉動量
        self.q_lp = lp
        a30 = self.alpha_rest + (1.0 - self.alpha_rest) * min(1.0, max(0.0, (self.w - self.slow) / (self.fast - self.slow)))
        a = 1.0 - (1.0 - a30) ** (dt * 30.0)      # 依實際間隔換算
        if np.dot(self.q, q) < 0:
            q = -q
        self.q = (1 - a) * self.q + a * q
        self.q /= np.linalg.norm(self.q)
        return self.q
