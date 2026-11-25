"""即時追蹤每一幀的計算：模型輸出（畫面位置＋相對深度）→ 世界座標的骨架 → 骨架約束 → 去抖動。

引擎負責來源、推論、輸出和執行緒；這裡只有數學，測試直接用同一份程式量精度（tests/test_accuracy.py）。
"""
import numpy as np

import limbs as LB
import pose as P
import temporal as TM
from lift import Grounder, Lifter
from precise import Precise, Smoother
from skeleton import Skeleton, confidence
from steady import DepthSteady, HeadSteady, flat_feet

SMOOTH_MAX_S = 0.3          # 「平滑」拉到最右邊時，前後各看幾秒


class Tracker:
    """tracker(t, k, z, sc, hands, zt) → 延遲後的 (t, X (26,3), ok, hands) 或 None。

    k (26,2) 畫素、z (26,) 相對深度、sc (26,) 分數；hands：(手的 2D (42,2), 深度 (42,), 分數 (42,)) 或 None；
    zt：時間序列模型給的深度 (17,) 或 None。t：這一幀拍攝的時間（秒）。"""

    def __init__(self, size, hfov_deg, height, thr, precise=False, decide_s=0.8, smoothness=0.85, steady=True):
        self.thr = thr
        self.steady = steady
        self.lifter = Lifter(size, hfov_deg, height, thr)
        self.ground = Grounder()
        self.sk = Skeleton(height, learn=True)
        self.limbs = LB.Limbs()
        self.depth = DepthSteady(thr)                      # 相對深度的穩定化（真實畫面上深度很吵）
        self.head = HeadSteady(thr, height)
        self.zt_bias = None                                # 時間序列模型的深度 − 單張模型的深度（每個關節，慢慢更新）
        # 精準模式：判斷左右／前後＋去抖動；關掉時只去抖動。兩者都以每一幀為中心看前後（不會落後），代價是延遲
        if precise:
            self.post = Precise(60.0, decide_s=float(decide_s), smooth_s=0.3)
            self.delay_s = self.post.delay_s + 0.25
        else:
            self.post = Smoother(SMOOTH_MAX_S * float(np.clip(smoothness, 0.0, 1.0)))
            self.delay_s = self.post.delay_s + 0.12

    def solve(self, t, k, z, sc, hands=None, zt=None):
        """這一幀的骨架（還沒去抖動）：(X, ok, hands {'Left'|'Right': (21,3)}, 相機位置) 或 None。"""
        hz = None if hands is None else hands[1]
        if zt is not None:
            z2 = TM.apply_depth(z, zt)
        elif self.zt_bias is not None and self.steady:
            z2 = z + self.zt_bias                         # 時間序列模型這一幀來不及：用它平常和單張模型的差（不然兩種深度切來切去，姿勢會跳）
        else:
            z2 = None
        if z2 is not None:
            if zt is not None and np.isfinite(z2).all() and np.isfinite(z).all():
                d = z2 - z
                self.zt_bias = d if self.zt_bias is None else self.zt_bias + 0.05 * (d - self.zt_bias)
            if hands is not None:
                dl, dr = TM.wrist_shift(z, z2)
                hz = np.concatenate([hands[1][:21] + dl, hands[1][21:] + dr])
            z = z2
        if self.steady:                                   # 深度的穩定化（時間序列模型給的深度也一樣要過）
            z0 = z
            z = self.depth(k, z, sc, t)
            if hands is not None:                         # 手的點跟著手腕的深度一起移
                hz = np.concatenate([hz[:21] + (z[P.L_WRI] - z0[P.L_WRI]), hz[21:] + (z[P.R_WRI] - z0[P.R_WRI])])
        lifter = self.lifter
        X, ok = lifter(k, z, sc, t)
        if X is None:
            return None
        hp = {}                                           # 手的點只用來算手掌的方向（相對位置），不需要跟著身體平移
        if hands is not None:
            hk, _, hs = hands
            for side, o in (('Left', 0), ('Right', 21)):
                if min(hs[o], hs[o + 5], hs[o + 9], hs[o + 17]) > self.thr:
                    hp[side] = lifter.extra(hk[o:o + 21], hz[o:o + 21])
        X = self.sk.apply(X, ok, confidence(sc, self.thr), use_toes=False, cam=(0.0, lifter.last_h, 0.0))
        if self.steady:
            X = flat_feet(X, ok, (0.0, lifter.last_h, 0.0), lifter.height)       # 腳尖、腳跟：深度很吵，放回「腳掌放平」
        X = self.ground(X, ok)
        cam = np.array([0.0, lifter.last_h - (self.ground.off or 0.0), 0.0])
        if self.steady:
            X = self.head(X, ok, k, sc, cam, lifter.R, t)     # 頭的方向：看臉上的點在畫面上的位置
        X, hp = self.limbs(X, ok, sc, hp, cam, getattr(hands, 'cue', None))
        return X, ok, hp, cam

    def __call__(self, t, k, z, sc, hands=None, zt=None):
        r = self.solve(t, k, z, sc, hands, zt)
        if r is None:
            return None
        X, ok, hp, cam = r
        return self.post.push(t, X, sc, ok, hp, cam)
