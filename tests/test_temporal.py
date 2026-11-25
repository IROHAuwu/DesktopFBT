"""時間序列 3D（MotionBERT）：重新取樣、即時交出順序與不卡住；有模型時量深度的誤差。執行：python tests/test_temporal.py"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import motion as M  # noqa: E402
import pose as P  # noqa: E402
import temporal as TM  # noqa: E402

CAM = np.array([0.3, 1.1, -3.4])


def look_at(pos, target):
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f])


R = look_at(CAM, np.array([0, 0.9, 0]))
F, W, H = 600.0, 1280, 720


def frame(t, noise=0.0, rng=None):
    """回傳 (k 畫素, z 相對骨盆的深度（真值）, s)。"""
    J = M.person(t)
    Xc = (R @ (J - CAM).T).T
    k = F * Xc[:, :2] / Xc[:, 2:3] + [W / 2, H / 2]
    if noise and rng is not None:
        k = k + rng.normal(0, noise, k.shape)
    return k, Xc[:, 2] - Xc[P.HIP, 2], np.full(P.NUM_JOINTS, 0.9)


def test_resample():
    ts = np.array([0.0, 0.1, 0.2, 1.0])
    seq = np.zeros((4, 17, 3))
    seq[:, :, 0] = ts[:, None] * 10
    seq[:, :, 2] = 1.0
    out = TM.resample(ts, seq, np.array([0.05, 0.15, 0.6, 1.6]))
    assert abs(out[0, 0, 0] - 0.5) < 1e-9 and abs(out[1, 0, 0] - 1.5) < 1e-9
    assert out[2, 0, 2] == 0.0 and out[3, 0, 2] == 0.0           # 中間缺了 0.8 秒、超出範圍：分數 0


class FakeModel:
    """回傳固定深度；可以設定很慢。"""
    device = 'cpu'
    T = TM.T
    h36m = staticmethod(TM.to_h36m)

    def __init__(self, delay=0.0):
        self.delay = delay
        self.ms = 0.0
        self.calls = 0

    def run(self, seqs, tta=False):
        time.sleep(self.delay)
        self.calls += 1
        out = []
        for s in seqs:
            Y = np.zeros((TM.T, 17, 3))
            Y[..., :2] = 0.0
            Y[:, 4, 1], Y[:, 5, 1], Y[:, 6, 1] = 0, 0.245, 0.491     # 腿：讓尺度 = 1
            Y[:, 1, 1], Y[:, 2, 1], Y[:, 3, 1] = 0, 0.245, 0.491
            Y[:, 11, 1], Y[:, 12, 1], Y[:, 13, 1] = 0, 0.186, 0.332
            Y[:, 14, 1], Y[:, 15, 1], Y[:, 16, 1] = 0, 0.186, 0.332
            Y[..., 2] = 0.123
            Y[:, 0, 2] = 0.0
            out.append(Y)
        return out


def test_live_order():
    """幀照順序交出、換成模型的深度；模型很慢時等太久的幀照原樣交出（不卡住）。"""
    for delay, expect_used in ((0.0, True), (3.0, False)):
        m = FakeModel(delay)
        tl = TM.Live(m, 1.0, lag=0.2, min_interval=0.02, give_up=0.6)
        t0 = time.perf_counter()
        got = []
        for i in range(150):
            t = t0 + i / 60
            while time.perf_counter() < t:
                time.sleep(0.001)
            k, z, s = frame(i / 60)
            if i % 37 == 5:
                tl.skip(t, (None, None, None, None))
            else:
                tl.push(t, k, s, (k, z, s, None))
            got += tl.pull(time.perf_counter())
        end = time.perf_counter() + 1.5
        while time.perf_counter() < end:
            got += tl.pull(time.perf_counter())
            time.sleep(0.02)
        tl.stop()
        ts = [g[0] for g in got]
        assert len(got) == 150 and ts == sorted(ts), (len(got), delay)
        used = [g for g in got if g[2] is not None]
        print(f'  模型延遲 {delay}s：換深度 {len(used)} / 150 幀')
        if expect_used:
            assert len(used) > 100 and all(abs(g[2][3] - 0.123) < 1e-6 for g in used)
        else:
            assert not used


def test_apply_depth():
    k, z, s = frame(1.0)
    zt = np.arange(17) * 0.01
    z2 = TM.apply_depth(z, zt)
    assert z2[P.L_KNEE] == zt[5] and z2[P.HIP] == zt[0]
    assert abs((z2[P.L_HEEL] - z2[P.L_ANK]) - (z[P.L_HEEL] - z[P.L_ANK])) < 1e-12


def test_real_model():
    for name, (_, kind) in TM.MODELS.items():
        path = TM.model_path(name)
        if not path:
            print(f'  skip {name}：沒有模型檔')
            continue
        model = TM.Model(path, 'cpu', kind)
        model.size = (W, H)
        _check(name, model)


def _check(name, model):
    rng = np.random.default_rng(0)
    fps = 30
    times = [i / fps for i in range(10 * fps)]
    frames, truth = [], []
    for t in times:
        k, z, s = frame(t, 2.0, rng)
        frames.append((k, s))
        truth.append(z)
    zts = TM.sequence(model, times, frames, M.HEIGHT, tta=False)
    idx = [i for i, z in enumerate(zts) if z is not None]
    assert len(idx) > 0.9 * len(times)
    J = [j for j in TM.H36M if j is not None]
    est = np.array([[zts[i][TM.H36M.index(j)] for j in J] for i in idx])
    tru = np.array([[truth[i][j] for j in J] for i in idx])
    err = np.abs(est - tru)
    flipped = np.abs(-est - tru)
    print(f'  {name}：深度誤差中位數 {np.median(err) * 100:.1f} cm（正負號反過來會是 {np.median(flipped) * 100:.1f} cm），'
          f'90% {np.percentile(err, 90) * 100:.1f} cm')
    assert np.median(err) < np.median(flipped) and np.median(err) < 0.08


def main():
    for f in (test_resample, test_apply_depth, test_live_order, test_live_fallback, test_dml_safe, test_real_model):
        f()
        print('ok', f.__name__)
    print('全部通過')


def test_dml_safe():
    """DirectML 用的版本：沒有 5 維的 MatMul／Softmax，結果和原本完全相同。"""
    import onnx
    import onnxruntime as ort
    for name in ('agformer_s', 'motionbert'):
        src = TM.model_path(name)
        m, n = TM.dml_safe(onnx.load(src))
        assert n > 0
        a = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
        b = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
        T = a.get_inputs()[0].shape[1]
        x = np.random.default_rng(0).uniform(-1, 1, (2, T, 17, 3)).astype(np.float32)
        d = float(np.abs(a.run(None, {'x': x})[0] - b.run(None, {'x': x})[0]).max())
        print(f'  {name}：攤平 {n} 個 5 維運算，差異 {d:.2e}')
        assert d < 1e-5


def test_live_fallback():
    """顯示卡上的模型一直失敗 → 換成備用模型繼續（不會一直 0%）。"""
    import time

    class Bad:
        T, kind, device, ms, size = 27, 'screen', 'dml', 0.0, (1920, 1080)

        def h36m(self, k, s):
            return np.concatenate([np.asarray(k, float)[:17], np.asarray(s, float)[:17, None]], 1)

        def run(self, seqs, tta=False):
            raise RuntimeError('boom')

    class Good(Bad):
        device = 'cpu'

        def run(self, seqs, tta=False):
            self.ms = 1.0
            return [np.random.default_rng(0).normal(0, 0.1, (self.T, 17, 3))]

    got = []
    tl = TM.Live(Bad(), 1.7, lag=0.1, min_interval=0.0, fallback=lambda e: got.append(e) or Good())
    t0 = time.perf_counter()
    k = np.random.default_rng(1).uniform(100, 500, (26, 2))
    s = np.ones(26)
    used = 0
    while time.perf_counter() - t0 < 3.0 and not used:
        tl.push(time.perf_counter(), k, s, None)
        used += sum(z is not None for _, _, z in tl.pull(time.perf_counter()))
        time.sleep(1 / 60)
    tl.stop()
    assert got and 'boom' in got[0] and isinstance(tl.model, Good) and used > 0
    print('  備用模型：', got[0])

    tl = TM.Live(Bad(), 1.7, lag=0.1, min_interval=0.0)          # 沒有備用：放棄，之後的幀馬上照原樣交出
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0 and not tl.dead:
        tl.push(time.perf_counter(), k, s, None)
        time.sleep(1 / 60)
    tl.push(time.perf_counter(), k, s, None)
    out = tl.pull(time.perf_counter())
    tl.stop()
    assert tl.dead and 'boom' in tl.last_error and out and all(z is None for _, _, z in out)


if __name__ == '__main__':
    main()
