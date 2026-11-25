"""直接輸出的 .anim：Humanoid 肌肉曲線、根的位置／旋轉都在，數值合理。執行：python tests/test_anim.py"""
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anim as A  # noqa: E402
import motion  # noqa: E402


def test_anim():
    X = np.stack([motion.person(i / 30.0) for i in range(90)])
    cv = A.curves(X, 30.0)
    assert {'RootT.x', 'RootT.y', 'RootT.z', 'RootQ.w'} <= set(cv)
    mus = [k for k in cv if not k.startswith('Root')]
    assert len(mus) >= 40 and all(k in A.MUSCLE_NAMES for k in mus)
    for k, v in cv.items():
        assert np.all(np.isfinite(v)), k
        if not k.startswith('Root'):
            assert np.all(np.abs(v) <= 1.0 + 1e-6), k
    y = np.median(cv['RootT.y'])
    q = np.stack([cv['RootQ.' + a] for a in 'xyzw'], 1)
    assert 0.9 < y < 1.1, y                                  # 站著時 bodyPosition.y ≈ 1（標準骨架）
    assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4)
    assert np.all(np.sum(q[1:] * q[:-1], axis=1) > 0)          # 四元數連續
    path = A.write(os.path.join(tempfile.mkdtemp(), 'walk.anim'), X, 30.0, 'walk')
    text = open(path, encoding='utf-8').read()
    assert text.startswith('%YAML 1.1') and 'm_Name: walk' in text and 'classID: 95' in text
    assert text.count('attribute: ') == 2 * len(cv)            # m_FloatCurves 和 m_EditorCurves
    keys = len(re.findall(r'time: ', text)) // 2
    assert 2 * len(cv) <= keys < len(cv) * len(X)                # 去掉了多餘的關鍵影格
    print(f'  .anim：{len(cv)} 條曲線、{len(X)} 幀，RootT.y {y:.3f}，{os.path.getsize(path) // 1024} KB')


def test_missing_joints():
    """整段都沒拍到耳朵（背對鏡頭）、中間有幾幀沒有人：有些骨頭的旋轉算不出來，沿用上一幀；以前 .anim 會直接出錯。"""
    import pose as P
    X = np.stack([motion.person(i / 30.0) for i in range(60)])
    X[:, [P.L_EAR, P.R_EAR]] = 0.0
    X[20:24] = 0.0
    cv = A.curves(X, 30.0)
    assert all(np.all(np.isfinite(v)) for v in cv.values())
    assert A.write(os.path.join(tempfile.mkdtemp(), 'back.anim'), X, 30.0, 'back')
    import bvh
    assert 'Frames: 60' in bvh.to_bvh(X, 30.0)


if __name__ == '__main__':
    test_anim()
    test_missing_joints()
    print('全部通過')
