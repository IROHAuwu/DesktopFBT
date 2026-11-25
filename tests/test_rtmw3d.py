"""用真正的 RTMW3D 模型檢查 3D 輸出的格式與方向（只在 GitHub Actions 上跑，需要網路）。
圖片與正確答案來自 mmpose 的測試資料（Human3.6M 的 4 張圖，有相機參數與 3D 座標）。
執行：DESKTOPFBT_REAL_MODEL=1 python tests/test_rtmw3d.py
"""
import io
import os
import pickle
import sys
import tempfile
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
TMP = tempfile.mkdtemp()
import paths  # noqa: E402
paths.base_dir = lambda: TMP

import models  # noqa: E402
import pose as P  # noqa: E402
from lift import Pose3DEstimator, camera_points  # noqa: E402

RAW = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/h36m/'
# H36M 17 點 → Halpe26
H36 = {1: P.R_HIP, 2: P.R_KNEE, 3: P.R_ANK, 4: P.L_HIP, 5: P.L_KNEE, 6: P.L_ANK,
       11: P.L_SHO, 12: P.L_ELB, 13: P.L_WRI, 14: P.R_SHO, 15: P.R_ELB, 16: P.R_WRI}


def note(msg):
    print(msg)
    print(f'::notice title=RTMW3D::{msg}')


def get(name):
    with urllib.request.urlopen(RAW + name, timeout=60) as r:
        return r.read()


def main():
    if os.environ.get('DESKTOPFBT_REAL_MODEL') != '1':
        print('略過（需要網路與真的模型）')
        return
    ann = np.load(io.BytesIO(get('test_h36m_body3d.npz')), allow_pickle=True)
    cams = pickle.loads(get('cameras.pkl'), encoding='latin1')
    paths_ = models.ensure_pose3d()
    est = Pose3DEstimator(paths_, 'cpu', 1)
    corr, signs, mpjpe, depth_err = [], [], [], []
    for i, name in enumerate(ann['imgname']):
        img = cv2.imdecode(np.frombuffer(get(name), np.uint8), cv2.IMREAD_COLOR)
        est.bbox = None
        k, z, s = est(img)
        if k is None:
            note(f'{name}: 沒偵測到人')
            continue
        S = ann['S'][i][:, :3]
        gt_rel = S[:, 2] - S[0, 2]
        ours = np.array([z[h] for h in H36.values()]) - 0.5 * (z[P.L_HIP] + z[P.R_HIP])
        gt = np.array([gt_rel[j] for j in H36])
        c = float(np.corrcoef(ours, gt)[0, 1])
        corr.append(c)
        signs.append(float(np.sign(np.polyfit(gt, ours, 1)[0])))
        subj, cam_id = name.split('/')[0], name.split('.')[1].split('/')[0]
        cp = cams[(subj, cam_id)]
        f, cc = np.array(cp['f']).ravel(), np.array(cp['c']).ravel()
        K = np.array([[f[0], 0, cc[0]], [0, f[1], cc[1]], [0, 0, 1.0]])
        height = 1.0 / 0.491 * float(np.mean([np.linalg.norm(S[1] - S[3]), np.linalg.norm(S[4] - S[6])])) * 1.0
        Xc, ok, zr = camera_points(k, z, s, K, height, 0.3)
        m = [h for h in H36.values() if ok[h]]
        gj = [j for j, h in H36.items() if ok[h]]
        root_o = 0.5 * (Xc[P.L_HIP] + Xc[P.R_HIP])
        root_g = 0.5 * (S[1] + S[4])
        e = np.linalg.norm((Xc[m] - root_o) - (S[gj] - root_g), axis=1)
        mpjpe.append(float(np.mean(e)))
        depth_err.append(float(root_o[2] - root_g[2]))
        note(f'{name}: z 相關 {c:.2f}、斜率符號 {signs[-1]:+.0f}、相對骨盆誤差 {mpjpe[-1] * 100:.1f}cm、'
             f'骨盆深度 {root_o[2]:.2f}m（實際 {root_g[2]:.2f}）、分數 {s[:17].mean():.2f}、'
             f'z 範圍 {z.min():.2f}~{z.max():.2f}')
    note(f'平均 z 相關 {np.mean(corr):.2f}、相對誤差 {np.mean(mpjpe) * 100:.1f}cm、深度誤差 '
         f'{np.mean(np.abs(depth_err)) * 100:.0f}cm')
    assert len(corr) >= 3
    assert np.mean(signs) > 0, 'z 方向和預期相反'
    assert np.mean(mpjpe) < 0.2
    print('全部通過')


if __name__ == '__main__':
    main()
