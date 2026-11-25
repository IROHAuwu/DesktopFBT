"""影片模式測試：用假的 3D 姿勢模型（照著已知動作投影）跑「即時輸出」和「匯出 BVH」，
再把 BVH 讀回來和真實動作比較。Windows 上另外測試視窗擷取。執行：python tests/test_video.py
"""
import os
import socket
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

TMP = tempfile.mkdtemp()
import paths  # noqa: E402
paths.base_dir = lambda: TMP

import bvh  # noqa: E402
import engine as E  # noqa: E402
import motion as M  # noqa: E402
import pose as P  # noqa: E402
import video as V  # noqa: E402
from pythonosc.dispatcher import Dispatcher  # noqa: E402
from pythonosc.osc_server import BlockingOSCUDPServer  # noqa: E402

W, H, FPS, SECONDS, HFOV = 640, 360, 30, 24, 60.0
CAM = np.array([0.4, 1.15, -3.6])          # 人面向 -z，相機在前方 → 拍到正面


def look_at(pos, target):
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f])


RCW = look_at(CAM, np.array([0, 0.9, 0]))
F = 0.5 * W / np.tan(np.radians(HFOV) / 2)
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def encode(fid):
    img = np.full((H, W, 3), 40, np.uint8)
    for b in range(16):
        if fid >> b & 1:
            x = 10 + b * 38
            img[10:40, x:x + 30] = 255
    return img


def decode(img):
    fid = 0
    for b in range(16):
        x = 10 + b * 38
        if img[15:35, x + 5:x + 25].mean() > 150:
            fid |= 1 << b
    return fid


def make_video(path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for i in range(FPS * SECONDS):
        vw.write(encode(i))
    vw.release()


RNG = np.random.default_rng(1)


class FakeEst3D:
    def __init__(self, *a, **k):
        self.bbox = None
        self.n = 0
        self.freq = 8
        self.tta = False
        self.precision = 'fp32'
        self.device = 'cpu'
        self.ms = 1.0

    def batch(self, imgs):
        return [self(im) + (None,) for im in imgs]

    def __call__(self, img):
        fid = decode(img)
        J = M.person(fid / FPS)
        Xc = (RCW @ (J - CAM).T).T
        uv = (K @ Xc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        z = Xc[:, 2] - 0.5 * (Xc[P.L_HIP, 2] + Xc[P.R_HIP, 2])
        return (uv + RNG.normal(0, 1.5, uv.shape), z + RNG.normal(0, 0.03, z.shape),
                np.full(P.NUM_JOINTS, 0.9))


E.Pose3DEstimator = FakeEst3D
E.models.pose3d_ready = lambda: True
E.models.ensure_pose3d = lambda progress=None: {}
E.P.resolve_device = lambda pref: 'cpu'


def wait_idle(eng, timeout):
    t = time.time()
    while time.time() - t < timeout:
        if eng.get_state(False)['mode'] == 'idle':
            return True
        time.sleep(0.2)
    return False


def kabsch_y(A, B):
    """只允許繞 y 軸旋轉 + 平移，把 A 對齊到 B。"""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    num = np.sum(a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2])
    den = np.sum(a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2])
    th = np.arctan2(num, den)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return lambda X: (X - ma) @ R.T + mb


def check_package(path, motions):
    import tarfile
    assert path and os.path.exists(path), path
    with tarfile.open(path, 'r:gz') as tar:
        names = tar.getnames()
        paths = {}
        for n in names:
            if n.endswith('/pathname'):
                paths[tar.extractfile(n).read().decode('utf-8')] = n.split('/')[0]
        print('unitypackage', os.path.basename(path), sorted(paths))
        assert 'Assets/DesktopFBT/Editor/DesktopFBTBvhImporter.cs' in paths
        for m in motions:
            g = paths['Assets/DesktopFBT/Motions/' + m]
            assert f'{g}/asset' in names and f'{g}/asset.meta' in names
            meta = tar.extractfile(f'{g}/asset.meta').read().decode()
            assert f'guid: {g}' in meta
        cs = tar.extractfile(paths['Assets/DesktopFBT/Editor/DesktopFBTBvhImporter.cs'] + '/asset').read()
        assert b'ScriptedImporter' in cs


def test_window_capture():
    import tkinter as tk
    root = tk.Tk()
    root.title('DesktopFBT capture test')
    root.geometry('420x300+50+50')
    c = tk.Canvas(root, width=420, height=300, bg='#20c040', highlightthickness=0)
    c.pack()
    root.update()
    time.sleep(0.5)
    root.update()
    wins = V.list_windows()
    hit = [w for w in wins if w['title'] == 'DesktopFBT capture test']
    print('windows', len(wins), hit)
    assert hit, wins
    img = V.grab_window(hit[0]['id'])
    print('capture', img.shape, img[150, 200])
    b, g, r = [int(v) for v in img[img.shape[0] // 2, img.shape[1] // 2]]
    assert g > 150 and r < 80 and b < 120, (b, g, r)
    src = V.WindowSource(hit[0]['id'])
    fr = src.read(-1)
    assert fr is not None and fr[0].shape[0] > 100
    src.stop()
    root.destroy()


def main():
    path = os.path.join(TMP, 'dance.mp4')
    make_video(path)
    got = []
    d = Dispatcher()
    d.map('/avatar/parameters/*', lambda a, *v: got.append((a, v)))
    srv = BlockingOSCUDPServer(('127.0.0.1', 9000), d)          # 一個執行緒收（每個封包開執行緒會來不及、掉封包）
    srv.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    eng = E.Engine(preload=False)
    # 假模型給的是接近真值的深度：關掉時間序列模型才量得出這一段流程本身的誤差（模型另外在 test_temporal 測）。
    # 運算裝置指定 CPU：「自動」在 Windows 會先向系統查 AMD 加速，CI 上要等 4 分鐘才逾時（這個測試以前因此在 CI 上一直沒過）
    eng.update_config({'video': {'hfov_deg': HFOV, 'height_m': M.HEIGHT, 'loop': True}, 'live': {'temporal': False},
                       'pose': {'device': 'cpu'}})

    # 即時：影片的人 → Avatar 參數（OSC）
    eng.start_live({'kind': 'file', 'path': path})
    time.sleep(8)
    st = eng.get_state()
    print('live', st['mode'], st['phase'], st['fps'], st['src_fps'], st['job'], st['osc'], st['error'])
    assert st['mode'] == 'live' and st['preview'].startswith('data:image/jpeg')
    assert st['job']['position'] > 1.0 and st['fps'] > 20
    eng.stop()
    assert wait_idle(eng, 5)
    addrs = {a for a, _ in got}
    print('OSC', len(got), len(addrs))
    for k in ('YawS', 'YawC', 'TiltX', 'Y', 'm21', 'm21_0'):
        assert f'/avatar/parameters/YFBT/{k}' in addrs, k
    ys = [v[0] for a, v in got if a == '/avatar/parameters/YFBT/Y']
    assert len(ys) > 60 and all(-1.0 <= y <= 1.0 for y in ys)

    # 匯出 BVH
    eng.start_export(path)
    assert wait_idle(eng, 240)
    st = eng.get_state(False)
    print('export', st['error'], st['job'])
    assert st['error'] is None and st['job']['export'] and os.path.exists(st['job']['export'])
    text = open(st['job']['export'], encoding='utf-8').read()
    fk = bvh.forward_kinematics(text)
    n = len(fk['Hips'])
    assert n == FPS * SECONDS, n
    truth = np.array([M.person(i / FPS) for i in range(n)])
    names = {'Hips': P.HIP, 'LeftLeg': P.L_KNEE, 'RightLeg': P.R_KNEE, 'LeftFoot': P.L_ANK,
             'RightFoot': P.R_ANK, 'LeftHand': P.L_WRI, 'RightHand': P.R_WRI}
    A = np.concatenate([fk[nm] for nm in names])
    B = np.concatenate([truth[:, j] for j in names.values()])
    tf = kabsch_y(A, B)
    for nm, j in names.items():
        e = np.linalg.norm(tf(fk[nm]) - truth[:, j], axis=1)
        print(f'  {nm:10s} 誤差中位數 {np.median(e) * 100:5.1f} cm  90% {np.percentile(e, 90) * 100:5.1f} cm')
    e_all = np.linalg.norm(tf(A) - B, axis=1)
    rel = []
    for nm, j in names.items():
        if nm == 'Hips':
            continue
        a = fk[nm] - fk['Hips']
        b = truth[:, j] - truth[:, P.HIP]
        rel.append(np.linalg.norm(tf(a + 0) - tf(np.zeros_like(a)) - b, axis=1))
    rel = np.concatenate(rel)
    foot_h = np.minimum(fk['LeftFoot'][:, 1], fk['RightFoot'][:, 1])
    print(f'整體誤差中位數 {np.median(e_all) * 100:.1f} cm；相對骨盆 {np.median(rel) * 100:.1f} cm；'
          f'較低那隻腳踝高度中位數 {np.median(foot_h) * 100:.1f} cm（實際 {M.ANKLE_H * 100:.1f}）')
    assert np.median(rel) < 0.06 and np.median(e_all) < 0.15
    assert abs(np.median(foot_h) - M.ANKLE_H) < 0.04
    pkg = st['job']['unity']
    check_package(pkg, ['dance.anim'])

    # 直接放進 Unity 專案：選到專案裡的 Assets 也要能找到專案根目錄
    proj = os.path.join(TMP, 'My project')
    os.makedirs(os.path.join(proj, 'Assets'))
    os.makedirs(os.path.join(proj, 'ProjectSettings'))
    assert eng.set_unity_project(os.path.join(proj, 'Assets')) == proj
    assert eng.set_unity_project(TMP) is None                    # 不是專案

    # 圖片：每張變成一個姿勢（含 Unity 套件）
    imgs = []
    for fid in (45, 300, 610):
        p = os.path.join(TMP, f'照片_{fid}.png')                     # 中文檔名
        cv2.imencode('.png', encode(fid))[1].tofile(p)
        imgs.append(p)
    eng.start_image_export(imgs)
    assert wait_idle(eng, 60)
    st = eng.get_state(False)
    print('images', st['error'], st['job'])
    assert st['error'] is None and st['job']['saved'] == 3
    check_package(st['job']['unity'], [f'照片_{f}_pose.anim' for f in (45, 300, 610)])
    assert st['job']['project'] == proj
    cs = os.path.join(proj, 'Assets', 'DesktopFBT', 'Editor', 'DesktopFBTBvhImporter.cs')
    assert os.path.exists(cs) and os.path.exists(cs + '.meta')
    for f in (45, 300, 610):
        assert os.path.exists(os.path.join(proj, 'Assets', 'DesktopFBT', 'Motions', f'照片_{f}_pose.anim'))
    print('Unity 專案：', sorted(os.listdir(os.path.join(proj, 'Assets', 'DesktopFBT', 'Motions'))))
    for fid in (45, 300, 610):
        text = open(os.path.join(os.path.dirname(st['job']['export']), f'照片_{fid}_pose.bvh'),
                    encoding='utf-8').read()
        fk = bvh.forward_kinematics(text)
        J = M.person(fid / FPS)
        # 相對骨盆的姿勢（容許整體繞 y 軸旋轉）
        names = {'LeftLeg': P.L_KNEE, 'RightLeg': P.R_KNEE, 'LeftFoot': P.L_ANK, 'RightFoot': P.R_ANK,
                 'LeftHand': P.L_WRI, 'RightHand': P.R_WRI, 'Head': P.NECK}
        A = np.array([fk[n][0] - fk['Hips'][0] for n in names])
        B = np.array([J[j] - J[P.HIP] for j in names.values()])
        tf = kabsch_y(np.vstack([A, [[0, 0, 0]]]), np.vstack([B, [[0, 0, 0]]]))
        e = np.linalg.norm(tf(A) - tf(np.zeros(3)) + 0 - B + (tf(np.zeros(3)) - 0) * 0, axis=1)
        e = np.linalg.norm((tf(A) - tf(np.zeros((1, 3)))) - B, axis=1)
        foot = min(fk['LeftFoot'][0][1], fk['RightFoot'][0][1])
        print(f'  照片 {fid}: 關節誤差中位數 {np.median(e) * 100:.1f} cm，較低腳踝高度 {foot * 100:.1f} cm')
        assert np.median(e) < 0.08 and 0.0 < foot < 0.2
    eng.shutdown()

    if sys.platform == 'win32':
        test_window_capture()
    print('全部通過')


if __name__ == '__main__':
    main()
