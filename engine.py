"""DesktopFBT 引擎：影片／視窗 → 3D 姿勢 → VRChat Avatar 參數（OSC）；影片、圖片 → 動作檔。"""
import base64
import collections
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback

import cv2
import numpy as np

import models
import pose as P
import safe_write
import unity_pkg
import gemx as GX
import temporal as TM
import anim as ANIM
import clips as CL
import limbs as LB
import phone_server as PS
import video as V
from bvh import fill_and_smooth, lock_feet, to_bvh
from lift import SOLE, Lifter, Pose3DEstimator, lift_image
from paths import data
from skeleton import Skeleton, confidence

VERSION = '1.13.2'

DEFAULT_CONFIG = {
    'language': 'auto',
    'user_height_m': 1.65,
    'pose': {'device': 'auto', 'precision': 'auto', 'det_frequency': 8, 'score_thr': 0.4, 'batch': 1, 'tta': False,
             'head_cm': 10},
    'vrchat': {'host': '127.0.0.1', 'port': 9000},
    'live': {'fps': 30, 'precise': False, 'delay': 0.8, 'smoothness': 0.85, 'grounded': True, 'temporal': True,
             'temporal_model': 'motionbert'},
    'video': {'hfov_deg': 60, 'height_m': 0, 'mirror': False, 'loop': True, 'unity_project': '',
              'method': 'standard'},
    'motion': {'loop': True, 'speed': 1.0},
    'app': {'onboarded': False},
}
COUNTDOWN = 5.0                     # 相機追蹤：按下開始後倒數幾秒（讓人走到定位）

# 程式內部參數：舊版存下的值不蓋掉新版的預設
INTERNAL_KEYS = {'pose': ['det_frequency']}
SECTIONS = set(DEFAULT_CONFIG)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _clean(cfg):
    """只留下這個版本認得的設定（舊版的相機、校正等設定直接丟掉）。"""
    out = {}
    for k, v in (cfg or {}).items():
        if k not in SECTIONS:
            continue
        if isinstance(DEFAULT_CONFIG[k], dict):
            if not isinstance(v, dict):
                continue
            v = {a: b for a, b in v.items() if a in DEFAULT_CONFIG[k] and a not in INTERNAL_KEYS.get(k, ())}
        out[k] = v
    return out


def _diff(cfg, base):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            d = _diff(v, base[k])
            if d:
                out[k] = d
        elif base.get(k, object()) != v:
            out[k] = v
    return out


class Stopped(Exception):
    pass


_CUE_TEXT = {-1: '手在頭後面', 1: '手在臉前面', 0: '判斷不出來'}


MOTION_FADE_S = 0.25            # 動作庫：切換動作時的過渡
LIVE_REC = 2700                 # 即時追蹤結束時存下最近幾幀的模型輸出（30fps 約 90 秒；回報問題用）
GEMX_KEEP_S = 180.0             # 匯出結束後 GEM-X 的模型留多久（接著匯出下一個檔不用重新載入）


def _far(X, to_cam):
    """GEM-X 的結果：相機的位置（遠遠地在「指向相機」的方向上），給頭和手的前後判斷用。"""
    return 0.5 * (X[P.L_EAR] + X[P.R_EAR]) + 50.0 * np.asarray(to_cam, float)


def _cue_record(raw, cues, steady, details):
    """和動作一起存進 _joints.npz 的記錄（回報「手和頭的前後不對」時用來對照）：修正前的關節、每幀左／右手的判斷與依據。"""
    n = len(cues)
    rec = {k: np.full((n, 2), np.nan, np.float32) for k in ('cue_under', 'cue_local', 'cue_ref', 'cue_hand')}
    for i, d in enumerate(details):
        for j, side in enumerate(('Left', 'Right')):
            v = (d or {}).get(side)
            if v:
                for k in ('under', 'local', 'ref', 'hand'):
                    if v.get(k) is not None:
                        rec['cue_' + k][i, j] = v[k]
    pick = lambda cs: np.array([[int((c or {}).get(side, 0)) for side in ('Left', 'Right')] for c in cs], np.int8).reshape(n, 2)   # noqa: E731
    return dict(rec, raw=np.asarray(raw, np.float32), cue=pick(cues), cue_used=pick(steady))


def _r2(v):
    return '—' if v is None else f'{v:.2f}'


def _unique(base, ext):
    out = base + ext
    n = 2
    while os.path.exists(out):
        out = f'{base} ({n}){ext}'
        n += 1
    return out


def check(k, s, thr, size=None):
    """畫面裡的人看得清楚嗎（介面提示用）。"""
    if k is None:
        return dict(person=False, feet=False, joints=0, edge=False)
    edge = False
    if size is not None:
        w, h = size
        for j in (P.HEAD, P.L_ANK, P.R_ANK, P.L_HEEL, P.R_HEEL):
            if s[j] > 0.2:
                u, v = k[j]
                edge |= bool(u < 0.01 * w or u > 0.99 * w or v < 0.025 * h or v > 0.975 * h)
    return dict(person=True, feet=bool(s[P.L_ANK] > thr and s[P.R_ANK] > thr),
                joints=int((s > thr).sum()), edge=edge)


class Engine:
    def __init__(self, preload=True):
        self.config_path = data('config.json')
        self._log_file = data('log.txt')
        self._lock = threading.RLock()
        self._cfg_write_lock = threading.Lock()
        self.cfg = self._load_config()
        self._worker = None
        self._stop = threading.Event()
        self._est = None
        self._est_key = None
        self._tm = None
        self._gx = None                                         # GEM-X：(裝置, 模型)
        self._gx_lock = threading.Lock()
        self._gx_gen = 0
        self._phone = None
        self._auto_dev = None
        self._accel_logged = False
        if preload and sys.platform == 'win32' and self.cfg['pose'].get('device', 'auto') in ('auto', 'migraphx'):
            import accel
            threading.Thread(target=accel.ensure, daemon=True).start()
        self._preview = None
        self._motion_last = None                # 動作庫最後送出的參數：換另一個動作時從這裡接過去
        self.state = dict(version=VERSION, mode='idle', phase='', error=None, progress=0.0,
                          fps=0.0, src_fps=0.0, model_ms=0.0, precision='', device='', delay=0.0,
                          model=dict(status='unknown', progress=0.0), check=None, job=None, osc=None,
                          temporal=None)

    # ------------------------------------------------------------ 設定
    def _load_config(self):
        cfg = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, encoding='utf-8') as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        live = cfg.get('live') if isinstance(cfg, dict) else None
        if isinstance(live, dict) and live.get('output') == 'vr':
            # 舊版的 VR 模式（已移除）切換時會自動關掉時間序列 3D 和精準模式：那不是使用者自己選的，回到預設
            live.pop('temporal', None)
            live.pop('precise', None)
        video = cfg.get('video') if isinstance(cfg, dict) else None
        if isinstance(video, dict) and video.get('method') == 'gvhmr':
            video['method'] = 'gemx'                    # GVHMR 已換成 GEM-X：同樣是「高精度」那個選項
        return _merge(DEFAULT_CONFIG, _clean(cfg))

    def save_config(self, cfg=None):
        out = _clean(_diff(self.cfg if cfg is None else cfg, DEFAULT_CONFIG))
        safe_write.write_json_atomic(self.config_path, {k: v for k, v in out.items() if v != {}},
                                     ensure_ascii=False, indent=2)

    def get_config(self):
        with self._lock:
            return copy.deepcopy(self.cfg)

    def update_config(self, patch):
        with self._cfg_write_lock:
            with self._lock:
                new = _merge(self.cfg, _clean(patch))
            try:
                self.save_config(new)
            except safe_write.SaveError as e:
                self.log(f'設定無法存檔：{e.file}（{type(e.cause).__name__}），這次的變更沒有套用')
                raise RuntimeError(f'ERR_SAVE_CONFIG:{e.file}') from None
            with self._lock:
                self.cfg = new
        return self.get_config()

    def log(self, msg):
        line = time.strftime('%H:%M:%S ') + str(msg)
        try:
            print(line)
        except Exception:
            pass
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass

    # ------------------------------------------------------------ 狀態
    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def get_state(self, with_preview=True):
        with self._lock:
            st = copy.deepcopy(self.state)
            if with_preview:
                st['preview'] = self._preview
        return st

    def _show(self, img, k, s, thr, marks=None):
        h, w = img.shape[:2]
        scale = 640.0 / max(w, h)
        out = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if k is not None:
            out = P.draw(out, k * scale, s, thr, marks)
        ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with self._lock:
            self._preview = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode() if ok else None

    # ------------------------------------------------------------ 工作
    def _stop_event(self):
        """這條執行緒所屬工作的停止旗標。每個工作各有一個：上一個工作停得慢（例如相機還在開、模型還在編譯）時，
        新的工作開始也不會把它的旗標清掉讓它「復活」，它跑到下一個檢查點就會結束。"""
        return getattr(threading.current_thread(), 'stop_ev', self._stop)

    def _check_stop(self):
        if self._stop_event().is_set():
            raise Stopped()

    def _start(self, mode, fn, *args):
        self.stop()
        self._stop = threading.Event()
        self._set(mode=mode, phase='', error=None, progress=0.0, fps=0.0, src_fps=0.0, check=None, countdown=None,
                  job=None, osc=None, delay=0.0, temporal=None)
        with self._lock:
            self._preview = None
        if fn != self._play_motion:
            self._motion_last = None                            # Avatar 接下來由別的東西驅動
        if mode != 'export':
            self._gemx_free()                                   # 即時追蹤不用 GEM-X：顯示記憶體留給 VRChat
        try:
            LB.set_head_size(float(self.cfg['pose'].get('head_cm', 10)) / 100.0)      # 手不穿過頭：頭的大小
        except (TypeError, ValueError):
            LB.set_head_size(LB.HEAD_R)

        def run():
            try:
                fn(*args)
            except Stopped:
                pass
            except Exception as e:
                self.log(traceback.format_exc())
                if isinstance(e, P.GPUError):           # 顯示卡出錯後模型就不能用了：下次開始時重新載入
                    self._est = self._tm = None
                    self._gemx_free()
                msg = str(e)
                self._set(error=msg if msg.startswith('ERR_') else f'ERR_GENERIC:{msg}')
            finally:
                if mode == 'export':
                    self._gemx_free(GEMX_KEEP_S)
                if self._worker is threading.current_thread():      # 已經被新的工作取代時不要動它的狀態
                    self._set(mode='idle', phase='')
        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.stop_ev = self._stop
        self._worker.start()

    def stop(self):
        w = self._worker
        if w and w.is_alive():
            self._stop.set()
            w.join(timeout=5.0)
            if w.is_alive():                                    # 還卡在某個等待裡：旗標留著，它之後自己結束
                self.log('上一個工作還沒完全停下來（之後會自己結束）')
                self._set(mode='idle', phase='')
        self._worker = None

    def shutdown(self):
        self.stop()
        if self._phone is not None:
            self._phone.stop()

    # ------------------------------------------------------------ 模型
    def _accel(self):
        """AMD MIGraphX（Windows ML）：第一次要查詢／下載，在背景做。"""
        if self.cfg['pose'].get('device', 'auto') not in ('auto', 'migraphx'):
            return
        import accel
        if not accel.status()['done']:
            # 第一次要向系統查詢／下載，可能要等很久：顯示「準備中」，而且等的時候按停止要有反應
            self._set(phase='preparing_accel')
            th = threading.Thread(target=accel.ensure, daemon=True)
            th.start()
            while th.is_alive():
                self._check_stop()
                th.join(0.1)
        ok = accel.ensure()
        if not self._accel_logged:
            self._accel_logged = True
            found = accel.gpus()
            if found:
                self.log('顯示卡：' + '、'.join(f'{n}（{v}）' for v, n in found))
            if ok:
                self.log('AMD MIGraphX：可以使用')
            elif accel.wanted(found):
                self.log(f'AMD MIGraphX 無法使用（改用 DirectML）：{accel.status()["error"]}')
            else:
                self.log('沒有 AMD 顯示卡：使用 DirectML（NVIDIA、Intel 都支援）')

    def _auto_gpu(self, src):
        """「自動」時：MIGraphX 和 DirectML 在這台電腦上實測，用比較快的（只測一次，結果存檔）。"""
        import accel
        model = models.fp16_model(src) or models.batch_model(src) or src
        self._set(phase='benchmarking')
        ms = accel.bench(model, data('models', 'gpu_bench.json'), 4, VERSION + str(accel.status().get('ok')))
        ok = {k: v for k, v in ms.items() if v}
        dev = min(ok, key=ok.get) if ok else 'dml'
        self._auto_dev = dev
        self.log('顯示卡實測（每批 4 幀）：' + '，'.join(f'{k} {v:.1f} ms' if v else f'{k} 無法使用' for k, v in ms.items())
                 + f' → 使用 {dev}')
        self._set(phase='loading_model')
        return dev

    def _estimator(self):
        p = self.cfg['pose']
        self._accel()
        device = P.resolve_device(p.get('device', 'auto'))
        if p.get('device', 'auto') == 'auto' and device == 'migraphx' and self._auto_dev:
            device = self._auto_dev
        prec = p.get('precision', 'auto')
        key = (device, prec)
        if self._est is not None and self._est_key == key:
            self._est.tta = self._est.live_tta = False
            return self._est

        def prog(part, frac):
            base, span = (0.0, 0.2) if part == 'det' else (0.2, 0.8)
            self._set(model=dict(status='downloading', progress=base + span * (frac or 0.0)))
            self._check_stop()

        try:
            for attempt in (0, 1):
                if not models.pose3d_ready():
                    self._set(phase='downloading_model', model=dict(status='downloading', progress=0.0))
                    paths = models.ensure_pose3d(prog)
                else:
                    paths = models.ensure_pose3d()
                self._set(phase='loading_model', model=dict(status='loading', progress=1.0))
                if p.get('device', 'auto') == 'auto' and device == 'migraphx':
                    device = self._auto_gpu(paths['pose3d'])
                    key = (device, prec)
                try:
                    chain = P.fallbacks(device)
                    for i, dev in enumerate(chain):
                        if dev == 'migraphx':
                            self._set(phase='compiling_model')       # 第一次要編譯（之後有快取）
                        try:
                            est = Pose3DEstimator(paths, dev, p.get('det_frequency', 8), prec if dev != 'cpu' else 'fp32')
                            device = dev
                            break
                        except Exception as e:
                            if i == len(chain) - 1:
                                raise
                            self.log(f'{dev} 無法使用，改用 {chain[i + 1]}：{e}')
                        finally:
                            self._set(phase='loading_model')
                    break
                except Exception:
                    bad = [paths[k] for k in ('det', 'pose3d') if attempt == 0 and models.damaged(paths[k])]
                    if not bad:
                        raise
                    for f in bad:
                        self.log(f'模型檔損壞，刪除後重新下載：{os.path.basename(f)}')
                        try:
                            os.remove(f)
                        except OSError as e:
                            raise models.ModelDownloadError(os.path.basename(f), [('local', 'delete failed', e)])
        except Stopped:
            raise
        except Exception as e:
            if isinstance(e, models.ModelDownloadError):
                self.log(f'模型下載失敗：{e.file}：{e.detail()}')
            self._set(model=dict(status='error', progress=1.0))
            raise
        finally:
            for w in models.take_warnings():
                self.log(w)
        self._est, self._est_key = est, key
        self._set(device=device, precision=est.precision, model=dict(status='ready', progress=1.0))
        return est

    def _temporal(self):
        """時間序列 3D 模型（打包在程式裡）；沒有或載入失敗時 None。"""
        name = self.cfg['live'].get('temporal_model', 'motionbert')
        if name not in TM.MODELS:
            name = 'motionbert'
        path = TM.model_path(name)
        if path is None:
            return None
        device = self._est.device if self._est is not None else P.resolve_device(self.cfg['pose'].get('device', 'auto'))
        if self._tm is None or self._tm[0] != (device, name):
            try:
                self._tm = ((device, name), TM.Model(path, device, TM.MODELS[name][1]))
                self._tm[1].name = name
                self.log(f'時間序列 3D 模型：{name}（{self._tm[1].device}，{self._tm[1].T} 幀）')
                if self._tm[1].error:
                    self.log(f'時間序列 3D 在顯示卡上無法使用：{self._tm[1].error[:700]}')
            except Exception as e:
                self.log(f'時間序列 3D 模型無法載入：{e}')
                return None
        return self._tm[1]

    def _height(self):
        return float(self.cfg['video'].get('height_m') or self.cfg.get('user_height_m', 1.65))

    # ------------------------------------------------------------ 即時：影片／視窗 → VRChat
    def list_windows(self):
        try:
            return V.list_windows()
        except Exception as e:
            self.log(f'列出視窗失敗：{e}')
            return []

    def _open_source(self, src):
        kind = src.get('kind')
        if kind == 'file':
            path = src.get('path') or ''
            if not os.path.isfile(path):
                raise RuntimeError('ERR_VIDEO_OPEN')
            return V.VideoFile(path, loop=bool(self.cfg['video'].get('loop', True))), os.path.basename(path)
        if kind == 'window':
            fps = int(self.cfg['live'].get('fps', 30))
            return V.WindowSource(int(src['id']), fps=max(15, min(fps, 120))), src.get('title', '')
        if kind == 'camera':
            fps = int(self.cfg['live'].get('fps', 30))
            return V.CameraSource(int(src['id']), fps=fps), src.get('title') or f'Camera {src["id"]}'
        if kind == 'url':
            url = (src.get('url') or '').strip()
            if not url.lower().startswith(('http://', 'https://', 'rtsp://', 'rtmp://', 'udp://')):
                raise RuntimeError('ERR_CAMERA_URL')
            return V.CameraSource(url), url
        if kind == 'phone':
            hub = self._phone_hub()
            self._set(phase='waiting_phone')
            try:
                ps = PS.PhoneSource(hub, should_stop=self._stop_event().is_set)
            except InterruptedError:
                raise Stopped() from None
            return ps, ps.name
        raise RuntimeError('ERR_VIDEO_OPEN')

    # ------------------------------------------------------------ 手機（掃 QR code 直連）
    def _phone_hub(self):
        if self._phone is None:
            self._phone = PS.PhoneHub()
        if not self._phone.running and not self._phone.start():
            if self._phone.error_detail:
                self.log(f'手機伺服器無法啟動：{self._phone.error_detail}')
            raise RuntimeError(self._phone.error or 'ERR_PHONE_SERVER')
        fps = int(self.cfg['live'].get('fps', 30))
        self._phone.set_format(1280, 720, max(15, min(fps, 60)))
        return self._phone

    def phone_info(self, start=True):
        try:
            hub = self._phone_hub() if start else self._phone
        except RuntimeError as e:
            return dict(running=False, error=str(e))
        return hub.info(with_qr=True) if hub is not None else dict(running=False)

    def list_cameras(self):
        try:
            return V.list_cameras()
        except Exception as e:
            self.log(f'列出相機失敗：{e}')
            return []

    def start_live(self, src):
        self._start('live', self._live, src)

    def _output(self, delay):
        """輸出：Avatar 參數（OSC）。回傳 (Playback, 送出端, 轉換函式)。"""
        import desktop as DK
        v, lc = self.cfg['vrchat'], self.cfg['live']
        desk = DK.DesktopPose(grounded=bool(lc.get('grounded', True)))
        sink = DK.AvatarOSC(v.get('host', '127.0.0.1'), int(v.get('port', 9000)))
        return DK.Playback(sink, delay), sink, (lambda t, X, ok, hands: desk(X, ok, hands, t))

    def _live(self, src):
        import desktop as DK
        from track import Tracker
        cfg = self.cfg
        lc, vc = cfg['live'], cfg['video']
        thr = cfg['pose']['score_thr']
        est = self._estimator()
        self._check_stop()                                # 載入模型可能很久：這段時間按了停止就不要再開來源
        source, name = self._open_source(src)
        is_file = isinstance(source, V.VideoFile)
        mirror = bool(vc.get('mirror'))
        B = max(1, int(cfg['pose'].get('batch', 1)))
        est.live_tta = bool(cfg['pose'].get('tta', False))          # 每幀多算一次左右翻轉再平均（模型時間加倍）
        est.pad_batch = B * (2 if est.live_tta else 1) if est.device == 'migraphx' else 0
        frames = collections.deque(maxlen=32)
        cond = threading.Condition()
        stop_ev = threading.Event()
        q = queue.Queue(maxsize=4)
        stats = dict(src_n=0, src_t=time.perf_counter(), dropped=0)

        def feed():                                   # 讀畫面（自己一條執行緒：解碼／擷取不會拖慢推論）
            last = -1
            try:
                while not stop_ev.is_set():
                    r = source.read() if is_file else source.read(last, timeout=0.5)
                    if r is None:
                        if is_file:
                            with cond:
                                frames.append(None)
                                cond.notify()
                            return
                        continue
                    img, t_src, last = r
                    with cond:
                        frames.append((img, time.perf_counter(), float(t_src)))
                        stats['src_n'] += 1
                        cond.notify()
            except BaseException as e:            # noqa: BLE001
                with cond:
                    frames.append(e)
                    cond.notify()

        def infer():                                  # 一次拿最近的幾幀一起推論；跟不上時丟掉最舊的
            t_view = 0.0
            try:
                while not stop_ev.is_set():
                    with cond:
                        while not frames and not stop_ev.is_set():
                            cond.wait(0.2)
                        items = list(frames)
                        frames.clear()
                    if not items:
                        continue
                    end = next((x for x in items if x is None or isinstance(x, BaseException)), 'none')
                    items = [x for x in items if isinstance(x, tuple)]
                    if len(items) > B:
                        stats['dropped'] += len(items) - B
                        items = items[-B:]
                    if items:
                        imgs = [cv2.flip(x[0], 1) if mirror else x[0] for x in items]
                        view = None
                        now = time.perf_counter()
                        if now - t_view > 2.0:
                            t_view = now
                            crop, (x0, y0) = V.crop_black(imgs[-1])     # 去掉播放器的黑邊，視角才算得準
                            view = (x0, y0, crop.shape[1], crop.shape[0])
                        res = est.batch(imgs)
                        q.put((imgs[-1], [(x[1], r) for x, r in zip(items, res)], view))
                    if end != 'none':
                        q.put(end)
                        return
            except BaseException as e:            # noqa: BLE001
                q.put(e)

        th_feed = threading.Thread(target=feed, daemon=True)
        th_inf = threading.Thread(target=infer, daemon=True)
        osc = out = tl = None
        try:
            self._check_stop()
            height = self._height()
            trk = Tracker(source.size, getattr(source, 'hfov', None) or vc.get('hfov_deg', 60), height, thr,
                          precise=bool(lc.get('precise', False)), decide_s=float(lc.get('delay', 0.8)),
                          smoothness=float(lc.get('smoothness', 0.85)))
            lifter = trk.lifter
            prec_delay = trk.delay_s
            v = cfg['vrchat']
            tm = self._temporal() if lc.get('temporal', True) else None
            if tm is not None:
                tm.size = source.size
            if tm is not None and tm.device == 'cpu':
                self.log('時間序列 3D 只能用 CPU 跑（太慢），即時模式不使用')
                tm = None
            tl = TM.Live(tm, height) if tm is not None else None
            delay = prec_delay + (tl.lag + 0.35 if tl is not None else 0.0)
            out, osc, convert = self._output(delay)
            self.log(f'即時開始：{name}，{est.device}/{est.precision}，批次 {B}{"（翻轉平均）" if est.live_tta else ""}，延遲 {delay:.2f}s')
            old_ctrl = DK.stale_avatar()
            if old_ctrl:
                self.log(f'Avatar「{old_ctrl}」的動畫控制器是舊版（改名前的 Phone FBT 或沒有 TiltX／TiltZ）。'
                         '請在 Unity 重新產生控制器並重新上傳 Avatar。')
            self._warn_old_sync()
            info = dict(kind=src.get('kind'), name=name, duration=source.duration if is_file else 0.0, position=0.0)
            self._set(phase='running', job=info, delay=delay,
                      osc=dict(host=v.get('host', '127.0.0.1'), port=int(v.get('port', 9000)), sent=0))
            th_feed.start()
            th_inf.start()
            n, t_fps, fps, t_ui = 0, time.perf_counter(), 0.0, 0.0
            lags = collections.deque(maxlen=150)          # 最近每一幀從擷取到算好姿勢花了多久
            rec = dict(inp=collections.deque(maxlen=LIVE_REC), out=collections.deque(maxlen=LIVE_REC), dropped=0,
                       meta=dict(version=VERSION, source=str(src.get('kind')), size=list(source.size), height=height,
                                 hfov=float(getattr(source, 'hfov', None) or vc.get('hfov_deg', 60)), thr=thr,
                                 precise=bool(lc.get('precise', False)), decide=float(lc.get('delay', 0.8)),
                                 smoothness=float(lc.get('smoothness', 0.85)), grounded=bool(lc.get('grounded', True)),
                                 temporal=None if tm is None else getattr(tm, 'name', 'on'), batch=B, device=est.device))
            logged = getattr(est, 'trial_result', None) is not None
            t_go = self._countdown_start(src)
            while True:
                self._check_stop()
                try:
                    item = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                img, results, view = item
                if t_go:                                     # 倒數中：只顯示畫面，還不送動作
                    left = t_go - time.perf_counter()
                    if left < 1.5:                           # 最後一秒半：人站直，取樣定出「上」
                        for _, (k_, z_, sc_, _) in results:
                            if k_ is not None:
                                lifter.stand_sample(k_, z_, sc_)
                    t_go = self._countdown_tick(t_go)
                    if t_go:
                        k, _, sc, _ = results[-1][1]
                        self._show(img, k, sc, thr)
                        continue
                    pitch = lifter.stand_lock()
                    rec['meta']['pitch_lock'] = pitch
                    self.log('站直校正：' + (f'相機俯仰 {pitch:.1f}°，之後固定' if pitch is not None
                                            else '倒數結束時沒有拍到站直的全身，改用自動估計'))
                if view is not None:
                    lifter.set_view(view)
                ready = []
                for t_cap, item in results:
                    if tl is None:
                        ready.append((t_cap, item, None))
                    elif item[0] is None:
                        tl.skip(t_cap, item)
                    else:
                        tl.push(t_cap, item[0], item[2], item)
                if tl is not None and tl.dead:                # 顯示卡上跑不起來：不用了（不改用 CPU）
                    self.log(f'時間序列 3D 在 {tl.model.device} 上執行失敗，即時模式停用：{(tl.last_error or "")[:700]}')
                    ready += tl.pull(time.perf_counter())
                    tl.stop()
                    tl = None
                    out.delay = prec_delay
                if tl is not None:
                    ready = tl.pull(time.perf_counter())
                    want = prec_delay + tl.delay_s + 0.15
                    if want > out.delay + 0.05:              # 時間序列模型比預期慢：延遲跟著拉長，不要卡頓
                        out.delay = min(want, prec_delay + tl.give_up + 0.2)
                for t_cap, (k, z, sc, hr), zt in ready:
                    n += 1
                    if k is None:
                        continue
                    rec['inp'].append((t_cap, k, z, sc, zt, hr))
                    r = trk(t_cap, k, z, sc, hr, zt)
                    prm = convert(r[0], r[1], r[2], r[3]) if r is not None else None
                    if prm is not None:
                        out.add(r[0], prm)
                        lags.append(time.perf_counter() - r[0])
                        rec['out'].append((r[0], r[1], prm.get('Y', 0.0)))
                now = time.perf_counter()
                if now - t_fps > 0.5:
                    # 電腦比較慢、算好的姿勢經常來不及在設定的延遲內送到：延遲跟著拉長（只加不減），不然動作會一格一格的。
                    # 看最近的第 90 百分位：偶爾卡一下（剛開始、切換視窗）不算
                    lag = float(np.percentile(lags, 90)) if len(lags) >= 60 else 0.0
                    if lag + 0.02 > out.delay and out.delay < delay + 1.0:
                        out.delay = min(lag + 0.06, delay + 1.0)
                        self.log(f'處理比預期慢（{lag:.2f}s），輸出延遲拉長到 {out.delay:.2f}s')
                        lags.clear()
                    fps = n / (now - t_fps)
                    sfps = stats['src_n'] / (now - stats['src_t'])
                    n, t_fps = 0, now
                    stats['src_n'], stats['src_t'] = 0, now
                    self._set(models=dict(pose='RTMW3D-x', device=est.device,
                                          temporal=None if tl is None else getattr(tl.model, 'name', None)),
                              fps=round(fps, 1), src_fps=round(sfps, 1), model_ms=round(est.ms, 1),
                              precision=est.precision, delay=round(out.delay, 2),
                              temporal=None if tl is None else dict(
                                  ms=round(tl.model.ms, 1), used=tl.used, passed=tl.passed, device=tl.model.device))
                    tr = getattr(est, 'trial_result', None)
                    if tr and not logged:
                        logged = True
                        self.log(f'半精度比對：誤差 {tr["err"]:.4f}，全精度 {tr["ms32"]:.1f} ms，半精度 {tr["ms16"]:.1f} ms → 使用 {tr["used"]}')
                    if stats['dropped'] >= 30:
                        rec['dropped'] += stats['dropped']
                        if now - rec.get('warned', -1e9) > 20.0:      # 一直跟不上時每 20 秒提醒一次就好（以前每 2 秒一行，log 都是它）
                            rec['warned'] = now
                            self.log(f'推論跟不上，丟掉 {stats["dropped"]} 幀（可以在設定把「一次推論幾幀」調高）')
                        stats['dropped'] = 0
                    rec['meta'].update(fps=round(fps, 1), src_fps=round(sfps, 1), model_ms=round(est.ms, 1), delay=round(out.delay, 2),
                                       temporal_ms=None if tl is None else round(tl.model.ms, 1),
                                       temporal_used=None if tl is None else tl.used, temporal_passed=None if tl is None else tl.passed)
                if now - t_ui > 0.066:
                    t_ui = now
                    k, _, sc, _ = results[-1][1]
                    self._show(img, k, sc, thr, dict(trk.sk.fs_now))      # 正對鏡頭的手腳：標出朝前還是朝後
                    pos = float(source.idx / source.fps) if is_file else 0.0
                    self._set(check=check(k, sc, thr, (img.shape[1], img.shape[0])),
                              job=dict(info, position=pos), osc=dict(self.state['osc'] or {}, sent=osc.n))
        finally:
            stop_ev.set()
            with cond:
                cond.notify_all()
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            for th in (th_inf, th_feed):
                if th.is_alive():
                    th.join(timeout=3.0)
            if out is not None:
                out.stop()
            if osc is not None and hasattr(osc, 'close'):
                osc.close()
            if tl is not None:
                tl.stop()
            source.stop()
            for w in V.take_warnings():
                self.log(w)
            self._save_live_record(locals().get('rec'))

    def _save_live_record(self, rec):
        """即時追蹤結束時：寫一行統計，並把最近一段的模型輸出存成 live_debug.npz（只有關節座標，沒有影像）。
        回報「動作不對」時可以用同一份資料在開發環境重跑整個流程，找出是哪一步的問題。"""
        if not rec or not rec['inp']:
            return
        try:
            m = dict(rec['meta'], dropped=rec['dropped'] + 0)
            tu, tp = m.get('temporal_used'), m.get('temporal_passed')
            self.log(f'即時結束：來源 {m.get("src_fps")} fps、處理 {m.get("fps")} fps、模型 {m.get("model_ms")} ms'
                     + (f'、時間序列 {m.get("temporal_ms")} ms（用上 {tu} 幀、沒趕上 {tp} 幀）' if tu is not None else '')
                     + f'、丟掉 {m["dropped"]} 幀、延遲 {m.get("delay")}s')
            inp, outp = list(rec['inp']), list(rec['out'])
            nan17, z42 = np.full(17, np.nan, np.float32), np.zeros(42, np.float32)
            hands = [h for *_, h in inp]
            np.savez_compressed(
                data('live_debug.npz'), meta=json.dumps(m, ensure_ascii=False),
                t=np.array([f[0] for f in inp]), k=np.array([f[1] for f in inp], np.float32),
                z=np.array([f[2] for f in inp], np.float32), s=np.array([f[3] for f in inp], np.float32),
                zt=np.array([nan17 if f[4] is None else f[4] for f in inp], np.float32),
                hk=np.array([np.zeros((42, 2), np.float32) if h is None else h[0] for h in hands], np.float32),
                hz=np.array([z42 if h is None else h[1] for h in hands], np.float32),
                hs=np.array([z42 if h is None else h[2] for h in hands], np.float32),
                cue=np.array([[int((getattr(h, 'cue', None) or {}).get(sd, 0)) for sd in ('Left', 'Right')] for h in hands], np.int8),
                out_t=np.array([o[0] for o in outp]), out_x=np.array([o[1] for o in outp], np.float32).reshape(len(outp), -1, 3),
                out_y=np.array([o[2] for o in outp], np.float32))
        except Exception as e:                      # 記錄失敗不能影響正常使用
            self.log(f'即時記錄沒有存成：{type(e).__name__}: {e}')

    # ------------------------------------------------------------ 倒數
    def _countdown_start(self, src):
        """相機（含手機）：按下開始後先倒數，讓人走到定位。回傳結束時間（不倒數時 0）。"""
        if src.get('kind') not in ('camera', 'url', 'phone'):
            return 0.0
        self._set(phase='countdown', countdown=int(COUNTDOWN))
        return time.perf_counter() + COUNTDOWN

    def _countdown_tick(self, t_go):
        left = t_go - time.perf_counter()
        if left <= 0:
            self._set(phase='running', countdown=None)
            return 0.0
        n = int(np.ceil(left))
        if self.state.get('countdown') != n:
            self._set(countdown=n)
        return t_go

    # ------------------------------------------------------------ 匯出：影片 → 動作檔
    def _warn_old_sync(self):
        import desktop as DK
        try:
            name = DK.old_sync_avatar()
        except Exception:
            name = None
        if name:
            self.log(f'Avatar「{name}」用的是舊的同步方式：別人看到的身體會歪、細節少。'
                     '用新版的 Unity 工具重新「追加」並上傳後會改善（VRChat 的 OSC 設定沒更新時這個提醒可能是舊的）。')

    # ------------------------------------------------------------ 動作庫：.anim 直接播成 OSC
    def list_motions(self):
        return dict(items=CL.list_all(), hidden=CL.hidden_count())

    def restore_motions(self):
        return CL.restore_hidden()

    def add_motions(self, paths):
        r = CL.add(paths)
        for e in r['errors']:
            self.log(f'加入動作失敗：{e["name"]}（{e["error"]}）')
        if r['added']:
            self.log('已加入動作：' + ', '.join(os.path.basename(p) for p in r['added']))
        return r

    def remove_motion(self, path):
        return CL.remove(path)

    def start_motion(self, path):
        self._start('live', self._play_motion, path)

    def _play_motion(self, path):
        """把一個 .anim 照時間送到 Avatar（不用模型、不用顯示卡）。循環、速度隨時可以在介面上改。"""
        import desktop as DK
        if not path or not os.path.isfile(path):
            raise RuntimeError('ERR_MOTION_MISSING')
        self._set(phase='loading_motion', job=dict(kind='motion', path=path, name=os.path.splitext(os.path.basename(path))[0]))
        try:
            clip = CL.Clip(path)
        except ValueError as e:
            raise RuntimeError(str(e)) from None
        v = self.cfg['vrchat']
        osc = DK.AvatarOSC(v.get('host', '127.0.0.1'), int(v.get('port', 9000)))
        info = dict(kind='motion', path=path, name=clip.name, duration=clip.seconds, position=0.0, pose=clip.pose)
        self.log(f'播放動作：{clip.name}（{"姿勢" if clip.pose else f"{clip.seconds:.1f} 秒"}）')
        if clip.over > 0:
            self.log(f'這個動畫有 {100 * clip.over:.0f}% 的幀超過 Avatar 參數的範圍（肌肉值 > 1）：被夾住的部分由相鄰的骨頭補回來，姿勢可能和原本略有不同')
        if clip.transforms:
            self.log('這個動畫另外有直接移動骨頭的曲線（旋轉／位置）：那部分不會送出，播出來可能和原本不同')
        old_ctrl = DK.stale_avatar()
        if old_ctrl:
            self.log(f'Avatar「{old_ctrl}」的動畫控制器是舊版。請在 Unity 重新產生控制器並重新上傳 Avatar。')
        self._warn_old_sync()
        self._set(phase='running', job=info, osc=dict(host=v.get('host', '127.0.0.1'), port=int(v.get('port', 9000)), sent=0))
        dt = 1.0 / CL.RATE
        pos, last, nxt, t_ui, n, t_fps = 0.0, time.perf_counter(), time.perf_counter(), 0.0, 0, time.perf_counter()
        t0, prev = last, self._motion_last          # 上一個動作留下的姿勢（中間沒有做過即時追蹤）：用一小段時間接過來，不要瞬間跳過去
        prm = None
        try:
            while not self._stop_event().is_set():
                now = time.perf_counter()
                mc = self.cfg.get('motion') or {}
                loop = bool(mc.get('loop', True))
                try:
                    speed = float(np.clip(float(mc.get('speed', 1.0)), 0.1, 4.0))
                except (TypeError, ValueError):
                    speed = 1.0
                pos += (now - last) * speed
                last = now
                if not clip.pose and not loop and pos >= clip.seconds:      # 不循環：播到最後就停（Avatar 留在最後的姿勢）
                    prm = clip.sample(clip.seconds, False)
                    try:
                        osc.send(prm)
                    except OSError:
                        pass
                    break
                if loop and clip.seconds > 0:
                    pos %= clip.seconds                                     # 播到第幾圈不重要：中途關掉循環時，是把這一圈播完，不是馬上跳到結尾
                prm = clip.sample(pos, loop)
                if prev is not None:
                    if now - t0 < MOTION_FADE_S:
                        prm = CL.mix(prev, prm, (now - t0) / MOTION_FADE_S)
                    else:
                        prev = None
                try:
                    osc.send(prm)
                except (OSError, ValueError, OverflowError, TypeError):
                    pass
                n += 1
                if now - t_ui > 0.08:
                    t_ui = now
                    ok, buf = cv2.imencode('.jpg', CL.draw(prm), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    with self._lock:
                        self._preview = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode() if ok else None
                    el = max(now - t_fps, 1e-6)
                    self._set(fps=n / el if el > 0.5 else CL.RATE, osc=dict(self.state['osc'] or {}, sent=osc.n),
                              job=dict(info, position=0.0 if clip.pose else (pos % clip.seconds if loop else min(pos, clip.seconds))))
                    if el > 2.0:
                        n, t_fps = 0, now
                nxt += dt
                wait = nxt - time.perf_counter()
                if wait > 0:
                    self._stop_event().wait(wait)
                elif wait < -0.5:                                           # 電腦卡了一下：不要補送
                    nxt = time.perf_counter()
        finally:
            if prm is not None:
                self._motion_last = prm
            osc.close()

    def start_export(self, path):
        self._start('export', self._video_export, path)

    def _video_export(self, path):
        cfg = self.cfg
        vc = cfg['video']
        thr = cfg['pose']['score_thr']
        est = self._estimator()
        if not os.path.isfile(path or ''):
            raise RuntimeError('ERR_VIDEO_OPEN')
        if vc.get('method') == 'gemx':
            return self._gemx_export(path, est)
        freq0 = est.freq
        src = V.VideoFile(path, loop=False)
        frames = []
        cues, details = [], []                         # 每幀：手和頭重疊時在前面還是後面（limbs.head_cue）與判斷用的數字
        try:
            name = os.path.splitext(os.path.basename(path))[0]
            info = dict(kind='file', name=os.path.basename(path), duration=src.duration, position=0.0)
            self._set(phase='analyzing', job=info)
            t_ui = 0.0
            est.bbox = None
            est.tta, est.freq = True, 1                # 匯出不用即時：每幀找人、左右翻轉平均
            t0 = time.perf_counter()
            while True:
                self._check_stop()
                r = src.read_next()
                if r is None:
                    break
                img, t_src, i = r
                if vc.get('mirror'):
                    img = cv2.flip(img, 1)
                k, z, sc = est(img)
                frames.append(None if k is None else (k, z, sc))
                cues.append(None if k is None else getattr(est, 'cue', None))
                details.append(None if k is None else getattr(est, 'cue_detail', None))
                now = time.perf_counter()
                if now - t_ui > 0.15:
                    t_ui = now
                    self._show(img, k, sc, thr)
                    prog = (i + 1) / src.frames if src.frames else 0.0
                    self._set(job=dict(info, position=float(t_src)), progress=min(prog, 0.99),
                              fps=round(len(frames) / max(now - t0, 1e-3), 1), model_ms=round(est.ms, 1),
                              check=check(k, sc, thr, (img.shape[1], img.shape[0])))
        finally:
            src.stop()
            for w in V.take_warnings():
                self.log(w)
            est.tta, est.freq = False, freq0
        found = [f for f in frames if f is not None]
        if len(found) < max(10, 0.2 * len(frames)):
            raise RuntimeError('ERR_NO_PERSON_VIDEO')
        self._set(phase='writing')
        height = self._height()
        tm = self._temporal() if cfg['live'].get('temporal', True) else None
        if tm is not None:                              # 整段影片用時間序列模型重新推算深度
            tm.size = src.size
            zts = TM.sequence(tm, [i / src.fps for i in range(len(frames))],
                              [None if f is None else (f[0], f[2]) for f in frames], height, tta=True)
            frames = [f if f is None or zt is None else (f[0], TM.apply_depth(f[1], zt), f[2])
                      for f, zt in zip(frames, zts)]
            self.log(f'時間序列 3D：{sum(z is not None for z in zts)} / {len(found)} 幀')
        lifter = Lifter(src.size, vc.get('hfov_deg', 60), height, thr, smooth=False)
        first = []
        for i, f in enumerate(frames):                 # 第一輪：估計地板與相機俯仰
            if f is not None:
                x, ok = lifter(*f, i / src.fps)
                if x is not None:
                    first.append((x, ok, (0.0, lifter.last_h, 0.0)))
        lifter.freeze()
        sk = Skeleton(height, learn=False)
        sk.set_lengths_from(first)                     # 整段影片學出這個人的骨長
        X = np.zeros((len(frames), P.NUM_JOINTS, 3))
        OK = np.zeros((len(frames), P.NUM_JOINTS), dtype=bool)
        for i, f in enumerate(frames):                 # 第二輪：逐幀 3D
            if f is None:
                continue
            x, ok = lifter(*f, i / src.fps)
            if x is not None:
                X[i], OK[i] = sk.apply(x, ok, confidence(f[2], thr), cam=(0.0, lifter.last_h, 0.0)), ok
        if OK[:, P.HIP].sum() < 10:
            raise RuntimeError('ERR_NO_PERSON_VIDEO')
        X = self._fix_sequence(X, OK, frames, src.fps, lifter.camera_height() or 1.2)
        cam = np.array([0.0, lifter.camera_height() or 1.2, 0.0])
        limbs = LB.Limbs()
        steady = LB.steady_cues(cues, src.fps)         # 前後的判斷：整段影片前後幾幀一起看，不會零星跳動
        raw = X.copy()
        for i, f in enumerate(frames):                 # 手在頭後／前、手不穿過頭和軀幹、膝蓋不往後折
            if f is not None and OK[i, P.HIP]:
                X[i] = limbs(X[i], OK[i], f[2], None, cam, steady[i])[0]
        self._log_cues(cues, limbs)
        self._write_motion(X, OK, src.fps, name, info, len(frames), len(found), extra=_cue_record(raw, cues, steady, details))

    def _log_cues(self, cues, limbs=None):
        """整段影片裡「手和頭重疊」被判斷成前面／後面的幀數、實際調整了幾次（方便回報問題時對照）。"""
        back = sum(1 for c in cues if c and -1 in c.values())
        front = sum(1 for c in cues if c and 1 in c.values())
        moved = limbs.head_side.get('moved', 0) if limbs is not None else 0
        if back or front or moved:
            self.log(f'頭與手：判斷成手在頭後面 {back} 幀、在臉前面 {front} 幀；為了不穿過頭調整了 {moved} 次')

    def _write_motion(self, X, OK, fps, name, info, n_frames, n_found, sigma_root=0.12, sigma_rel=0.04, extra=None):
        root = fill_and_smooth(X[:, [P.HIP]], OK[:, [P.HIP]], fps, sigma_root)
        rel = fill_and_smooth(X - X[:, [P.HIP]], OK & OK[:, [P.HIP]], fps, sigma_rel)
        Xs = rel + root
        low = np.min(Xs[:, :, 1], axis=1)              # 每幀最低的點（倒立時是手和頭）
        floor = float(np.percentile(np.minimum(low, Xs[:, list(SOLE), 1].min(axis=1)), 5))
        Xs[:, :, 1] -= floor
        under = np.minimum(Xs[:, :, 1].min(axis=1), 0.0)
        Xs[:, :, 1] -= under[:, None]                  # 任何一點都不會穿到地板下
        Xs = lock_feet(Xs, fps)
        out = _unique(os.path.join(V.export_dir(), name), '.bvh')
        text = to_bvh(Xs, fps)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text)
        np.savez_compressed(out[:-4] + '_joints.npz', joints=Xs.astype(np.float32), fps=fps, valid=OK, **(extra or {}))
        # Unity 用的 Humanoid 動畫檔（直接就是 .anim，不需要 Unity 先轉換）
        anim_path = ANIM.write(out[:-4] + '.anim', Xs, fps, os.path.splitext(os.path.basename(out))[0])
        with open(anim_path, 'rb') as fh:
            anim_data = fh.read()
        motions = [(os.path.basename(anim_path), anim_data)]
        pkg = unity_pkg.build(out[:-4] + '.unitypackage', motions)
        self.log(f'匯出完成：{out}（{n_frames} 幀，偵測到人 {n_found / n_frames:.0%}）')
        self._set(progress=1.0, job=dict(info, export=out, anim=anim_path, unity=pkg, frames=n_frames,
                                         **self._to_unity_project(motions)))

    def _gemx(self, repair=True):
        """GEM-X 的檔案（三個模型＋骨架）：缺的自動下載；載入失敗時核對檔案，壞掉的重新下載一次。"""
        if GX.missing():
            self._gemx_download()
        p = GX.paths()
        dev = self._est.device if self._est is not None else 'cpu'
        key = ('gemx', dev)
        with self._gx_lock:
            self._gx_gen += 1                                   # 取消排定的釋放
        if self._gx is None or self._gx[0] != key:
            self._gx = None                                     # 換裝置：先放掉舊的再載入
            self._set(phase='loading_model')
            try:
                body = GX.Body(p['gemx_skeleton.npz'])
                feat = GX.Features(p['gemx_body_fp16.onnx'], dev)
                kp2d = GX.Keypoints(p['gemx_kp2d_fp16.onnx'], dev)
                net = GX.Net(p['gemx_motion.onnx'], dev)
            except Exception as e:
                self.log(f'GEM-X 載入失敗：{e}')
                bad = GX.broken() if repair else []
                if bad:                                         # 檔案內容不對（複製到一半、磁碟錯誤）：重新下載再試一次
                    self.log('GEM-X：檔案損壞，重新下載 ' + ', '.join(bad))
                    return self._gemx(False)
                raise RuntimeError('ERR_GEMX_MODEL') from None
            self._gx = (key, (feat, kp2d, net, body))
            self.log(f'GEM-X：影像特徵 {feat.device}、2D 關鍵點 {kp2d.device}、時序模型 {net.device}')
        return self._gx[1]

    def _gemx_free(self, delay=0.0):
        """放掉 GEM-X 的模型（約 3.4 GB 顯示記憶體）。delay：過這麼久還沒有下一個 GEM-X 工作才放（連續匯出不用重新載入）。"""
        with self._gx_lock:
            self._gx_gen += 1
            gen = self._gx_gen
            if delay <= 0:
                self._gx = None
                return
            if self._gx is None:
                return

        def free():
            with self._gx_lock:
                if self._gx_gen != gen or self._gx is None:
                    return
                self._gx = None
            self.log('GEM-X：一陣子沒用，已釋放模型（下次使用時重新載入）')
        t = threading.Timer(delay, free)
        t.daemon = True
        t.start()

    def _gemx_download(self):
        """缺的模型自動下載（約 3.5 GB，可以中斷後接著下載）。來源還不能公開下載時，請使用者自己放。"""
        self.log('GEM-X：下載模型 ' + ', '.join(GX.missing()))
        self._set(phase='downloading_gemx', model=dict(status='downloading', progress=0.0))
        t_ui = [0.0]

        def prog(done, total):
            self._check_stop()
            now = time.perf_counter()
            if now - t_ui[0] > 0.3:
                t_ui[0] = now
                self._set(model=dict(status='downloading', progress=done / total))
        try:
            GX.download(prog)
        except GX.Unavailable:
            raise RuntimeError('ERR_GEMX_MISSING:' + ', '.join(GX.missing())) from None
        except GX.DownloadError as e:
            self.log(f'GEM-X 模型下載失敗：{e.file}：{e.why}：{e.detail}')
            raise
        finally:
            self._set(model=dict(status='ready', progress=1.0))
        self.log('GEM-X：模型下載完成')

    @staticmethod
    def _gemx_image(gx, img, k, sc, cue=None):
        """一張圖用 GEM-X：人物框 → 影像特徵與 2D 關鍵點 → 時序模型（一幀）→ 26 點。站在地板上、骨盆在原點。看不到人時回傳 None。"""
        feat, kp2d, net, body = gx
        bb = GX.person_box(k, sc)
        if bb is None:
            return None
        size = (img.shape[1], img.shape[0])
        boxes = GX.square_boxes(bb[None])
        rgb = np.ascontiguousarray(img[..., ::-1])
        f = feat([GX.crop_body(rgb, boxes[0])], GX.cliff(boxes, GX.intrinsics(*size)))
        kp = kp2d([GX.crop_kp(rgb, boxes[0])], boxes)
        X, _, to_cam = GX.solve(net, body, kp, boxes, f, size, 30.0, view=True)
        X = X[0]
        X = LB.Limbs().head(X, np.ones(P.NUM_JOINTS, bool), np.ones(P.NUM_JOINTS), None, _far(X, to_cam[0]), cue)[0]
        X[:, 1] -= X[:, 1].min()
        X[:, [0, 2]] -= X[P.HIP, [0, 2]]
        return X

    def _gemx_export(self, path, est):
        """GEM-X：第一輪找人（每幀的人物框），框平滑後第二輪裁切、算影像特徵與 2D 關鍵點，最後整段一起求動作。"""
        vc = self.cfg['video']
        thr = self.cfg['pose']['score_thr']
        feat, kp2d, net, body = self._gemx()
        name = os.path.splitext(os.path.basename(path))[0]
        src = V.VideoFile(path, loop=False)
        stride = max(1, int(round(src.fps / 30.0)))           # 模型以 30fps 訓練：60fps 的影片每兩幀取一幀
        fps = src.fps / stride
        info = dict(kind='file', name=os.path.basename(path), duration=src.duration, position=0.0)
        self._set(phase='analyzing', job=info)
        freq0 = est.freq
        seen = []                                             # 每幀：(骨架, 分數) 或 None
        try:
            est.bbox = None
            est.tta, est.freq = False, 1
            t_ui, t0 = 0.0, time.perf_counter()
            while True:
                self._check_stop()
                r = src.read_next()
                if r is None:
                    break
                img, t_src, i = r
                if i % stride:
                    continue
                if vc.get('mirror'):
                    img = cv2.flip(img, 1)
                k, _, sc = est(img)
                seen.append(None if k is None else (k, sc, getattr(est, 'cue', None), getattr(est, 'cue_detail', None)))
                now = time.perf_counter()
                if now - t_ui > 0.15:
                    t_ui = now
                    self._show(img, k, sc, thr)
                    prog = (i + 1) / src.frames if src.frames else 0.0
                    self._set(job=dict(info, position=float(t_src)), progress=min(0.1 * prog, 0.1),
                              fps=round(len(seen) / max(now - t0, 1e-3), 1), model_ms=round(est.ms, 1),
                              check=check(k, sc, thr, (img.shape[1], img.shape[0])))
        finally:
            src.stop()
            est.tta, est.freq = False, freq0
        L = len(seen)
        raw = [None if f is None else GX.person_box(f[0], f[1]) for f in seen]
        have = np.array([i for i in range(L) if raw[i] is not None], dtype=int)
        if len(have) < max(10, 0.2 * L):
            raise RuntimeError('ERR_NO_PERSON_VIDEO')
        xyxy = np.zeros((L, 4))
        for c in range(4):                                    # 沒有人的幀：框用前後內插
            xyxy[:, c] = np.interp(np.arange(L), have, [raw[i][c] for i in have])
        boxes = GX.square_boxes(xyxy)
        K = GX.intrinsics(*src.size)
        cliffs = GX.cliff(boxes, K)
        kp = np.zeros((L, GX.NJ, 3), np.float32)
        feats = np.zeros((L, 1024), np.float32)
        ok = np.zeros(L, bool)
        ok[have] = True
        got = np.zeros(L, bool)                               # 第二輪真的算過的幀
        src = V.VideoFile(path, loop=False)
        batch = []                                            # (幀編號, RGB 影像)

        def flush():
            idx = [n for n, _ in batch]
            feats[idx] = feat([GX.crop_body(im, boxes[n]) for n, im in batch], cliffs[idx])
            kp[idx] = kp2d([GX.crop_kp(im, boxes[n]) for n, im in batch], boxes[idx])
            got[idx] = True
            batch.clear()
        try:
            t_ui, t0, n, done = 0.0, time.perf_counter(), -1, 0
            while True:
                self._check_stop()
                r = src.read_next()
                if r is None:
                    break
                img, t_src, i = r
                if i % stride:
                    continue
                n += 1
                if n >= L:
                    break
                if vc.get('mirror'):
                    img = cv2.flip(img, 1)
                if ok[n]:
                    batch.append((n, np.ascontiguousarray(img[..., ::-1])))
                    if len(batch) >= 4:
                        flush()
                        done += 4
                now = time.perf_counter()
                if now - t_ui > 0.3:
                    t_ui = now
                    f = seen[n]
                    self._show(img, None if f is None else f[0], None if f is None else f[1], thr)
                    self._set(job=dict(info, position=float(t_src)), progress=min(0.1 + 0.87 * (n + 1) / L, 0.97),
                              fps=round(done / max(now - t0, 1e-3), 1))
            if batch:
                flush()
        finally:
            src.stop()
            for w in V.take_warnings():
                self.log(w)
        if got.sum() < ok.sum():                              # 第二次讀影片時少了幾幀（損壞的檔案）：當成沒有人
            self.log(f'GEM-X：第二次讀取影片時少了 {int(ok.sum() - got.sum())} 幀')
            ok &= got
            have = np.flatnonzero(ok)
            if len(have) < max(10, 0.2 * L):
                raise RuntimeError('ERR_NO_PERSON_VIDEO')
        idx = np.searchsorted(have, np.arange(L))             # 沒有人的幀：特徵用最近的幀，關鍵點留空（信心 0）
        lo, hi = have[np.clip(idx - 1, 0, len(have) - 1)], have[np.clip(idx, 0, len(have) - 1)]
        near = np.where(np.abs(np.arange(L) - lo) <= np.abs(hi - np.arange(L)), lo, hi)
        feats[~ok] = feats[near[~ok]]
        self._set(phase='writing')
        X, scale, to_cam = GX.solve(net, body, kp, boxes, feats, src.size, fps, view=True)
        self.log(f'GEM-X：{L} 幀（{fps:.1f} fps），體型比例 {scale:.2f}，每幀 {(time.perf_counter() - t0) / max(len(have), 1) * 1000:.0f} ms')
        OK = np.zeros((L, P.NUM_JOINTS), bool)
        OK[ok] = True
        # 手和頭重疊時的前後：只在臉部的點判斷得出來時才改 GEM-X 的結果；另外手腕不會在頭裡面
        head = LB.Limbs()
        every, same = np.ones(P.NUM_JOINTS, bool), np.ones(P.NUM_JOINTS)
        cues = [None if f is None else f[2] for f in seen]
        steady = LB.steady_cues(cues, fps)
        raw = X.copy()
        for i in range(L):
            if ok[i]:
                X[i] = head.head(X[i], every, same, None, _far(X[i], to_cam[i]), steady[i])[0]
        self._log_cues(cues, head)
        self._write_motion(X, OK, fps, name, info, L, len(have),
                           extra=dict(_cue_record(raw, cues, steady, [None if f is None else f[3] for f in seen]), to_cam=to_cam))

    @staticmethod
    def _fix_sequence(X, OK, frames, fps, cam_h):
        """整段影片一起決定轉身、前後與左右（和即時一樣的方法）。"""
        from precise import Precise
        prec = Precise(fps, decide_s=1.2, smooth_s=1e-3)
        cam = np.array([0.0, cam_h, 0.0])
        out = X.copy()
        idx = [i for i in range(len(X)) if OK[i, P.HIP]]
        res = [prec.push(i / fps, X[i], frames[i][2], OK[i], None, cam) for i in idx]
        last, dt = idx[-1], 1.0 / fps
        for k in range(1, int((prec.delay_s + 0.5) / dt) + 2):   # 尾端：用最後一幀把剩下的擠出來
            r = prec.push(last / fps + k * dt, X[last], frames[last][2], OK[last], None, cam)
            if r is not None and r[0] <= last / fps + 1e-6:
                res.append(r)
        for r in res:
            if r is None:
                continue
            i = int(round(r[0] * fps))
            if 0 <= i < len(out) and OK[i, P.HIP]:
                out[i] = r[1]
        return out

    # ------------------------------------------------------------ 匯出：圖片 → 姿勢
    def start_image_export(self, paths):
        self._start('export', self._image_export, list(paths or []))

    def _image_export(self, paths):
        vc = self.cfg['video']
        thr = self.cfg['pose']['score_thr']
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            raise RuntimeError('ERR_IMAGE_OPEN')
        est = self._estimator()
        height = self._height()
        gx = self._gemx() if vc.get('method') == 'gemx' else None      # 高精度：GEM-X（單張圖也能用）
        info = dict(kind='image', name=os.path.basename(paths[0]), done=0, total=len(paths), skipped=[])
        self._set(phase='analyzing', job=info)
        motions, saved, skipped = [], [], []
        est.tta = True
        try:
            for i, path in enumerate(paths):
                self._check_stop()
                name = os.path.splitext(os.path.basename(path))[0]
                try:
                    img = V.read_image(path)
                except Exception:
                    skipped.append(os.path.basename(path))
                    continue
                if vc.get('mirror'):
                    img = cv2.flip(img, 1)
                est.bbox = None
                est.n = 0
                k, z, sc = est(img)
                self._show(img, k, sc, thr)
                X = None
                if k is not None:
                    if gx is not None:
                        X = self._gemx_image(gx, img, k, sc, getattr(est, 'cue', None))
                    else:
                        X, ok = lift_image(k, z, sc, (img.shape[1], img.shape[0]), vc.get('hfov_deg', 60), height, thr)
                        if X is not None:
                            X = LB.Limbs()(X, ok, sc, cue=getattr(est, 'cue', None))[0]
                    if X is not None:
                        for side, d in (getattr(est, 'cue_detail', None) or {}).items():
                            self.log(f'{os.path.basename(path)} 頭與手（{side}）：{_CUE_TEXT[d["cue"]]}'
                                     f'（手蓋住 {d["under"]} 個臉部點，分數 {_r2(d["local"])}／臉的其他部分 {_r2(d["ref"])}，手 {d["hand"]}）')
                if X is None:
                    skipped.append(os.path.basename(path))
                else:
                    text = to_bvh(np.repeat(X[None], 2, axis=0), 30.0)
                    out = _unique(os.path.join(V.export_dir(), name + '_pose'), '.bvh')
                    with open(out, 'w', encoding='utf-8') as fh:
                        fh.write(text)
                    ap = ANIM.write(out[:-4] + '.anim', np.repeat(X[None], 2, axis=0), 30.0,
                                    os.path.splitext(os.path.basename(out))[0])
                    with open(ap, 'rb') as fh:
                        motions.append((os.path.basename(ap), fh.read()))
                    saved.append(out)
                self._set(progress=(i + 1) / len(paths), check=check(k, sc, thr, (img.shape[1], img.shape[0])),
                          job=dict(info, done=i + 1, skipped=list(skipped), name=os.path.basename(path)))
        finally:
            est.tta = False
        if not motions:
            raise RuntimeError('ERR_NO_PERSON_IMAGE')
        base = os.path.splitext(saved[0])[0] if len(saved) == 1 else \
            os.path.join(V.export_dir(), f'poses_{time.strftime("%Y%m%d_%H%M%S")}')
        pkg = unity_pkg.build(_unique(base, '.unitypackage'), motions)
        self.log(f'圖片姿勢匯出完成：{len(saved)} 張，略過 {len(skipped)} 張' + ('（GEM-X）' if gx is not None else ''))
        self._set(progress=1.0, job=dict(info, done=len(paths), skipped=skipped, export=saved[0], unity=pkg,
                                         saved=len(saved), **self._to_unity_project(motions)))

    # ------------------------------------------------------------ Unity
    def _to_unity_project(self, motions):
        proj = self.cfg['video'].get('unity_project') or ''
        if not proj:
            return dict(project=None, project_err=None)
        try:
            files = unity_pkg.install(proj, motions)
            self.log(f'已放進 Unity 專案：{files}')
            return dict(project=proj, project_err=None)
        except Exception as e:
            self.log(f'放進 Unity 專案失敗：{e}')
            return dict(project=None, project_err=str(e) if str(e).startswith('ERR_') else 'ERR_UNITY_PROJECT')

    def install_unity_tools(self):
        """Unity 工具：做成 .unitypackage 直接用 Unity 開啟（開著的專案會跳出匯入視窗）。"""
        try:
            out = unity_pkg.build(_unique(os.path.join(V.export_dir(), 'DesktopFBT_Tools'), '.unitypackage'), [])
            self.import_to_unity(out)
            return dict(package=out)
        except Exception as e:
            self.log(f'安裝 Unity 工具失敗：{e}')
            return dict(error=str(e) if str(e).startswith('ERR_') else 'ERR_UNITY_PROJECT')

    def set_unity_project(self, path):
        if not path:
            self.update_config({'video': {'unity_project': ''}})
            return ''
        proj = unity_pkg.find_project(path)
        if proj:
            self.update_config({'video': {'unity_project': proj}})
        return proj

    # ------------------------------------------------------------ 其他
    def open_exports(self, path=None):
        """開啟匯出資料夾；給了檔案時在檔案總管裡選取它。"""
        d = V.export_dir()
        if sys.platform == 'win32':
            if path and os.path.isfile(path):
                subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
            else:
                os.startfile(d)
        return d

    def import_to_unity(self, path):
        """用 Unity 開啟 .unitypackage（和在檔案總管雙擊一樣：開著的 Unity 專案會跳出匯入視窗）。"""
        if not path or not os.path.isfile(path):
            raise RuntimeError('ERR_UNITY_PROJECT')
        if sys.platform == 'win32':
            os.startfile(path)
        return path

    def open_folder(self):
        path = data()
        if sys.platform == 'win32':
            os.startfile(path)
        return path

    def environment(self):
        return dict(version=VERSION, devices=P.available_devices(), model=models.pose3d_ready(), folder=data())
