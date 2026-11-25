"""姿勢模型的下載與管理（存在 EXE 旁邊的 models/ 資料夾）。"""
import os
import shutil
import tempfile
import urllib.request
import uuid
import zipfile

from paths import data

# 找人：YOLOX-m（HumanArt）；姿勢：RTMW3D-x（全身 133 點＋相對深度）
_BASE = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
DET_URL = _BASE + 'yolox_m_8xb8-300e_humanart-c2c7a14a.zip'
DET_SIZE = (640, 640)
MIRROR = 'https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/'
WARNINGS = []          # 不致命的提醒（例如寫入測試檔刪不掉）；引擎取出後寫進 log.txt


def take_warnings():
    out = WARNINGS[:]
    del WARNINGS[:]
    return out


def models_dir():
    d = data('models')
    os.makedirs(d, exist_ok=True)
    return d


def _local_path(url):
    name = os.path.basename(url).rsplit('.', 1)[0] + '.onnx'
    return os.path.join(models_dir(), name)


def _category(step, e):
    """給使用者看的原因分類（介面翻譯 MODEL_WHY_<分類>）；詳細的例外只寫在記錄檔。"""
    import socket
    import urllib.error
    if step in ('delete failed', 'replace'):        # 換掉 / 刪除正式檔失敗：可能被使用中，也可能是權限
        return 'in_use'
    if step in LOCAL_STEPS:                          # 本機寫入失敗（模型資料夾、暫存資料夾、磁碟空間）
        return 'storage'
    if step == 'incomplete':
        return 'incomplete'
    if step == 'extract':
        return 'storage' if isinstance(e, OSError) else 'invalid'
    if step == 'cannot open':
        return 'unsupported'
    if step in ('not a zip', 'no .onnx', 'invalid model'):
        return 'invalid'
    if isinstance(e, (urllib.error.URLError, socket.timeout, ConnectionError)):
        return 'network'
    if step == 'download' and isinstance(e, OSError):   # 本機寫入已另外標示；這裡剩下的是傳輸過程的錯誤
        return 'network'
    return 'unknown'


# 這些步驟的失敗發生在這台電腦上，換下載來源也沒用：不再試鏡像
LOCAL_STEPS = ('write probe', 'temp folder', 'local write')


def _local_failure(step, e):
    return step in LOCAL_STEPS or step == 'replace' or (step == 'extract' and isinstance(e, OSError))


class ModelDownloadError(RuntimeError):
    """下載或檢查模型失敗（所有來源都試過了）。訊息是 ERR_MODEL_DOWNLOAD:<檔名>|<原因分類>，介面會顯示翻譯好的說明；
    每個來源的詳細錯誤用 detail() 寫進記錄檔。"""

    def __init__(self, file, attempts):
        self.file, self.attempts = file, attempts
        cat = _category(attempts[-1][1], attempts[-1][2]) if attempts else 'unknown'
        super().__init__(f'ERR_MODEL_DOWNLOAD:{file}|{cat}')

    def detail(self):
        return '; '.join(f'{src}: {step}: {type(e).__name__}: {e}' for src, step, e in self.attempts)


class _LocalWrite(Exception):
    """打開或寫入下載中的檔案（暫存資料夾的壓縮檔或 models 裡的 .part）失敗：本機的問題，不是網路。"""

    def __init__(self, original):
        super().__init__(repr(original))
        self.original = original


class _Abort(Exception):
    """進度回呼丟出的例外（例如使用者按停止）：原樣往外傳，不能被當成「這個來源壞了」而改用鏡像。"""

    def __init__(self, original):
        super().__init__(repr(original))
        self.original = original


def _download(url, dst, progress):
    """下載到 dst。有 Content-Length 時收到的位元組數必須相同，否則丟出例外（不會留下「看起來完整」的檔案）。"""
    req = urllib.request.Request(url, headers={'User-Agent': 'DesktopFBT'})
    with urllib.request.urlopen(req, timeout=30) as r:
        try:
            f = open(dst, 'wb')
        except OSError as e:
            raise _LocalWrite(e) from e
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        finished = False
        try:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                try:
                    f.write(chunk)
                except OSError as e:
                    raise _LocalWrite(e) from e
                done += len(chunk)
                try:
                    progress(done, total)
                except Exception as e:
                    raise _Abort(e) from e
            finished = True
        finally:
            try:
                f.close()
            except OSError as e:
                if finished:                         # 其他例外（例如按停止）正在傳出時，不要蓋掉它
                    raise _LocalWrite(e) from e
    if total and done != total:
        raise IOError(f'incomplete download: {done} of {total} bytes')


# 檢查失敗時，只有這個（檔案本身無法解析：截斷、亂碼、HTML）才算「模型檔損壞」。
# InvalidGraph（例如這版執行環境不支援某個運算）、Fail、記憶體不足等都不是檔案壞掉：不刪檔、不重新下載。
_DAMAGED = ('InvalidProtobuf',)


def _check_onnx(path):
    """用 onnxruntime 的 CPU 開一次（不用 GPU），確認檔案是完整可讀的模型；用完立刻釋放，Windows 才能換名。"""
    import gc
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    del s
    gc.collect()


def damaged(path):
    """這個已存在的模型檔是否確定壞掉（0 位元組，或 CPU 開啟時是格式錯誤）。其他失敗回傳 False，不刪檔。"""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return True
    try:
        _check_onnx(path)
        return False
    except Exception as e:
        return type(e).__name__ in _DAMAGED


def _remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _probe_write(folder):
    """真的要下載之前，在模型資料夾建立再刪除一個唯一的小檔，確認可以寫入（不建立資料夾）。
    建立失敗 → OSError 往外丟；建立成功但刪不掉 → 只記一筆提醒，不算失敗。"""
    p = os.path.join(folder, f'.desktopfbt-write-test-{os.getpid()}-{uuid.uuid4().hex}.tmp')
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        os.remove(p)
    except OSError as e:
        WARNINGS.append(f'模型資料夾的寫入測試檔刪不掉（不影響下載）：{os.path.basename(p)}（{type(e).__name__}）')


def _fetch(sources, dst, progress, zipped):
    """依序試每個來源（每個最多一次）：下載 →（壓縮檔：解出 .onnx）→ CPU 檢查 → 才換成正式檔名。
    下載前先確認模型資料夾可以寫入（不能寫入就不連任何來源）。來源的問題（網路、壞檔、下載不完整）換下一個來源；
    本機寫入的問題（資料夾、暫存資料夾、磁碟空間、換不掉正式檔）換來源也沒用，立刻停止。
    任何一步失敗都刪掉暫存檔；全部失敗丟出 ModelDownloadError，不會留下壞掉的正式檔。"""
    attempts = []
    part = dst + '.part'
    try:
        _probe_write(os.path.dirname(dst))
    except OSError as e:
        raise ModelDownloadError(os.path.basename(dst), [('local', 'write probe', e)]) from None
    for name, url in sources:
        try:
            tmp = tempfile.mkdtemp(prefix='desktopfbt_')
        except OSError as e:
            attempts.append((name, 'temp folder', e))
            raise ModelDownloadError(os.path.basename(dst), attempts) from None
        step = 'download'
        try:
            if zipped:
                z = os.path.join(tmp, 'm.zip')
                _download(url, z, progress)
                step = 'not a zip'
                with zipfile.ZipFile(z) as zf:
                    onnx = [n for n in zf.namelist() if n.endswith('.onnx')]
                    if not onnx:
                        step = 'no .onnx'
                        raise RuntimeError('模型壓縮檔裡沒有 .onnx')
                    step = 'extract'
                    with zf.open(onnx[0]) as src, open(part, 'wb') as f:
                        shutil.copyfileobj(src, f)
            else:
                _download(url, part, progress)
            step = 'invalid model'
            try:
                _check_onnx(part)
            except Exception as e:
                if os.path.getsize(part) and type(e).__name__ not in _DAMAGED:
                    step = 'cannot open'          # 檔案不像是壞掉，但這個執行環境開不起來（記憶體、版本等）
                raise
            step = 'replace'
            os.replace(part, dst)
            return
        except _Abort as a:
            _remove(part)
            raise a.original
        except _LocalWrite as w:
            attempts.append((name, 'local write', w.original))
            _remove(part)
            raise ModelDownloadError(os.path.basename(dst), attempts) from None
        except Exception as e:
            if step == 'download' and ('incomplete' in str(e).lower() or type(e).__name__ == 'IncompleteRead'):
                step = 'incomplete'
            attempts.append((name, step, e))
            _remove(part)
            if _local_failure(step, e):
                raise ModelDownloadError(os.path.basename(dst), attempts) from None
        except BaseException:                      # KeyboardInterrupt 等：不留下暫存檔，原樣往外傳
            _remove(part)
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise ModelDownloadError(os.path.basename(dst), attempts)


POSE3D_URL = ('https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/'
              'rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx')


def pose3d_ready():
    return os.path.exists(_local_path(DET_URL)) and os.path.exists(
        os.path.join(models_dir(), os.path.basename(POSE3D_URL)))


def ensure_pose3d(progress=lambda part, frac: None):
    """需要時下載模型。回傳 dict(det, det_size, pose3d)。偵測模型有主站＋鏡像；RTMW3D 只有一個來源。
    下載的檔案要能用 onnxruntime（CPU）開啟才算完成。"""
    det = _local_path(DET_URL)
    if not os.path.exists(det):
        _fetch([('primary', DET_URL), ('mirror', MIRROR + os.path.basename(DET_URL))], det,
               lambda d, t: progress('det', d / t if t else None), zipped=True)
    progress('det', 1.0)
    pose = os.path.join(models_dir(), os.path.basename(POSE3D_URL))
    if not os.path.exists(pose):
        _fetch([('huggingface', POSE3D_URL)], pose, lambda d, t: progress('pose', d / t if t else None), zipped=False)
    progress('pose', 1.0)
    return dict(det=det, det_size=DET_SIZE, pose3d=pose)


def _derived(src, tag, convert):
    """由原始模型轉出的檔案（存在旁邊；轉一次就好）。失敗時回傳 None 並記一筆提醒。"""
    dst = src[:-5] + tag + '.onnx'
    if os.path.exists(dst):
        return dst
    tmp = dst + '.part'
    try:
        import onnx
        m = convert(onnx.load(src))
        onnx.save(m, tmp)
        os.replace(tmp, dst)
        return dst
    except Exception as e:
        WARNINGS.append(f'模型轉換失敗（{tag}）：{type(e).__name__}: {e}')
        _remove(tmp)
        return None


def _batchable(m):
    for v in list(m.graph.input) + list(m.graph.output):
        v.type.tensor_type.shape.dim[0].dim_param = 'N'
    del m.graph.value_info[:]
    return m


def _half(m):
    import warnings
    from onnxconverter_common import float16
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float16.convert_float_to_float16(m, keep_io_types=True)


def batch_model(src):
    """第一維改成可變：一次送好幾幀（結果和一幀一幀算完全相同）。"""
    return _derived(src, '_nb', _batchable)


def fp16_model(src):
    """半精度＋可變批次（AMD／Intel／NVIDIA 用 DirectML 時通常快很多，檔案也小一半）。"""
    return _derived(src, '_nb_fp16', lambda m: _half(_batchable(m)))
