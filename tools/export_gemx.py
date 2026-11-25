"""GEM-X（NVIDIA）的三個網路轉成 ONNX，並和官方 PyTorch 實作比對。在 GitHub Actions 執行；程式本身不需要 PyTorch。

用法：python tools/export_gemx.py <GEM-X 原始碼> <權重資料夾 或 random> <輸出資料夾> [SOMA 資產資料夾|none] [測試影片|none]
輸出：
  gemx_body_fp16.onnx(.data)   img (N,3,256,256)、cliff (N,3) → feat (N,1024)
  gemx_kp2d_fp16.onnx(.data)   img (N,3,256,192) → heat (N,77,64,48)
  gemx_motion.onnx             obs (1,L,77,3)、cliff (1,L,3)、angvel (1,L,6)、img (1,L,1024)、mask (L,L) bool、pos (1,L)
                               → x (1,L,655)（已還原正規化、體型已平均）、cam (1,L,3)、static (1,L,6)
  gemx_skeleton.npz            SOMA 零姿勢關節位置、父關節、名稱（有 SOMA 資產時）
  gemx_report.json             每一項比對的數字
random：用很小的隨機模型只檢查流程（本機、沒有權重時）。
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gemx as G  # noqa: E402

REPORT = {}
KP_RELEASE = 'vitpose77-dinov3-huge-metrosim-256x192-v1'
NET_CFG = dict(output_dim=655, max_len=120, cliffcam_dim=3, cam_angvel_dim=6, imgseq_dim=1024, latent_dim=512,
               num_layers=12, num_heads=8, mlp_ratio=4.0, pred_cam_dim=3, static_conf_dim=6, dropout=0.1,
               avgbeta=True, obs_num_joints=77, sequence_static_slices=[[456, 584], [584, 640]])
COND = dict(pose_release_id='gem-pose-body-s240000-v1:pose_tokens', observation_source=KP_RELEASE,
            observation_joint_set='soma77_native_v1')


def note(key, value):
    REPORT[key] = value
    print(f'[report] {key} = {value}', flush=True)


def stage(name, fn):
    """每一段各自失敗不影響其他段：一次 CI 就能看到全部結果。"""
    print(f'\n===== {name} =====', flush=True)
    t = time.time()
    try:
        out = fn()
        note(f'{name}.ok', True)
        return out
    except Exception as e:
        traceback.print_exc()
        note(f'{name}.ok', False)
        note(f'{name}.error', f'{type(e).__name__}: {e}'[:400])
        return None
    finally:
        note(f'{name}.seconds', round(time.time() - t, 1))


def small_config(bf16=False):
    from transformers import DINOv3ViTConfig
    return DINOv3ViTConfig(image_size=224, patch_size=16, num_channels=3, hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=128, use_gated_mlp=True, hidden_act='silu',
                           num_register_tokens=4, rope_theta=100.0, query_bias=True, key_bias=False,
                           value_bias=True, proj_bias=True, mlp_bias=True, layerscale_value=1e-5,
                           layer_norm_eps=1e-6 if bf16 else 1e-5, pos_embed_rescale=None if bf16 else 2.0,
                           drop_path_rate=0.0, _attn_implementation='sdpa')


class ConstRope(nn.Module):
    """輸入大小固定：位置編碼的 cos/sin 先用原本的模組算好（保留它的進位方式），匯出時當常數。"""

    def __init__(self, rope, shape):
        super().__init__()
        with torch.no_grad():
            cos, sin = rope(torch.zeros(1, 3, *shape))
        self.register_buffer('cos', cos.float())
        self.register_buffer('sin', sin.float())

    def forward(self, pixel_values):
        return self.cos.to(pixel_values.dtype), self.sin.to(pixel_values.dtype)


def save_single(path):
    """整理成「一個 .onnx ＋ 一個 .onnx.data」，清掉匯出時產生的零散權重檔。"""
    import onnx
    d, base = os.path.dirname(path), os.path.basename(path)
    before = set(os.listdir(d))
    m = onnx.load(path)
    data = base + '.data'
    if os.path.exists(os.path.join(d, data)):
        os.remove(os.path.join(d, data))
    onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=1024)
    del m
    keep = {f for f in before if f.startswith('gemx_') and (f.endswith(('.onnx', '.onnx.data', '.npz', '.json')))}
    for f in os.listdir(d):
        if f not in keep and f != data and f != base:
            os.remove(os.path.join(d, f))
    size = os.path.getsize(path) + os.path.getsize(os.path.join(d, data))
    return round(size / 2 ** 20, 1)


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


# ---------------------------------------------------------------- Body Pose 特徵
class BodyFeat(nn.Module):
    """backbone → 解碼器最後一層的查詢向量（時序模型用的 1024 維特徵）；不做後面的參數解碼與 FK。"""

    def __init__(self, pose, half):
        super().__init__()
        self.backbone, self.head, self.half_ = pose.backbone, pose.task.head, half

    def forward(self, img, cliff):
        if self.half_:
            img, cliff = img.half(), cliff.half()
        x = self.backbone(img)
        B = x.shape[0]
        feat = x.flatten(2).transpose(1, 2)
        token = self.head._build_token(B, x.device, x.dtype, cliff)
        h = self.head.transformer(token, context=feat, return_intermediate=False)
        return h[:, 0].float()


def load_body(src_weights, random):
    from gem.pose.api import body_config, build_pose_model, load_pose_checkpoint
    release = body_config()
    if random:
        cfg = small_config()
        release.backbone = {'name': 'dinov3_vith16plus', 'config': cfg}
        release.head['transformer_context_dim'] = cfg.hidden_size
        return build_pose_model(release).eval()
    return load_pose_checkpoint(os.path.join(src_weights, 'body_s240000.ckpt'), release, device='cpu')


def export_body(weights, random, out, crops, cliffs):
    """crops (N,3,256,256)、cliffs (N,3)：比對用的輸入（真影片的裁切，或隨機）。"""
    pose = load_body(weights, random)
    x, c = torch.from_numpy(crops), torch.from_numpy(cliffs)
    K = torch.eye(3).repeat(len(x), 1, 1)                     # cliff = (框 − 主點, 邊長) / 焦距 → 用 f=1、主點 0 還原
    batch = {'img': x, 'bbx_xys': c, 'K_fullimg': K}
    with torch.inference_mode():
        t = time.time()
        ref = pose.encode(batch).pose_tokens.float().numpy()
        note('body.torch_fp32_s_per_frame', round((time.time() - t) / len(x), 2))
        mine = BodyFeat(pose, False).eval()(x, c).numpy()
    note('body.wrapper_vs_official', float(np.abs(mine - ref).max()))
    assert np.abs(mine - ref).max() < 1e-4
    pose.backbone.encoder.rope_embeddings = ConstRope(pose.backbone.encoder.rope_embeddings, (256, 256))
    pose = pose.half()
    dst = os.path.join(out, 'gemx_body_fp16.onnx')
    m = BodyFeat(pose, True).eval()
    with torch.no_grad():
        torch.onnx.export(m, (x[:1], c[:1]), dst, input_names=['img', 'cliff'], output_names=['feat'],
                          dynamic_axes={'img': {0: 'n'}, 'cliff': {0: 'n'}, 'feat': {0: 'n'}},
                          opset_version=17, dynamo=False)
    del m, pose
    note('body.onnx_mb', save_single(dst))
    s = session(dst)
    t = time.time()
    y = np.concatenate([s.run(None, {'img': x[i:i + 2].numpy(), 'cliff': c[i:i + 2].numpy()})[0]
                        for i in range(0, len(x), 2)])
    note('body.onnx_cpu_s_per_frame', round((time.time() - t) / len(x), 2))
    rel = float(np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9))
    cos = float(((y * ref).sum(-1) / (np.linalg.norm(y, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-9)).min())
    note('body.fp16_rel_diff', rel)
    note('body.fp16_min_cosine', cos)
    return ref, y


# ---------------------------------------------------------------- 2D 關鍵點
class PlainAttention(nn.Module):
    """官方 _BF16DetectorAttention 去掉 bfloat16 來回轉換的版本（ONNX Runtime 沒有 bfloat16）。
    合併的 QKV 投影照舊；Q/K 的旋轉改用模型本身的精度。和官方的差距由比對結果決定能不能用。"""

    def __init__(self, a):
        super().__init__()
        self.num_heads, self.head_dim, self.scaling = a.num_heads, a.head_dim, a.scaling
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = a.q_proj, a.k_proj, a.v_proj, a.o_proj

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb
        B, N, _ = hidden_states.shape
        w = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight))
        b = torch.cat((self.q_proj.bias, torch.zeros_like(self.q_proj.bias), self.v_proj.bias))
        qkv = F.linear(hidden_states, w, b).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        o = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        return self.o_proj(o.transpose(1, 2).reshape(B, N, -1)), None


class KpNet(nn.Module):
    def __init__(self, model, half):
        super().__init__()
        self.m, self.half_ = model, half

    def forward(self, img):
        return self.m(img.half() if self.half_ else img).float()


def load_kp(weights, random):
    from gem.common.backbones import DinoV3Backbone
    from gem.common.pose2d.native77 import Native77Model, Native77Pose2D, load_native77_detector
    if random:
        model = Native77Model(DinoV3Backbone(config=small_config(bf16=True), rotary_precision='bf16')).eval()
        return Native77Pose2D(model, release_id=KP_RELEASE, batch_size=4)
    return load_native77_detector(os.path.join(weights, 'vitpose.pth'), release_id=KP_RELEASE, device='cpu',
                                  batch_size=4)


def export_kp(weights, random, out, frames, boxes):
    """frames：RGB uint8 的清單；boxes (N,3)。和官方 predict（含左右翻轉平均）比關鍵點的畫素差。"""
    det = load_kp(weights, random)
    t = time.time()
    ref = det.predict(frames, torch.from_numpy(boxes)).numpy()
    note('kp2d.torch_s_per_frame', round((time.time() - t) / len(frames), 2))
    model = det.model
    enc = model.backbone.encoder
    enc.rope_embeddings = ConstRope(enc.rope_embeddings, (256, 192))
    for layer in enc.layer:
        layer.attention = PlainAttention(layer.attention)
    crops = np.stack([G.crop_kp(f, b) for f, b in zip(frames, boxes)])

    def keypoints(run):
        both = np.concatenate([run(np.concatenate([crops[i:i + 1], crops[i:i + 1, ..., ::-1]]))
                               for i in range(len(crops))])
        heat = 0.5 * (both[0::2] + G.flip_heatmaps(both[1::2]))
        return G.decode_heatmaps(heat, boxes), G.decode_heatmaps(both[0::2], boxes)

    def compare(tag, kp):
        d = np.linalg.norm(kp[..., :2] - ref[..., :2], axis=-1) / boxes[:, None, 2] * 256      # 換算成裁切圖上的畫素
        sure = ref[..., 2] > 0.5
        note(f'kp2d.{tag}_px_mean', float(d.mean()))
        note(f'kp2d.{tag}_px_max', float(d.max()))
        note(f'kp2d.{tag}_px_max_confident', float(d[sure].max()) if sure.any() else None)
        note(f'kp2d.{tag}_conf_max_diff', float(np.abs(kp[..., 2] - ref[..., 2]).max()))

    official = os.path.join(weights, 'onnx', 'vitpose.onnx') if not random else ''
    if os.path.exists(official):                                    # NVIDIA 自己放的 ONNX（全精度）
        try:
            so = session(official)
            note('kp2d.official_onnx_io', [[i.name, str(i.shape), i.type] for i in so.get_inputs() + so.get_outputs()])
            name = so.get_inputs()[0].name
            t = time.time()
            kpo, _ = keypoints(lambda a: so.run(None, {name: np.ascontiguousarray(a)})[0])
            note('kp2d.official_onnx_cpu_s_per_frame', round((time.time() - t) / len(crops), 2))
            compare('official_onnx', kpo)
            del so
        except Exception as e:
            note('kp2d.official_onnx_error', f'{type(e).__name__}: {e}'[:300])
    with torch.inference_mode():
        f32 = KpNet(model, False).eval()
        kp32, _ = keypoints(lambda a: f32(torch.from_numpy(np.ascontiguousarray(a))).numpy())
    compare('fp32_no_bf16', kp32)
    note('kp2d.confident_joints_per_frame', float((ref[..., 2] > 0.5).sum(-1).mean()))
    model = model.half()
    dst = os.path.join(out, 'gemx_kp2d_fp16.onnx')
    m = KpNet(model, True).eval()
    with torch.no_grad():
        torch.onnx.export(m, (torch.from_numpy(crops[:1]),), dst, input_names=['img'], output_names=['heat'],
                          dynamic_axes={'img': {0: 'n'}, 'heat': {0: 'n'}}, opset_version=17, dynamo=False)
    del m, model, det
    note('kp2d.onnx_mb', save_single(dst))
    s = session(dst)
    t = time.time()
    kp16, kp16_noflip = keypoints(lambda a: s.run(None, {'img': np.ascontiguousarray(a)})[0])
    note('kp2d.onnx_cpu_s_per_frame', round((time.time() - t) / len(crops), 2))
    compare('onnx_fp16', kp16)
    compare('onnx_fp16_noflip', kp16_noflip)
    return ref, kp16


# ---------------------------------------------------------------- 時序模型
class Motion(nn.Module):
    def __init__(self, net, mean, std, ang_mean, ang_std):
        super().__init__()
        self.net = net
        self.register_buffer('mean', mean.float().clone())
        self.register_buffer('std', std.float().clone())
        self.register_buffer('ang_mean', ang_mean.float().clone())
        self.register_buffer('ang_std', ang_std.float().clone())

    def forward(self, obs, cliff, angvel, img, mask, pos):
        n, d = self.net, self.net.denoiser
        B, L = obs.shape[0], obs.shape[1]
        vis = obs[..., 2:3] > 0.5
        enc = n.learned_pos_linear(torch.where(vis, obs[..., :2], torch.zeros_like(obs[..., :2])))
        enc = torch.where(vis, enc, n.learned_pos_params[None, None]).flatten(-2)
        x = n.embed_noisyobs(enc) + d.cliffcam_embedder(cliff)
        x = x + d.cam_angvel_embedder((angvel - self.ang_mean) / self.ang_std) + d.imgseq_embedder(img)
        x = x + d.embed_timestep(torch.full((1,), 999, dtype=torch.long))
        x = d.add_cond_linear(torch.cat((x, torch.zeros_like(x[..., :1]).expand(-1, -1, 655)), dim=-1))
        for block in d.blocks:
            x = block(x, attn_mask=mask, tgt_key_padding_mask=None, positions=pos)
        sample = d.final_layer(x)
        static = sample[..., 456:640].mean(1, keepdim=True).expand(-1, L, -1)      # 體型、比例：整段一個
        sample = torch.cat((sample[..., :456], static, sample[..., 640:]), dim=-1)
        sample = sample * self.std + self.mean
        cam = d.pred_cam_head(x) * d.pred_cam_std + d.pred_cam_mean
        cam = torch.cat((cam[..., :1].clamp(min=0.25), cam[..., 1:]), dim=-1)
        return sample, cam, d.static_conf_head(x)


def load_motion(weights, random):
    from gem.motion.api import MotionConditionSpec, MotionPredictor
    from gem.motion.codec.soma_native_fixed_gs1 import SomaNativeFixedGs1EnDecoder
    from gem.motion.network import R3MatchedGEMNetwork
    from gem.motion.pipeline import MotionPipeline
    time_options = dict(reference_fps=30.0, default_fps=30.0, fps_by_data_name={})
    if random:
        from gem.inference.motion import TimedMotionPipeline
        torch.manual_seed(0)
        net = R3MatchedGEMNetwork(**NET_CFG)
        for p in net.parameters():                               # 零初始化的層給點數值，才比得出差異
            if float(p.abs().sum()) == 0:
                nn.init.normal_(p, std=0.02)
        codec = SomaNativeFixedGs1EnDecoder(stats_mean=torch.randn(655) * 0.1, stats_std=torch.rand(655) + 0.5)
        pipe = TimedMotionPipeline(MotionPipeline(net, codec).eval(), time_options).eval()
        return MotionPredictor(pipe, MotionConditionSpec(**COND))
    from gem.inference.motion import load_predictor
    cfg = dict(network=NET_CFG, condition_spec=COND, time_options=time_options)
    return load_predictor(os.path.join(weights, 'motion_native77_e774_s292175.ckpt'), cfg,
                          stats_path=os.path.join(weights, 'motion_stats_raw.npz'), device='cpu')


def motion_inputs(L, size, kp, boxes, feats):
    """和 App 一樣的做法準備時序模型的輸入（固定相機、30fps）。"""
    K = G.intrinsics(*size)
    return dict(obs=G.observations(kp, boxes)[None], cliff=G.cliff(boxes, K)[None],
                angvel=np.tile(np.float32([1, 0, 0, 0, 1, 0]), (1, L, 1)), img=feats[None].astype(np.float32),
                mask=G.attn_mask(L), pos=np.arange(L, dtype=np.float32)[None]), K


def official_motion(pred, L, size, kp, boxes, feats):
    K = torch.from_numpy(G.intrinsics(*size))
    seq = {'condition_spec': dict(COND), 'length': torch.tensor([L]), 'f_imgseq': torch.from_numpy(feats)[None].float(),
           'kp2d_native77': torch.from_numpy(kp)[None].float(), 'bbx_xys': torch.from_numpy(boxes)[None].float(),
           'K_fullimg': K.expand(L, 3, 3)[None].clone(),
           'cam_angvel': torch.tensor([1., 0, 0, 0, 1, 0]).expand(1, L, 6).clone(),
           'frame_times': (torch.arange(L, dtype=torch.float64) / 30)[None], 'sampling_fps': 30.0}
    return pred.run(seq)


def compare_motion(tag, x, cam, off, boxes, K):
    """App 的解碼（numpy）對官方管線的輸出：關節旋轉、相機座標與世界座標的根。"""
    from gem.common.geometry.rotations import axis_angle_to_matrix as aa
    d = G.decode(x)
    ic, gl = off['pred_soma_params_incam'], off['pred_soma_params_global']
    L = len(x)

    def geo(a, b):                                                  # 兩組旋轉矩陣的夾角（度）
        t = np.einsum('...ij,...ij->...', a, b)
        return float(np.degrees(np.arccos(np.clip((t - 1) / 2, -1, 1))).max())
    note(f'motion.{tag}_body_pose_deg', geo(d['R_body'], aa(ic['body_pose'][0].reshape(L, 76, 3)).numpy()))
    note(f'motion.{tag}_root_cam_deg', geo(d['R_c'], aa(ic['global_orient'][0]).numpy()))
    note(f'motion.{tag}_transl_cam_m', float(np.abs(G.camera_transl(cam, boxes, K) - ic['transl'][0].numpy()).max()))
    R_w, t_w = G.world_static(d['R_gv'], d['vel'])
    note(f'motion.{tag}_root_world_deg', geo(R_w, aa(gl['global_orient'][0]).numpy()))
    note(f'motion.{tag}_transl_world_m', float(np.abs(t_w - gl['transl'][0].numpy()).max()))
    note(f'motion.{tag}_identity_diff', float(np.abs(d['identity'] - ic['identity_coeffs'][0].numpy()).max()))
    note(f'motion.{tag}_scale_diff', float(np.abs(d['scale'] - ic['scale_params'][0, :, 1:].numpy()).max()))
    return d, R_w, t_w


def export_motion(weights, random, out, real):
    """real：(size, kp, boxes, feats) 真影片算出來的條件；另外一定會跑一段 150 幀的合成輸入（測超過 120 幀的遮罩）。"""
    pred = load_motion(weights, random)
    pipe = pred.pipeline.pipeline
    net, codec = pipe.denoiser3d, pipe.endecoder
    m = Motion(net, codec.mean, codec.std, pipe.cam_angvel_mean, pipe.cam_angvel_std).eval()
    rng = np.random.default_rng(0)
    L, size = 150, (1280, 720)
    boxes = np.tile(np.float32([640, 360, 500]), (L, 1)) + rng.normal(0, 3, (L, 3)).astype(np.float32)
    kp = np.concatenate([rng.uniform(300, 980, (L, 77, 2)), rng.uniform(0, 1, (L, 77, 1))], -1).astype(np.float32)
    feats = rng.normal(0, 1, (L, 1024)).astype(np.float32)
    feed, K = motion_inputs(L, size, kp, boxes, feats)
    dst = os.path.join(out, 'gemx_motion.onnx')
    dyn = {k: {1: 'L'} for k in ('obs', 'cliff', 'angvel', 'img', 'pos', 'x', 'cam', 'static')}
    dyn['mask'] = {0: 'L', 1: 'L'}
    from gem.motion.network import rotary

    def rotate(self, x, positions=None):
        # 原本用 repeat_interleave(2, -1)：舊版 ONNX 匯出器會把別處 reshape 的 -1 一起改成 2。算式相同，只在匯出時換掉。
        enc = positions.float()[..., None] * self.inv_freq.float()
        return rotary.apply_rotary_emb(torch.stack((enc, enc), dim=-1).flatten(-2)[:, None], x)
    original = rotary.ROPE.rotate_queries_or_keys
    rotary.ROPE.rotate_queries_or_keys = rotate
    try:
        with torch.no_grad():
            torch.onnx.export(m, tuple(torch.from_numpy(v) for v in feed.values()), dst, input_names=list(feed),
                              output_names=['x', 'cam', 'static'], dynamic_axes=dyn, opset_version=17, dynamo=False)
    finally:
        rotary.ROPE.rotate_queries_or_keys = original
    note('motion.onnx_mb', round(os.path.getsize(dst) / 2 ** 20, 1))
    s = session(dst)
    cases = [('synthetic150', size, kp, boxes, feats)]
    if real is not None:
        if len(real[1]) > 3:
            cases.append(('video_short', real[0], real[1][:3], real[2][:3], real[3][:3]))   # 長度會變：再測一個
        cases.append(('video', *real))
    last = None
    for tag, size, kp, boxes, feats in cases:
        L = len(kp)
        feed, K = motion_inputs(L, size, kp, boxes, feats)
        t = time.time()
        x, cam, static = s.run(None, feed)
        note(f'motion.{tag}_onnx_cpu_s', round(time.time() - t, 2))
        with torch.inference_mode():
            off = official_motion(pred, L, size, kp, boxes, feats)
        note(f'motion.{tag}_static_logits_diff', float(np.abs(static[0] - off['static_conf_logits'][0].numpy()).max()))
        d, R_w, t_w = compare_motion(tag, x[0], cam[0], off, boxes, K)
        last = dict(d=d, R_w=R_w, t_w=t_w, off=off, L=L)
    return last


# ---------------------------------------------------------------- SOMA 骨架
def export_skeleton(soma_assets, out, motion):
    """用官方 SOMA 執行環境取出零姿勢的關節位置，並驗證 App 的簡化 FK（平均體型）和官方 FK 的差距。"""
    from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
    from gem.common.geometry.rotations import matrix_to_axis_angle as m2aa
    body = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=soma_assets, device='cpu', lod='xlo',
                                                     apply_correctives=False, enable_procedural_transforms=True,
                                                     procedural_reference='runtime'))

    def joints(R_root, R_body, identity, scale, transl):
        L = len(R_root)
        with torch.inference_mode():
            o = body(global_orient=m2aa(torch.from_numpy(R_root).float()),
                     body_pose=m2aa(torch.from_numpy(R_body).float()).reshape(L, 76 * 3),
                     identity_coeffs=torch.from_numpy(identity).float(), scale_params=torch.from_numpy(scale).float(),
                     transl=torch.from_numpy(transl).float(), pose2rot=True, apply_correctives=False,
                     return_joints_only=True)
        return o['joints'].numpy().reshape(L, 77, 3)

    eye = np.eye(3)[None]
    rest = joints(eye, np.tile(np.eye(3), (1, 76, 1, 1)), np.zeros((1, 128)), np.ones((1, 57)), np.zeros((1, 3)))[0]
    rt = body.runtime
    names = [str(n) for n in rt.public_joint_names]
    par = np.asarray(rt.output_joint_parent_ids.cpu()).astype(int)
    note('skeleton.public_names', names)
    note('skeleton.public_parents', par.tolist())
    off = len(names) - 77                                          # 公開骨架多一個虛擬的 Root
    parents = np.array([p - off if p >= off else -1 for p in par[off:]])
    names = names[off:]
    assert parents[0] == -1 and all(parents[j] < j for j in range(1, 77)), parents
    note('skeleton.rest_root', rest[0].round(4).tolist())
    np.savez(os.path.join(out, 'gemx_skeleton.npz'), rest=rest.astype(np.float32), parents=parents.astype(np.int16),
             names=np.array(names))

    def bone_deg(a, b):
        va, vb = a[:, 1:] - a[:, parents[1:]], b[:, 1:] - b[:, parents[1:]]
        c = (va * vb).sum(-1) / (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1) + 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    # 1) 平均體型、隨機姿勢：位置應該完全一樣
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(1)
    L = 6
    R = Rotation.from_rotvec(rng.normal(0, 0.5, (L * 77, 3))).as_matrix().reshape(L, 77, 3, 3)
    tr = rng.normal(0, 1, (L, 3))
    a = joints(R[:, 0], R[:, 1:], np.zeros((L, 128)), np.ones((L, 57)), tr)
    b = G.fk(rest, parents, R[:, 0], R[:, 1:], tr)
    note('skeleton.neutral_root_offset_m', (a[:, 0] - tr).mean(0).round(4).tolist())
    b = b + (a[:, :1] - b[:, :1])
    note('skeleton.neutral_random_pose_max_m', float(np.abs(a - b).max()))
    note('skeleton.neutral_random_pose_bone_deg', float(bone_deg(a, b).max()))
    # 2) 模型預測的體型與姿勢：App 用平均體型，只比骨頭方向
    if motion is not None:
        d, L = motion['d'], motion['L']
        scale = np.concatenate([np.ones((L, 1)), d['scale']], -1)
        a = joints(motion['R_w'], d['R_body'], d['identity'], scale, motion['t_w'])
        b = G.fk(rest, parents, motion['R_w'], d['R_body'], motion['t_w'])
        deg = bone_deg(a, b)
        note('skeleton.predicted_shape_bone_deg_mean', float(deg.mean()))
        note('skeleton.predicted_shape_bone_deg_max', float(deg.max()))
        worst = np.argsort(deg.max(0))[-5:][::-1]
        note('skeleton.predicted_shape_worst_bones', [f'{names[j + 1]}:{deg.max(0)[j]:.1f}' for j in worst])
        la = np.linalg.norm(a[:, 1:] - a[:, parents[1:]], axis=-1).mean(0)
        lb = np.linalg.norm(b[:, 1:] - b[:, parents[1:]], axis=-1).mean(0)
        note('skeleton.predicted_shape_length_ratio_range', [float((la / (lb + 1e-9)).min()), float((la / (lb + 1e-9)).max())])
        g = motion['off']['pred_soma_params_global']
        with torch.inference_mode():
            o = body(**{k: v[0] for k, v in g.items()}, pose2rot=True, apply_correctives=False, return_joints_only=True)
        ref = o['joints'].numpy().reshape(L, 77, 3)
        note('skeleton.end_to_end_bone_deg_max', float(bone_deg(ref, b).max()))


# ---------------------------------------------------------------- 測試影片
def video_inputs(src, weights, video, n):
    """影片中段連續的 n 幀 ＋ 人物框。有官方偵測器就用，沒有就用畫面中央。"""
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total // 3, total - n)))     # 連續的 n 幀（30fps），追蹤和時序模型才有意義
    frames = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(f[..., ::-1]))
    cap.release()
    assert frames, 'no frames'
    h, w = frames[0].shape[:2]
    note('video.size', [w, h])
    note('video.frames', len(frames))
    boxes = None
    yolo = os.path.join(weights or '', 'yolox_humanart.onnx')
    if os.path.exists(yolo):
        try:
            from gem.common.detectors import YOLOXONNX, detect_track
            tracks = detect_track(frames, YOLOXONNX(yolo, device='cpu'))
            boxes = np.asarray(tracks.boxes_xys, np.float32)
            note('video.boxes', 'official YOLOX + tracker')
        except Exception as e:
            note('video.detector_error', f'{type(e).__name__}: {e}'[:300])
    if boxes is None:
        boxes = np.tile(np.float32([w / 2, h / 2, min(w, h) * 0.9]), (len(frames), 1))
        note('video.boxes', 'centre box (no detector)')
    note('video.first_box', boxes[0].round(1).tolist())
    return frames, boxes, (w, h)


def main(src, weights, out, soma='none', video='none'):
    sys.path.insert(0, src)
    os.makedirs(out, exist_ok=True)
    random = weights == 'random'
    torch.set_grad_enabled(False)
    note('torch', torch.__version__)
    n = 3 if random else 8
    if video != 'none' and os.path.exists(video):
        frames, boxes, size = stage('video', lambda: video_inputs(src, None if random else weights, video, n)) or (None,) * 3
    else:
        frames = None
    if frames is None:
        rng = np.random.default_rng(0)
        size = (640, 480)
        frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(n)]
        boxes = np.tile(np.float32([320, 240, 400]), (n, 1))
    K = G.intrinsics(*size)
    crops = np.stack([G.crop_body(f, b) for f, b in zip(frames, boxes)])
    cliffs = G.cliff(boxes, K)

    def check_crop():
        from gem.pose.data.crop import prepare_pose_batch
        from gem.common.pose2d.native77 import crop_native77_inputs
        b = prepare_pose_batch(frames, boxes, torch.from_numpy(K), crop_convention='motion_training')
        note('crop.body_max_diff', float(np.abs(b['img'].numpy() - crops).max()))
        k = crop_native77_inputs(frames, torch.from_numpy(boxes)).numpy()
        note('crop.kp_max_diff', float(np.abs(k - np.stack([G.crop_kp(f, bb) for f, bb in zip(frames, boxes)])).max()))
        from gem.pose.conditioning.cliff import cliff_features
        c = cliff_features(b['bbx_xys'], b['K_fullimg'][..., 0, 0], b['K_fullimg'][..., :2, 2]).numpy()
        note('crop.cliff_max_diff', float(np.abs(c - cliffs).max()))
    stage('crop', check_crop)
    body = stage('body', lambda: export_body(weights, random, out, crops, cliffs))
    kp = stage('kp2d', lambda: export_kp(weights, random, out, frames, boxes))
    real = None
    if body is not None and kp is not None:
        real = (size, kp[0], boxes, body[0])                       # 官方的條件 → 比時序模型本身
    motion = stage('motion', lambda: export_motion(weights, random, out, real))
    if soma != 'none':
        stage('skeleton', lambda: export_skeleton(soma, out, motion))
    files = {f: round(os.path.getsize(os.path.join(out, f)) / 2 ** 20, 1) for f in sorted(os.listdir(out))}
    note('files_mb', files)
    with open(os.path.join(out, 'gemx_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(REPORT, fh, indent=1, ensure_ascii=False)
    failed = [k for k, v in REPORT.items() if k.endswith('.ok') and not v]
    print('\nFAILED:', failed if failed else 'none')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:6]))
