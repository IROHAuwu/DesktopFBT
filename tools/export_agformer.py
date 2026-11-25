"""MotionAGFormer（Apache-2.0）轉成 ONNX。打包時在 GitHub Actions 執行；程式本身不需要 PyTorch。

用法：python tools/export_agformer.py <MotionAGFormer 原始碼> <checkpoint 或 random> <base|small> <輸出.onnx>
輸入 (1, T, 17, 3)：H36M 17 點，畫面座標正規化（x / 寬 * 2 - 1）＋ 分數；輸出 (1, T, 17, 3)：相對骨盆的 3D。
"""
import sys

import torch

CFG = {
    'base': dict(n_layers=16, dim_feat=128, n_frames=243),
    'small': dict(n_layers=26, dim_feat=64, n_frames=81),
}


def main(src, ckpt, variant, out):
    sys.path.insert(0, src)
    from model.MotionAGFormer import MotionAGFormer
    c = CFG[variant]
    m = MotionAGFormer(n_layers=c['n_layers'], dim_in=3, dim_feat=c['dim_feat'], dim_rep=512, dim_out=3,
                       mlp_ratio=4, act_layer=torch.nn.GELU, attn_drop=0.0, drop=0.0, drop_path=0.0,
                       use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                       num_heads=8, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                       use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False,
                       graph_only=False, neighbour_num=2, n_frames=c['n_frames'])
    if ckpt != 'random':
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
        m.load_state_dict({k.replace('module.', '', 1): v for k, v in sd.items()}, strict=True)
    m.eval()
    x = torch.randn(1, c['n_frames'], 17, 3)
    with torch.no_grad():
        ref = m(x)
        torch.onnx.export(m, (x,), out, input_names=['x'], output_names=['y'], opset_version=17, dynamo=False,
                          dynamic_axes={'x': {0: 'n'}, 'y': {0: 'n'}})
    import numpy as np
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    y = s.run(None, {'x': x.numpy()})[0]
    err = float(np.abs(y - ref.numpy()).max())
    x2 = torch.randn(2, c['n_frames'], 17, 3)
    with torch.no_grad():
        ref2 = m(x2).numpy()
    err2 = float(np.abs(s.run(None, {'x': x2.numpy()})[0] - ref2).max())
    print(variant, 'max diff', err, 'batch 2', err2)
    assert err < 1e-3 and err2 < 1e-3


if __name__ == '__main__':
    main(*sys.argv[1:5])
