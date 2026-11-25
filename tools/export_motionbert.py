"""MotionBERT-Lite（時間序列 2D → 3D）轉成 ONNX。打包時在 GitHub Actions 執行一次；程式本身不需要 PyTorch。

用法：python tools/export_motionbert.py <MotionBERT 原始碼資料夾> <best_epoch.bin 或 random> <輸出.onnx>
模型：Walter0807/MotionBERT（Apache-2.0），FT_MB_lite_MB_ft_h36m_global_lite。
輸入 (1, 243, 17, 3)：H36M 17 點、正規化到 [-1, 1] 的 2D ＋ 分數；輸出 (1, 243, 17, 3)：3D（同樣的單位）。
"""
import sys
from functools import partial

import torch
import torch.nn as nn


def main(src, ckpt, out):
    sys.path.insert(0, src)
    from lib.model.DSTformer import DSTformer
    m = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8, mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6), maxlen=243, num_joints=17)
    if ckpt != 'random':
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_pos']
        m.load_state_dict({k.replace('module.', '', 1): v for k, v in sd.items()}, strict=True)
    m.eval()
    x = torch.randn(1, 243, 17, 3)
    with torch.no_grad():
        ref = m(x)
        torch.onnx.export(m, (x,), out, input_names=['x'], output_names=['y'], opset_version=17, dynamo=False,
                          dynamic_axes={'x': {0: 'n'}, 'y': {0: 'n'}})
    import numpy as np
    import onnxruntime as ort
    y = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, {'x': x.numpy()})[0]
    err = float(np.abs(y - ref.numpy()).max())
    print('max diff', err)
    assert err < 1e-3


if __name__ == '__main__':
    main(*sys.argv[1:4])
