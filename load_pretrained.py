"""
将 ContextCluster-small ImageNet 预训练权重加载到 EfficientVRNet 的视觉流。

差异处理：
1. network 索引偏移：VRCoC 在 stage blocks 之间插入了 fusion/reducer 模块
2. token_mixer 属性名：CoC 用 f/proj/v，VRCoC 用 fc1/fc2/fc_v
3. mlp 权重形状：CoC 是 Linear (C_out, C_in)，VRCoC 是 Conv2d (C_out, C_in, 1, 1)
4. 只加载视觉流（network），跳过 patch_embed_radar、radar_enhance 等雷达专属模块
"""

import torch


# CoC network index → VRCoC network index（跳过 fusion 和 reducer 占用的位置）
NETWORK_IDX_MAP = {
    0: 0,   # stage 1 ClusterBlocks
    1: 2,   # down 1 (PointRecuder)
    2: 3,   # stage 2 ClusterBlocks
    3: 5,   # down 2
    4: 6,   # stage 3 ClusterBlocks
    5: 8,   # down 3
    6: 9,   # stage 4 ClusterBlocks
}

# token_mixer 属性名映射
MIXER_NAME_MAP = {
    'token_mixer.f.':    'token_mixer.fc1.',
    'token_mixer.proj.': 'token_mixer.fc2.',
    'token_mixer.v.':    'token_mixer.fc_v.',
}


def _remap_key(key: str) -> str | None:
    """
    将 CoC checkpoint 的 key 转换为 VRCoC 对应的 key。
    返回 None 表示该 key 不应被加载（如分类 head、norm 等）。
    """
    # 跳过分类层
    if key.startswith('head.') or key.startswith('norm.'):
        return None

    # patch_embed 直接映射（形状相同：CoC 也使用 5 通道输入）
    if key.startswith('patch_embed.'):
        return 'backbone.backbone.' + key

    # network.N.xxx 需要重映射索引
    if key.startswith('network.'):
        parts = key.split('.')
        coc_idx = int(parts[1])
        vr_idx = NETWORK_IDX_MAP.get(coc_idx)
        if vr_idx is None:
            return None
        rest = '.'.join(parts[2:])

        # token_mixer 属性名转换
        for old, new in MIXER_NAME_MAP.items():
            if old in rest:
                rest = rest.replace(old, new)
                break

        return f'backbone.backbone.network.{vr_idx}.' + rest

    return None


def _fix_weight_shape(key: str, tensor: torch.Tensor, model_sd: dict) -> torch.Tensor:
    """
    mlp Linear 权重 (C_out, C_in) → Conv2d 权重 (C_out, C_in, 1, 1)。
    """
    if 'mlp.fc' in key and key in model_sd:
        target_shape = tuple(model_sd[key].shape)
        if tensor.dim() == 2 and len(target_shape) == 4:
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    return tensor


def load_coc_pretrained(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    从 CoC ImageNet checkpoint 加载权重到 EfficientVRNet 的视觉流。

    - phi 必须为 'l'（width=1.0）才与 coc_small 权重维度匹配。
    - 雷达流（network_radar、patch_embed_radar 等）保持随机初始化。
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_sd = ckpt.get('state_dict', ckpt.get('model', ckpt))

    model_sd = model.state_dict()
    loaded, skipped_mismatch, skipped_nomap = [], [], []

    for src_key, src_tensor in src_sd.items():
        dst_key = _remap_key(src_key)
        if dst_key is None:
            skipped_nomap.append(src_key)
            continue

        if dst_key not in model_sd:
            skipped_nomap.append(src_key)
            continue

        src_tensor = _fix_weight_shape(dst_key, src_tensor, model_sd)

        if src_tensor.shape != model_sd[dst_key].shape:
            skipped_mismatch.append(
                f'{src_key} {tuple(src_tensor.shape)} → {dst_key} {tuple(model_sd[dst_key].shape)}'
            )
            continue

        model_sd[dst_key] = src_tensor
        loaded.append(dst_key)

    model.load_state_dict(model_sd)

    print(f'[pretrain] 成功加载 {len(loaded)} 个权重')
    if skipped_mismatch:
        print(f'[pretrain] 形状不匹配（跳过）{len(skipped_mismatch)} 个:')
        for s in skipped_mismatch:
            print(f'           {s}')
    print(f'[pretrain] 无映射/不需要加载 {len(skipped_nomap)} 个（head/norm/radar 模块，正常）')
