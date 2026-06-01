import os

import cv2
import numpy as np
import torch


DEFAULT_SOURCE_ORDER = "range,doppler,elevation,power"
DEFAULT_TARGET_ORDER = "range,elevation,velocity,power"
ORDER_ALIASES = {
    "vel": "velocity",
    "doppler": "velocity",
    "radial_velocity": "velocity",
    "comp_velocity": "velocity",
}


def _parse_order(order):
    if order is None:
        return []
    return [ORDER_ALIASES.get(item.strip().lower(), item.strip().lower()) for item in str(order).split(",") if item.strip()]


def reorder_radar_channels(radar_data, source_order=None, target_order=None):
    radar_data = np.asarray(radar_data, dtype=np.float32)
    if radar_data.ndim != 3:
        raise ValueError(f"Radar map must have shape [C,H,W], got {radar_data.shape}")

    source = _parse_order(source_order or os.environ.get("ASY_RADAR_SOURCE_ORDER", DEFAULT_SOURCE_ORDER))
    target = _parse_order(target_order or os.environ.get("ASY_RADAR_TARGET_ORDER", DEFAULT_TARGET_ORDER))
    if not source or not target or source == target:
        return radar_data
    if len(source) != radar_data.shape[0]:
        raise ValueError(
            f"Radar source order {source} has {len(source)} channels, but data has {radar_data.shape[0]} channels"
        )

    source_to_idx = {}
    for idx, name in enumerate(source):
        source_to_idx.setdefault(name, idx)

    indices = []
    for name in target:
        if name not in source_to_idx:
            raise ValueError(f"Cannot map radar channel {name!r}; source order is {source}")
        indices.append(source_to_idx[name])
    return radar_data[indices]


def _radar_point_mask(radar_data, eps=0.0):
    return np.any(np.abs(radar_data) > eps, axis=0)


def _splat_sparse_radar(radar_data, target_h, target_w, scale_x, scale_y, dx=0, dy=0):
    radar_data = np.asarray(radar_data, dtype=np.float32)
    out = np.zeros((radar_data.shape[0], target_h, target_w), dtype=np.float32)
    ys, xs = np.where(_radar_point_mask(radar_data))
    if len(xs) == 0:
        return out

    tx = np.floor(xs.astype(np.float32) * scale_x + dx).astype(np.int32)
    ty = np.floor(ys.astype(np.float32) * scale_y + dy).astype(np.int32)
    keep = (tx >= 0) & (tx < target_w) & (ty >= 0) & (ty < target_h)
    if not np.any(keep):
        return out

    xs = xs[keep]
    ys = ys[keep]
    tx = tx[keep]
    ty = ty[keep]

    # Channel 3 is power after REVP channel reordering. Larger power wins when
    # several sparse radar points land on the same target pixel.
    power = radar_data[3, ys, xs] if radar_data.shape[0] > 3 else np.ones_like(xs, dtype=np.float32)
    order = np.argsort(power)
    for idx in order:
        out[:, ty[idx], tx[idx]] = radar_data[:, ys[idx], xs[idx]]
    return out


def normalize_radar(data, eps=1e-6):
    data = np.asarray(data, dtype=np.float32)
    normalized = np.zeros_like(data, dtype=np.float32)

    for channel_idx in range(data.shape[0]):
        channel = data[channel_idx]
        valid = np.isfinite(channel) & (channel != 0)
        if not valid.any():
            continue

        values = channel[valid]
        data_min = np.min(values)
        data_max = np.max(values)
        denom = data_max - data_min
        if denom < eps:
            normalized[channel_idx, valid] = 1.0
        else:
            normalized[channel_idx, valid] = (values - data_min) / denom

    return normalized.astype(np.float32)


def resize_radar_map(radar_data, input_shape, preserve_points=True):
    """Resize a sparse REVP radar map to the network input size."""
    radar_data = np.asarray(radar_data, dtype=np.float32)
    if radar_data.ndim != 3:
        raise ValueError(f"Radar map must have shape [C,H,W], got {radar_data.shape}")

    h, w = int(input_shape[0]), int(input_shape[1])
    if preserve_points:
        src_h, src_w = radar_data.shape[-2:]
        return _splat_sparse_radar(radar_data, h, w, w / src_w, h / src_h)

    resized = np.empty((radar_data.shape[0], h, w), dtype=np.float32)
    for c in range(radar_data.shape[0]):
        resized[c] = cv2.resize(radar_data[c], (w, h), interpolation=cv2.INTER_NEAREST)
    return resized


def letterbox_radar_map(radar_data, image_size, input_shape, fill_value=0.0, preserve_points=True):
    """Align a raw radar feature map with the image letterbox transform."""
    radar_data = np.asarray(radar_data, dtype=np.float32)
    if radar_data.ndim != 3:
        raise ValueError(f"Radar map must have shape [C,H,W], got {radar_data.shape}")

    iw, ih = image_size
    h, w = int(input_shape[0]), int(input_shape[1])
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    if preserve_points and fill_value == 0.0:
        src_h, src_w = radar_data.shape[-2:]
        return _splat_sparse_radar(radar_data, h, w, nw / src_w, nh / src_h, dx=dx, dy=dy)

    aligned = np.full((radar_data.shape[0], h, w), fill_value, dtype=np.float32)
    for c in range(radar_data.shape[0]):
        resized = cv2.resize(radar_data[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
        aligned[c, dy:dy + nh, dx:dx + nw] = resized
    return aligned


def align_radar_map(radar_data, image_size, input_shape, align_mode="letterbox", preserve_points=None):
    align_mode = str(align_mode).lower()
    if preserve_points is None:
        preserve_points = os.environ.get("ASY_RADAR_PRESERVE_POINTS", "1").lower() not in {"0", "false", "no", "off"}
    if align_mode in {"resize", "direct"}:
        return resize_radar_map(radar_data, input_shape, preserve_points=preserve_points)
    if align_mode == "letterbox":
        return letterbox_radar_map(radar_data, image_size, input_shape, preserve_points=preserve_points)
    if align_mode == "none":
        radar_data = np.asarray(radar_data, dtype=np.float32)
        expected_shape = (int(input_shape[0]), int(input_shape[1]))
        if radar_data.shape[-2:] != expected_shape:
            raise ValueError(
                f"Radar map shape {radar_data.shape[-2:]} does not match input shape {expected_shape}"
            )
        return radar_data
    raise ValueError(f"Unsupported radar align mode: {align_mode!r}")


def load_radar_npz(radar_root, image_id, image_size, input_shape, normalize=False, align_mode="letterbox",
                   source_order=None, target_order=None, preserve_points=None):
    radar_path = os.path.join(radar_root, image_id + ".npz")
    radar_data = np.load(radar_path)["arr_0"]
    radar_data = reorder_radar_channels(radar_data, source_order=source_order, target_order=target_order)
    radar_data = align_radar_map(radar_data, image_size, input_shape, align_mode=align_mode,
                                 preserve_points=preserve_points)
    if normalize:
        radar_data = normalize_radar(radar_data)
    return radar_data


def radar_to_tensor(radar_data, device=None):
    tensor = torch.from_numpy(np.asarray(radar_data, dtype=np.float32)).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
