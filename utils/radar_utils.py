import os

import cv2
import numpy as np
import torch


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


def resize_radar_map(radar_data, input_shape):
    """Resize a square REVP radar map to the network input size."""
    radar_data = np.asarray(radar_data, dtype=np.float32)
    if radar_data.ndim != 3:
        raise ValueError(f"Radar map must have shape [C,H,W], got {radar_data.shape}")

    h, w = int(input_shape[0]), int(input_shape[1])
    resized = np.empty((radar_data.shape[0], h, w), dtype=np.float32)
    for c in range(radar_data.shape[0]):
        resized[c] = cv2.resize(radar_data[c], (w, h), interpolation=cv2.INTER_NEAREST)
    return resized


def letterbox_radar_map(radar_data, image_size, input_shape, fill_value=0.0):
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

    aligned = np.full((radar_data.shape[0], h, w), fill_value, dtype=np.float32)
    for c in range(radar_data.shape[0]):
        resized = cv2.resize(radar_data[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
        aligned[c, dy:dy + nh, dx:dx + nw] = resized
    return aligned


def align_radar_map(radar_data, image_size, input_shape, align_mode="letterbox"):
    align_mode = str(align_mode).lower()
    if align_mode in {"resize", "direct"}:
        return resize_radar_map(radar_data, input_shape)
    if align_mode == "letterbox":
        return letterbox_radar_map(radar_data, image_size, input_shape)
    if align_mode == "none":
        radar_data = np.asarray(radar_data, dtype=np.float32)
        expected_shape = (int(input_shape[0]), int(input_shape[1]))
        if radar_data.shape[-2:] != expected_shape:
            raise ValueError(
                f"Radar map shape {radar_data.shape[-2:]} does not match input shape {expected_shape}"
            )
        return radar_data
    raise ValueError(f"Unsupported radar align mode: {align_mode!r}")


def load_radar_npz(radar_root, image_id, image_size, input_shape, normalize=False, align_mode="letterbox"):
    radar_path = os.path.join(radar_root, image_id + ".npz")
    radar_data = np.load(radar_path)["arr_0"]
    radar_data = align_radar_map(radar_data, image_size, input_shape, align_mode=align_mode)
    if normalize:
        radar_data = normalize_radar(radar_data)
    return radar_data


def radar_to_tensor(radar_data, device=None):
    tensor = torch.from_numpy(np.asarray(radar_data, dtype=np.float32)).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
