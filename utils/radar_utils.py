import os

import cv2
import numpy as np
import torch


def normalize_radar(data, eps=1e-6):
    data = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data, dtype=np.float32)

    data_min = np.min(data[valid])
    data_max = np.max(data[valid])
    denom = data_max - data_min
    if denom < eps:
        return np.zeros_like(data, dtype=np.float32)

    data = (data - data_min) / denom
    data[~valid] = 0
    return data.astype(np.float32)


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


def load_radar_npz(radar_root, image_id, image_size, input_shape, normalize=True):
    radar_path = os.path.join(radar_root, image_id + ".npz")
    radar_data = np.load(radar_path)["arr_0"]
    radar_data = letterbox_radar_map(radar_data, image_size, input_shape)
    if normalize:
        radar_data = normalize_radar(radar_data)
    return radar_data


def radar_to_tensor(radar_data, device=None):
    tensor = torch.from_numpy(np.asarray(radar_data, dtype=np.float32)).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
