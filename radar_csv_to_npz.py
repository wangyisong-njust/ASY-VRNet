"""
WaterScenes 雷达点云 CSV → NPZ 转换脚本

CSV 列格式（来自 WaterScenes 数据集）:
  timestamp, range, doppler, azimuth, elevation, power,
  x, y, z, comp_height, comp_velocity, u, v, label, instance

输出 NPZ:
  arr_0.shape = (4, img_h, img_w)
  通道 0 = range
  通道 1 = doppler
  通道 2 = elevation
  通道 3 = power

说明：
  仓库内已有 VOCradar*.npz 使用 range,doppler,elevation,power 的存储顺序。
  训练/评测读取时会通过 utils/radar_utils.py 重排成论文使用的
  range,elevation,velocity,power(REVP) 顺序，避免旧数据和新转换数据不一致。
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# WaterScenes 原始图像分辨率（Sony IMX-317）
SRC_W = 1920
SRC_H = 1080


def csv_to_feature_map(csv_path: str, out_h: int, out_w: int) -> np.ndarray:
    """
    将单帧雷达 CSV 转换为 (4, out_h, out_w) 的 float32 特征图。
    同一像素有多个点时，保留 power 最大的那个点的所有特征。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] 读取失败 {csv_path}: {e}")
        return np.zeros((4, out_h, out_w), dtype=np.float32)

    required = {'u', 'v', 'range', 'doppler', 'elevation', 'power'}
    if not required.issubset(df.columns):
        print(f"[WARN] 缺少必要列 {csv_path}, 当前列: {list(df.columns)}")
        return np.zeros((4, out_h, out_w), dtype=np.float32)

    # 缩放 u, v 到目标分辨率
    scale_u = out_w / SRC_W
    scale_v = out_h / SRC_H
    df['u_scaled'] = (df['u'] * scale_u).astype(int)
    df['v_scaled'] = (df['v'] * scale_v).astype(int)

    # 只保留在图像范围内的点
    mask = (df['u_scaled'] >= 0) & (df['u_scaled'] < out_w) & \
           (df['v_scaled'] >= 0) & (df['v_scaled'] < out_h)
    df = df[mask].copy()

    feature_map = np.zeros((4, out_h, out_w), dtype=np.float32)

    if df.empty:
        return feature_map

    # 同一像素多个点时，保留 power 最大的
    df = df.sort_values('power', ascending=True)  # 升序，后写的覆盖前面的
    us = df['u_scaled'].values
    vs = df['v_scaled'].values
    feature_map[0, vs, us] = df['range'].values.astype(np.float32)
    feature_map[1, vs, us] = df['doppler'].values.astype(np.float32)
    feature_map[2, vs, us] = df['elevation'].values.astype(np.float32)
    feature_map[3, vs, us] = df['power'].values.astype(np.float32)

    return feature_map


def convert_all(radar_csv_dir: str, output_dir: str, out_h: int = 512, out_w: int = 512):
    """
    批量转换一个目录下所有 CSV 文件为 NPZ。
    输出文件名与输入相同，只替换扩展名为 .npz。
    """
    csv_dir = Path(radar_csv_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] 在 {radar_csv_dir} 下未找到 CSV 文件")
        return

    print(f"找到 {len(csv_files)} 个 CSV 文件，输出到 {output_dir}")
    success, skip = 0, 0

    for csv_path in tqdm(csv_files, desc="转换中"):
        out_path = out_dir / (csv_path.stem + ".npz")
        if out_path.exists():
            skip += 1
            continue
        feature_map = csv_to_feature_map(str(csv_path), out_h, out_w)
        np.savez_compressed(str(out_path), feature_map)
        success += 1

    print(f"完成：成功 {success} 个，跳过（已存在）{skip} 个")


def verify(npz_path: str):
    """验证一个 npz 文件的内容"""
    data = np.load(npz_path)['arr_0']
    print(f"shape : {data.shape}")
    print(f"dtype : {data.dtype}")
    print(f"range   ch0: min={data[0].min():.3f}, max={data[0].max():.3f}, nonzero={np.count_nonzero(data[0])}")
    print(f"doppler ch1: min={data[1].min():.3f}, max={data[1].max():.3f}, nonzero={np.count_nonzero(data[1])}")
    print(f"elev    ch2: min={data[2].min():.3f}, max={data[2].max():.3f}, nonzero={np.count_nonzero(data[2])}")
    print(f"power   ch3: min={data[3].min():.3f}, max={data[3].max():.3f}, nonzero={np.count_nonzero(data[3])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WaterScenes 雷达 CSV → NPZ 转换")
    parser.add_argument("--csv_dir",  required=True, help="雷达 CSV 文件夹路径")
    parser.add_argument("--out_dir",  required=True, help="输出 NPZ 文件夹路径")
    parser.add_argument("--height",   type=int, default=512, help="特征图高度（默认 512）")
    parser.add_argument("--width",    type=int, default=512, help="特征图宽度（默认 512）")
    parser.add_argument("--verify",   type=str, default="", help="验证单个 npz 文件（传入路径）")
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
    else:
        convert_all(args.csv_dir, args.out_dir, args.height, args.width)
