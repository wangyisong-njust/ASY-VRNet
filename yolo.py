import colorsys
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.efficient_vrnet import EfficientVRNet
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config, preprocess_input_radar)
from utils.utils_bbox import decode_outputs, non_max_suppression
from utils.radar_utils import apply_radar_drop, load_radar_npz, radar_to_tensor

'''
训练自己的数据集必看注释！
'''


PROJECT_ROOT = Path(__file__).resolve().parent


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_shape(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    parts = [int(part) for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain two integers, got {value!r}")
    return parts


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": os.environ.get("ASY_MODEL_PATH", str(PROJECT_ROOT / "logs" / "best_epoch_weights.pth")),
        "radar_root": os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")),
        "classes_path": os.environ.get("ASY_CLASSES_PATH", str(PROJECT_ROOT / "model_data" / "waterscenes.txt")),
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": env_shape("ASY_INPUT_SHAPE", [320, 320]),
        "num_seg_classes": 9,
        "radar_in_channels": int(os.environ.get("ASY_RADAR_CHANNELS", "4")),
        "radar_align_mode": os.environ.get("ASY_RADAR_ALIGN_MODE", "letterbox"),
        "radar_normalize": env_bool("ASY_RADAR_NORMALIZE", False),
        "radar_preserve_points": env_bool("ASY_RADAR_PRESERVE_POINTS", True),
        "radar_source_order": os.environ.get("ASY_RADAR_SOURCE_ORDER", "range,doppler,elevation,power"),
        "radar_target_order": os.environ.get("ASY_RADAR_TARGET_ORDER", "range,elevation,velocity,power"),
        "radar_legacy_preprocess": env_bool("ASY_RADAR_LEGACY_PREPROCESS", False),
        "fusion_mode": os.environ.get("ASY_FUSION_MODE", "baseline"),
        "radar_dropout": float(os.environ.get("ASY_RADAR_DROPOUT", "0.0")),
        # Inference-time radar degradation for robustness curves (0 = full radar).
        "radar_drop_ratio": float(os.environ.get("ASY_RADAR_DROP_RATIO", "0.0")),
        "task_loss_mode": os.environ.get("ASY_TASK_LOSS", "uncertainty"),
        # ---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        # ---------------------------------------------------------------------#
        "phi": os.environ.get("ASY_PHI", "l"),
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": env_float("ASY_CONFIDENCE", 0.3),
        "max_boxes": int(os.environ.get("ASY_MAX_BOXES", "100")),
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": env_float("ASY_NMS_IOU", 0.5),
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": env_bool("ASY_LETTERBOX_IMAGE", True),
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": env_bool("ASY_CUDA", True),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        config = dict(self._defaults)
        config.update(kwargs)
        self.__dict__.update(config)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self._config = config
        self.radar_in_channels = int(self.radar_in_channels)
        self.radar_align_mode = str(self.radar_align_mode).lower()
        self.radar_normalize = bool(self.radar_normalize)
        self.radar_dropout = float(self.radar_dropout)
        self.radar_drop_ratio = float(getattr(self, "radar_drop_ratio", 0.0))
        self.fusion_mode = str(self.fusion_mode).lower()
        self.task_loss_mode = str(self.task_loss_mode).lower()

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._config)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = EfficientVRNet(
            num_classes=self.num_classes,
            num_seg_classes=self.num_seg_classes,
            phi=self.phi,
            radar_in_channels=self.radar_in_channels,
            fusion_mode=self.fusion_mode,
            radar_dropout=self.radar_dropout,
            task_loss_mode=self.task_loss_mode,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        model_dict = self.net.state_dict()
        compatible_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        skipped = sorted(set(checkpoint.keys()) - set(compatible_dict.keys()))
        model_dict.update(compatible_dict)
        self.net.load_state_dict(model_dict)
        if skipped:
            print(f"Skipped {len(skipped)} incompatible keys when loading {self.model_path}: {skipped[:5]}")
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def _device(self):
        return torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')

    def _prepare_radar(self, image_id, image, input_shape=None, flip=False):
        radar_data = load_radar_npz(
            self.radar_root,
            image_id,
            image.size,
            input_shape if input_shape is not None else self.input_shape,
            normalize=self.radar_normalize,
            align_mode=self.radar_align_mode,
            source_order=self.radar_source_order,
            target_order=self.radar_target_order,
            preserve_points=self.radar_preserve_points,
            legacy_preprocess=self.radar_legacy_preprocess,
        )
        if flip:
            radar_data = radar_data[:, :, ::-1].copy()
        radar = radar_to_tensor(radar_data, device=self._device()).float()
        if self.radar_drop_ratio > 0.0:
            radar = apply_radar_drop(radar, self.radar_drop_ratio)
        return radar

    # ===================================================================== #
    #   Innovation 3: Radar-aware Test-Time Refinement.
    #   Multi-scale + horizontal-flip augmentations are fused with a
    #   Weighted Box Fusion whose per-detection weight is modulated by the
    #   amount of radar evidence falling inside each box. Boxes backed by
    #   radar returns (real on-water targets) are up-weighted relative to
    #   camera-only detections (frequent background/clutter false positives),
    #   which improves cross-image ranking and therefore COCO mAP.
    # ===================================================================== #
    def _radar_point_mask(self, image_id):
        """Binary radar-occupancy map in the radar npz's native resolution."""
        radar_path = os.path.join(self.radar_root, image_id + ".npz")
        radar_raw = np.load(radar_path)["arr_0"]
        radar_raw = np.asarray(radar_raw, dtype=np.float32)
        return np.any(np.abs(radar_raw) > 0.0, axis=0)  # [Hr, Wr]

    def _box_radar_support(self, boxes_xyxy, point_mask, image_w, image_h, tau=3.0):
        """Saturating radar support in [0,1) for each box (x1,y1,x2,y2 in image px)."""
        if point_mask is None or point_mask.size == 0:
            return np.zeros(len(boxes_xyxy), dtype=np.float32)
        hr, wr = point_mask.shape
        sx = wr / max(float(image_w), 1.0)
        sy = hr / max(float(image_h), 1.0)
        support = np.zeros(len(boxes_xyxy), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            rx1 = int(np.clip(np.floor(x1 * sx), 0, wr - 1))
            rx2 = int(np.clip(np.ceil(x2 * sx), 1, wr))
            ry1 = int(np.clip(np.floor(y1 * sy), 0, hr - 1))
            ry2 = int(np.clip(np.ceil(y2 * sy), 1, hr))
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            count = float(point_mask[ry1:ry2, rx1:rx2].sum())
            support[i] = 1.0 - np.exp(-count / tau)
        return support

    def _infer_scale(self, image, image_id, image_shape, input_shape, flip=False):
        """Run one augmented forward pass; return per-image detections [N,6]:
        x1,y1,x2,y2 (image px), score, label."""
        image_data = resize_image(image, (input_shape[1], input_shape[0]), self.letterbox_image)
        arr = np.array(image_data, dtype='float32')
        if flip:
            arr = arr[:, ::-1, :].copy()
        image_data = np.expand_dims(np.transpose(preprocess_input(arr), (2, 0, 1)), 0)
        radar_data = self._prepare_radar(image_id, image, input_shape=input_shape, flip=flip)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs, _ = self.net(images, radar_data)
            outputs = decode_outputs(outputs, input_shape)
            if flip:
                # un-flip normalized centre-x so boxes land in the original frame
                outputs[..., 0] = 1.0 - outputs[..., 0]
            results = non_max_suppression(outputs, self.num_classes, input_shape,
                                          image_shape, self.letterbox_image,
                                          conf_thres=self.confidence, nms_thres=self.nms_iou)
        if results[0] is None:
            return np.zeros((0, 6), dtype=np.float32)
        res = results[0]
        # non_max_suppression returns boxes as [top, left, bottom, right] = y1,x1,y2,x2
        y1, x1, y2, x2 = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
        score = res[:, 4] * res[:, 5]
        label = res[:, 6]
        return np.stack([x1, y1, x2, y2, score, label], axis=1).astype(np.float32)

    @staticmethod
    def _radar_soft_nms(boxes, scores, labels, support, sigma=0.5, beta=0.0,
                        score_thr=1e-3, iou_hard=None):
        """Class-aware Gaussian Soft-NMS with radar-modulated decay.

        For each surviving box, overlapping lower-scored boxes have their score
        multiplied by exp(-iou^2 / sigma_eff). Boxes carrying radar evidence get a
        wider sigma_eff = sigma * (1 + beta * support), i.e. they are decayed less
        and are more likely to be kept. Coordinates are never modified.
        """
        keep_boxes, keep_scores, keep_labels = [], [], []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            b = boxes[idx].astype(np.float64).copy()
            s = scores[idx].astype(np.float64).copy()
            sup = support[idx].astype(np.float64).copy()
            area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
            while True:
                if s.size == 0 or s.max() < score_thr:
                    break
                m = int(np.argmax(s))
                keep_boxes.append(b[m].copy())
                keep_scores.append(float(s[m]))
                keep_labels.append(int(c))
                # IoU of the picked box against the rest
                xx1 = np.maximum(b[m, 0], b[:, 0]); yy1 = np.maximum(b[m, 1], b[:, 1])
                xx2 = np.minimum(b[m, 2], b[:, 2]); yy2 = np.minimum(b[m, 3], b[:, 3])
                w = np.maximum(xx2 - xx1, 0.0); h = np.maximum(yy2 - yy1, 0.0)
                inter = w * h
                iou = inter / (area[m] + area - inter + 1e-9)
                # radar-widened sigma protects radar-backed boxes from decay
                sigma_eff = sigma * (1.0 + beta * sup)
                decay = np.exp(-(iou ** 2) / np.maximum(sigma_eff, 1e-6))
                decay[m] = 0.0  # remove the picked box from the pool
                s = s * decay
                # drop fully-decayed entries to keep the loop finite
                alive = s >= score_thr
                b, s, sup, area = b[alive], s[alive], sup[alive], area[alive]
            # (picked boxes already recorded above)
        if not keep_boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
        return (np.asarray(keep_boxes, np.float32),
                np.asarray(keep_scores, np.float32),
                np.asarray(keep_labels, np.int32))

    def get_map_txt_tta(self, image_id, image, class_names, map_out_path,
                        scales=None, flip=True, radar_alpha=0.5,
                        wbf_iou=0.55, skip_thr=0.0, radar_tau=3.0, fusion="nms"):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])  # (H, W)
        image = cvtColor(image)
        iw, ih = image.size
        if scales is None:
            scales = [self.input_shape]

        point_mask = None
        try:
            point_mask = self._radar_point_mask(image_id)
        except Exception:
            point_mask = None

        # Collect detections from every augmentation in original-image pixel coords.
        all_boxes, all_scores, all_labels, all_support = [], [], [], []
        for s in scales:
            input_shape = [int(s[0]), int(s[1])] if isinstance(s, (list, tuple)) else [int(s), int(s)]
            views = [False, True] if flip else [False]
            for fl in views:
                det = self._infer_scale(image, image_id, image_shape, input_shape, flip=fl)
                if det.shape[0] == 0:
                    continue
                boxes_xyxy = det[:, :4].astype(np.float32)
                score = det[:, 4].astype(np.float32)
                label = det[:, 5].astype(np.int32)
                support = self._box_radar_support(boxes_xyxy, point_mask, iw, ih, tau=radar_tau)
                if fusion != "softnms":
                    # global radar boost (used by nms/wbf paths)
                    score = score * (1.0 + radar_alpha * support)
                all_boxes.append(boxes_xyxy)
                all_scores.append(score)
                all_labels.append(label)
                all_support.append(support.astype(np.float32))

        if not all_boxes:
            f.close()
            return

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_support = np.concatenate(all_support, axis=0)

        if fusion == "softnms":
            # Radar-aware Gaussian Soft-NMS: overlapping boxes are decayed rather
            # than removed, and boxes with radar evidence are decayed less (a wider
            # sigma), so genuine on-water targets survive crowded suppression while
            # camera-only duplicates fade. Box coordinates are never moved, so tight
            # localization (AP75) is preserved.
            fused_boxes, fused_scores, fused_labels = self._radar_soft_nms(
                all_boxes, all_scores, all_labels, all_support,
                sigma=0.5, beta=radar_alpha, score_thr=1e-3,
            )
            if len(fused_scores) == 0:
                f.close()
                return
        elif fusion == "wbf":
            from ensemble_boxes import weighted_boxes_fusion
            nb = all_boxes.copy()
            nb[:, [0, 2]] = np.clip(nb[:, [0, 2]] / max(iw, 1), 0.0, 1.0)
            nb[:, [1, 3]] = np.clip(nb[:, [1, 3]] / max(ih, 1), 0.0, 1.0)
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                [nb.tolist()], [all_scores.tolist()], [all_labels.tolist()],
                iou_thr=wbf_iou, skip_box_thr=skip_thr,
            )
            fused_boxes = np.asarray(fused_boxes, dtype=np.float32)
            if len(fused_boxes) == 0:
                f.close()
                return
            fused_boxes[:, [0, 2]] *= iw
            fused_boxes[:, [1, 3]] *= ih
            fused_scores = np.asarray(fused_scores, dtype=np.float32)
            fused_labels = np.asarray(fused_labels, dtype=np.int32)
        else:
            # Plain class-aware NMS over the pooled augmentations: keeps the
            # best-localized box and its original (radar-boosted) score, so the
            # multi-scale recall gain is kept without smearing localization.
            from torchvision.ops import boxes as box_ops
            tb = torch.from_numpy(all_boxes)
            ts = torch.from_numpy(all_scores)
            tl = torch.from_numpy(all_labels.astype(np.int64))
            keep = box_ops.batched_nms(tb, ts, tl, self.nms_iou).numpy()
            fused_boxes = all_boxes[keep]
            fused_scores = all_scores[keep]
            fused_labels = all_labels[keep]

        if len(fused_scores) > self.max_boxes:
            keep = np.argsort(fused_scores)[::-1][:self.max_boxes]
            fused_boxes = fused_boxes[keep]
            fused_scores = fused_scores[keep]
            fused_labels = fused_labels[keep]

        for box, score, c in zip(fused_boxes, fused_scores, fused_labels):
            predicted_class = self.class_names[int(c)]
            if predicted_class not in class_names:
                continue
            left, top, right, bottom = box
            left = max(0.0, min(float(left), float(iw)))
            top = max(0.0, min(float(top), float(ih)))
            right = max(0.0, min(float(right), float(iw)))
            bottom = max(0.0, min(float(bottom), float(ih)))
            if right <= left or bottom <= top:
                continue
            f.write(
                f"{predicted_class} {float(score):.8f} "
                f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f}\n"
            )
        f.close()
        return

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, image_id, crop=False, count=False):
        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_data = self._prepare_radar(image_id, image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, _ = self.net(images, radar_data)
            outputs = decode_outputs(outputs, self.input_shape)

            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top    = max(0, np.floor(top).astype('int32'))
            left   = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right  = min(image.size[0], np.floor(right).astype('int32'))
            # ensure valid box
            top, bottom = min(top, bottom), max(top, bottom)
            left, right = min(left, right), max(left, right)
            if bottom <= top or right <= left:
                continue

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            bbox = draw.textbbox((0, 0), label, font=font)
            label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                if left + i >= right - i or top + i >= bottom - i:
                    break
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, image_id, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_data = self._prepare_radar(image_id, image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, _ = self.net(images, radar_data)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs, _ = self.net(images, radar_data)
                outputs = decode_outputs(outputs, self.input_shape)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, image_id, heatmap_save_path):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_data = self._prepare_radar(image_id, image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, _ = self.net(images, radar_data)

        outputs = [output.cpu().numpy() for output in outputs]
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200)
        print("Save to the " + heatmap_save_path)
        plt.cla()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')
        radar = torch.zeros(1, self.radar_in_channels, *self.input_shape).to('cpu')
        input_layer_names = ["images", "radars"]
        output_layer_names = ["det_s", "det_m", "det_l", "segmentation"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          (im, radar),
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_data = self._prepare_radar(image_id, image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, _ = self.net(images, radar_data)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                f.close()
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
            if len(top_conf) > self.max_boxes:
                keep = np.argsort(top_conf)[::-1][:self.max_boxes]
                top_label = top_label[keep]
                top_conf = top_conf[keep]
                top_boxes = top_boxes[keep]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = float(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            top = max(0.0, min(float(top), float(image.size[1])))
            left = max(0.0, min(float(left), float(image.size[0])))
            bottom = max(0.0, min(float(bottom), float(image.size[1])))
            right = max(0.0, min(float(right), float(image.size[0])))
            if right <= left or bottom <= top:
                continue

            f.write(
                f"{predicted_class} {score:.8f} "
                f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f}\n"
            )

        f.close()
        return
