import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr

from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)

from utils_seg.utils import get_lr
from utils_seg.utils_metrics import f_score

from utils.multitaskloss import MultiTaskLossWrapper


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


# ---- Innovation 3: modality-dropout consistency regularization ----
def consistency_enabled():
    return _env_flag("ASY_CONSISTENCY", False)


def make_radar_dropout(radars):
    """Return a radar tensor with the radar modality (partially) dropped.

    ``ASY_CONSISTENCY_RADAR_DROP`` in [0, 1] is the fraction of radar to remove:
    1.0 fully zeros the radar input (the model must rely on vision only),
    while a value < 1.0 randomly masks that fraction of spatial locations.
    """
    drop = _env_float("ASY_CONSISTENCY_RADAR_DROP", 1.0)
    if drop >= 1.0:
        return torch.zeros_like(radars)
    keep_mask = (torch.rand_like(radars[:, :1, :, :]) >= drop).type_as(radars)
    return radars * keep_mask


def compute_consistency_loss(model_train, images, radars, outputs_full, outputs_seg_full):
    """Encourage graceful degradation under radar failure.

    A second forward is run with the radar modality dropped; its detection and
    segmentation predictions are pulled toward the full-input predictions
    (used as a detached teacher). This makes the model robust when radar is
    unreliable or missing, without changing inference behaviour.
    """
    radars_dropped = make_radar_dropout(radars)
    outputs_dropped, outputs_seg_dropped = model_train(images, radars_dropped)

    # Compute MSE in fp32 to prevent fp16 overflow on large early-training logits.
    # Apply sigmoid to bound detection head outputs before MSE; otherwise raw
    # logits of ±1e3 produce MSE ~1e6 which overflows fp16 and propagates inf.
    det_cons = outputs_full[0].new_zeros(()).float()
    for o_full, o_drop in zip(outputs_full, outputs_dropped):
        det_cons = det_cons + F.mse_loss(
            torch.sigmoid(o_drop).float(),
            torch.sigmoid(o_full).detach().float(),
        )
    if len(outputs_full) > 0:
        det_cons = det_cons / len(outputs_full)

    seg_cons = F.mse_loss(
        outputs_seg_dropped.float(),
        outputs_seg_full.detach().float(),
    )

    w_det = _env_float("ASY_CONSISTENCY_DET_WEIGHT", 1.0)
    w_seg = _env_float("ASY_CONSISTENCY_SEG_WEIGHT", 0.5)
    return w_det * det_cons + w_seg * seg_cons


def maybe_add_consistency(total_loss, model_train, images, radars, outputs, outputs_seg):
    """Add the consistency term to ``total_loss`` with probability ``ASY_CONSISTENCY_PROB``.

    The sampling decision is broadcast from rank-0 so all DDP ranks always
    perform the same number of forward passes — mismatched forward counts cause
    a permanent NCCL allreduce deadlock.
    """
    if not consistency_enabled():
        return total_loss, 0.0
    prob = _env_float("ASY_CONSISTENCY_PROB", 0.5)
    # Synchronise the do/skip decision across all DDP ranks.
    flag = torch.zeros(1, device=images.device)
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        flag[0] = 1.0 if torch.rand(1).item() < prob else 0.0
    if dist.is_initialized():
        dist.broadcast(flag, src=0)
    if flag.item() < 0.5:
        return total_loss, 0.0
    weight = _env_float("ASY_CONSISTENCY_WEIGHT", 1.0)
    cons = compute_consistency_loss(model_train, images, radars, outputs, outputs_seg)
    return total_loss + weight * cons, float(cons.detach())


def get_loss_balancer(model_train):
    module = model_train.module if hasattr(model_train, "module") else model_train
    return getattr(module, "loss_balancer", None)


def combine_task_losses(loss_det, loss_seg, model_train):
    loss_balancer = get_loss_balancer(model_train)
    if loss_balancer is None:
        return loss_det + loss_seg
    return loss_balancer(loss_seg, loss_det)


def compute_detection_loss(yolo_loss, outputs, targets, model_train, radars=None):
    loss_balancer = get_loss_balancer(model_train)
    if loss_balancer is not None and getattr(loss_balancer, "task_num", None) == 4:
        loss_det, det_components = yolo_loss(outputs, targets, radars=radars, return_components=True)
        return loss_det, det_components
    return yolo_loss(outputs, targets, radars=radars), None


def combine_detection_segmentation_losses(loss_det, loss_seg, det_components, model_train):
    loss_balancer = get_loss_balancer(model_train)
    if loss_balancer is None:
        return loss_det + loss_seg
    if getattr(loss_balancer, "task_num", None) == 4:
        if det_components is None:
            raise ValueError("4-task uncertainty loss requires detection loss components.")
        loss_iou, loss_obj, loss_cls = det_components
        # Paper Eq. 10 weights bbox regression, object confidence,
        # detection classification and pixel classification separately.
        return loss_balancer(loss_iou, loss_obj, loss_cls, loss_seg)
    return loss_balancer(loss_det, loss_seg)


def best_checkpoint_score(mode, current_val_det, current_val_seg, det_history, seg_history):
    mode = mode.lower()
    if mode == "det":
        return current_val_det, list(det_history), "detection val loss"
    if mode == "seg":
        return current_val_seg, list(seg_history), "segmentation val loss"
    if mode == "total":
        previous = [
            det_loss + seg_loss
            for det_loss, seg_loss in zip(det_history, seg_history)
        ]
        return current_val_det + current_val_seg, previous, "total val loss"
    raise ValueError(f"Unsupported ASY_BEST_METRIC={mode!r}; use det, seg, or total.")


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, loss_history_seg, eval_callback, eval_callback_seg, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, dice_loss, focal_loss, cls_weights, num_class_seg, local_rank=0):
    total_loss_det = 0
    total_loss_seg = 0
    total_f_score = 0

    val_loss_det = 0
    val_loss_seg = 0
    val_f_score = 0

    total_loss_value = 0
    val_total_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets, radars, pngs, seg_labels = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, outputs_seg = model_train(images, radars)

            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg = loss_seg + main_dice

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_det, det_components = compute_detection_loss(yolo_loss, outputs, targets, model_train, radars=radars)

            total_loss = combine_detection_segmentation_losses(loss_det, loss_seg, det_components, model_train)

            total_loss, _ = maybe_add_consistency(total_loss, model_train, images, radars, outputs, outputs_seg)

            with torch.no_grad():
                train_f_score = f_score(outputs_seg, seg_labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            total_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs, outputs_seg = model_train(images, radars)

                if focal_loss:
                    loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                else:
                    loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)

                if dice_loss:
                    main_dice = Dice_loss(outputs_seg, seg_labels)
                    loss_seg = loss_seg + main_dice

                # ----------------------#
                #   calculate loss
                # ----------------------#
                loss_det, det_components = compute_detection_loss(yolo_loss, outputs, targets, model_train, radars=radars)

                total_loss = combine_detection_segmentation_losses(loss_det, loss_seg, det_components, model_train)

                total_loss, _ = maybe_add_consistency(total_loss, model_train, images, radars, outputs, outputs_seg)

                with torch.no_grad():
                    train_f_score = f_score(outputs_seg, seg_labels)

            # ----------------------#
            #   back-propagation
            # ----------------------#
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        total_loss_det += loss_det.item()
        total_loss_seg += loss_seg.item()
        total_loss_value += total_loss.item()
        total_f_score += train_f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'detection loss': total_loss_det / (iteration + 1),
                                'segmentation loss': total_loss_seg / (iteration + 1),
                                'total loss': total_loss_value / (iteration + 1),
                                'f score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    val_weights = torch.from_numpy(cls_weights)
    if cuda:
        val_weights = val_weights.cuda(local_rank)

    if ema:
        model_train_eval = ema.ema.eval()
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, radars, pngs, seg_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, outputs_seg = model_train_eval(images, radars)

            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, val_weights, num_classes=num_class_seg)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, val_weights, num_classes=num_class_seg)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg = loss_seg + main_dice

            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs_seg, seg_labels)

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value = yolo_loss(outputs, targets)
            loss_value_seg = loss_seg
            val_f_score += _f_score.item()

        val_loss_det += loss_value.item()
        val_loss_seg += loss_value_seg.item()
        val_total_loss = val_loss_det + val_loss_seg

        if local_rank == 0:
            pbar.set_postfix(**{'detection val_loss': val_loss_det / (iteration + 1),
                                'segmentation val_loss': val_loss_seg / (iteration + 1),
                                'val loss': val_total_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        current_val_det = val_loss_det / epoch_step_val
        current_val_seg = val_loss_seg / epoch_step_val
        current_val_total = current_val_det + current_val_seg
        best_metric = os.environ.get("ASY_BEST_METRIC", "det")
        current_best_score, previous_best_scores, best_metric_name = best_checkpoint_score(
            best_metric,
            current_val_det,
            current_val_seg,
            loss_history.val_loss,
            loss_history_seg.val_loss,
        )
        loss_history.append_loss(epoch + 1, total_loss_det / epoch_step, val_loss_det / epoch_step_val)
        loss_history_seg.append_loss(epoch + 1, total_loss_seg / epoch_step, val_loss_seg / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        eval_callback_seg.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss Det: %.3f  || Val Loss Seg: %.3f' % ((total_loss_value / epoch_step,
                                                                                  val_loss_det / epoch_step_val,
                                                                                 val_loss_seg / epoch_step_val)))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-det_val_loss%.3f-seg_val_loss%.3f.pth" % (
            epoch + 1, current_val_total, current_val_det, current_val_seg)))

        if not previous_best_scores or current_best_score <= min(previous_best_scores):
            print(f'Save best model to best_epoch_weights.pth by {best_metric_name}: {current_best_score:.6f}')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
