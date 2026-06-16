#!/usr/bin/env python3
"""Greedy model soup (Wortsman et al., ICML 2022).

Ranks candidate checkpoints by their individual validation mAP, then greedily
adds each (in descending rank order) to a uniformly-averaged soup, keeping a
candidate only if it does NOT decrease the soup's validation mAP. Selection
uses a fast val SUBSET; the final chosen soup should then be evaluated once on
the full val set separately.

Usage:
  python3 scripts/greedy_soup.py \
      --out logs_innov2_soup/greedy_soup.pth \
      --val_txt 2007_val_subset400.txt \
      --python ~/anaconda3/envs/PDPP/bin/python \
      --gpu 0 \
      --candidates \
        logs_multiscale_ft_full_phi_l_5frames_bs48_e50_320/best_epoch_weights.pth \
        logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth \
        logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320/ep160-*.pth \
        ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run_eval(python, gpu, ckpt, val_txt, out_dir, extra_args):
    """Eval one checkpoint on the (subset) val set; return mAP50-95."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        python, os.path.join(ROOT, "eval_paper_metrics.py"),
        "--model_path", ckpt,
        "--val_txt", val_txt,
        "--out_dir", out_dir,
        "--fusion_mode", "baseline", "--phi", "l",
        "--input_shape", "320", "320",
        "--confidence", "0.001", "--max_boxes", "100",
        "--radar_root", os.path.join(ROOT, "dataset/VOCradar_5_frames"),
        "--vocdevkit_path", os.path.join(ROOT, "dataset/VOCdevkit"),
        "--radar_legacy_preprocess", "--no_radar_preserve_points",
        "--radar_source_order", "range,doppler,elevation,power",
        "--radar_target_order", "range,doppler,elevation,power",
        "--task_loss", "sum",
    ] + extra_args
    subprocess.run(cmd, env=env, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(ROOT, out_dir, "paper_metrics.json")) as f:
        return json.load(f)["mAP50-95"]


def make_soup(python, members, out_path):
    cmd = [python, os.path.join(HERE, "make_checkpoint_soup.py"), "--out", out_path] + members
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_txt", default="2007_val_subset400.txt")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--extra_args", nargs="*", default=[])
    args = ap.parse_args()

    cands = [c for c in args.candidates if os.path.isfile(os.path.join(ROOT, c) if not os.path.isabs(c) else c)]
    missing = set(args.candidates) - set(cands)
    if missing:
        print(f"[WARN] skipping missing checkpoints: {missing}")
    if len(cands) < 2:
        raise SystemExit("Need >=2 existing candidates.")

    tmp = tempfile.mkdtemp(prefix="greedy_soup_")
    print(f"[greedy] tmp dir: {tmp}")
    print(f"[greedy] ranking {len(cands)} candidates on {args.val_txt} ...")

    # 1) rank candidates by individual mAP
    scored = []
    for i, c in enumerate(cands):
        m = run_eval(args.python, args.gpu, c, args.val_txt, f"{tmp}/rank_{i}", args.extra_args)
        scored.append((m, c))
        print(f"  [{m:.3f}] {c}")
    scored.sort(reverse=True)

    # 2) greedy add
    members = [scored[0][1]]
    best_m = scored[0][0]
    print(f"\n[greedy] start: [{best_m:.3f}] {os.path.basename(members[0])}")
    for m_indiv, c in scored[1:]:
        trial = f"{tmp}/trial_soup.pth"
        make_soup(args.python, members + [c], trial)
        m = run_eval(args.python, args.gpu, trial, args.val_txt, f"{tmp}/trial_eval", args.extra_args)
        if m >= best_m - 1e-9:
            members.append(c)
            print(f"  + KEEP [{m:.3f} >= {best_m:.3f}] {os.path.basename(c)}")
            best_m = m
        else:
            print(f"  - drop [{m:.3f} <  {best_m:.3f}] {os.path.basename(c)}")

    # 3) save final soup (if single member, just copy)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    if len(members) == 1:
        import shutil
        shutil.copy(os.path.join(ROOT, members[0]) if not os.path.isabs(members[0]) else members[0],
                    os.path.join(ROOT, args.out))
    else:
        make_soup(args.python, members, os.path.join(ROOT, args.out))

    print(f"\n[greedy] FINAL soup ({len(members)} members, subset mAP={best_m:.3f}):")
    for c in members:
        print(f"  - {c}")
    print(f"[greedy] saved -> {args.out}")
    print(f"[greedy] NOTE: now run full-val eval on this soup to get the paper number.")


if __name__ == "__main__":
    main()
