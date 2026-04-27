#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import pickle
import argparse
from pathlib import Path

import numpy as np


TARGET_MODELS = [
    "CasA-V",
    "SCAFNet",
    "CasA_V_fgsp_V1",
    "SCAF-FGSPNet",
]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_preds_in_sample(sample):
    """
    sample: dict with keys like name, score, bbox, boxes_lidar, frame_id
    """
    if not isinstance(sample, dict):
        return 0

    if "score" in sample:
        arr = sample["score"]
        if hasattr(arr, "shape"):
            if len(arr.shape) == 0:
                return int(arr.size)
            return int(arr.shape[0])
    if "name" in sample:
        arr = sample["name"]
        if hasattr(arr, "shape"):
            if len(arr.shape) == 0:
                return int(arr.size)
            return int(arr.shape[0])
    return 0


def summarize_result_pkl(pkl_path):
    data = load_pkl(pkl_path)

    num_samples = len(data)
    pred_counts = []
    empty_count = 0

    for sample in data:
        n = count_preds_in_sample(sample)
        pred_counts.append(n)
        if n == 0:
            empty_count += 1

    pred_counts = np.array(pred_counts, dtype=np.int32)

    summary = {
        "num_samples": int(num_samples),
        "empty_samples": int(empty_count),
        "non_empty_samples": int(num_samples - empty_count),
        "empty_ratio": float(empty_count / max(num_samples, 1)),
        "total_predictions": int(pred_counts.sum()),
        "mean_preds_per_sample": float(pred_counts.mean()),
        "max_preds_per_sample": int(pred_counts.max()) if len(pred_counts) > 0 else 0,
        "min_preds_per_sample": int(pred_counts.min()) if len(pred_counts) > 0 else 0,
        "median_preds_per_sample": float(np.median(pred_counts)) if len(pred_counts) > 0 else 0.0,
    }
    return summary


def find_epoch_result_pkls(model_dir):
    """
    找形如:
    model/default/eval/eval_with_train/epoch_110/val/result.pkl
    """
    results = []
    pattern = re.compile(r"epoch_(\d+)")
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f != "result.pkl":
                continue
            full_path = os.path.join(root, f)
            m = pattern.search(full_path)
            if m:
                epoch = int(m.group(1))
                results.append((epoch, full_path))
    results.sort(key=lambda x: x[0])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=TARGET_MODELS
    )
    parser.add_argument(
        "--save",
        type=str,
        default="epoch_result_summary.json"
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    save_path = os.path.abspath(args.save)

    all_results = {}

    for model in args.models:
        model_dir = os.path.join(root, model)
        if not os.path.exists(model_dir):
            print(f"[WARN] model dir not found: {model_dir}")
            continue

        print("=" * 100)
        print(f"[MODEL] {model}")
        print("=" * 100)

        epoch_pkls = find_epoch_result_pkls(model_dir)
        model_stats = []

        for epoch, pkl_path in epoch_pkls:
            stat = summarize_result_pkl(pkl_path)
            stat["epoch"] = epoch
            stat["result_pkl"] = pkl_path
            model_stats.append(stat)

            print(
                f"epoch={epoch:>3d} | "
                f"empty={stat['empty_samples']:>4d}/{stat['num_samples']} "
                f"({stat['empty_ratio']:.2%}) | "
                f"total_preds={stat['total_predictions']:>5d} | "
                f"mean_preds={stat['mean_preds_per_sample']:.3f}"
            )

        # 一个简单排序建议：
        # 先按 empty_ratio 升序，再按 total_predictions 降序
        ranked = sorted(
            model_stats,
            key=lambda x: (x["empty_ratio"], -x["total_predictions"], -x["mean_preds_per_sample"])
        )

        if ranked:
            best = ranked[0]
            print("-" * 100)
            print(
                f"[SUGGEST] {model}: epoch {best['epoch']} "
                f"(empty_ratio={best['empty_ratio']:.2%}, total_preds={best['total_predictions']})"
            )

        all_results[model] = {
            "epochs": model_stats,
            "suggested_best": ranked[0] if ranked else None
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"Saved to: {save_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
