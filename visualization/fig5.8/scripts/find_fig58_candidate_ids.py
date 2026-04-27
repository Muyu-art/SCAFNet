#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MODEL_CONFIG = {
    "CasA-V": 72,
    "SCAFNet": 119,
    "CasA_V_fgsp_V1": 72,
    "SCAF-FGSPNet": 113,
}


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def normalize_frame_id(frame_id):
    """
    统一成 6 位字符串，例如 3 -> '000003'
    """
    if isinstance(frame_id, np.ndarray):
        if frame_id.size == 0:
            return None
        frame_id = frame_id.reshape(-1)[0]

    s = ensure_str(frame_id).strip()
    if s == "":
        return None

    # 纯数字时补零
    if s.isdigit():
        return f"{int(s):06d}"
    return s


def sample_num_preds(sample):
    if "score" in sample:
        arr = sample["score"]
        if hasattr(arr, "shape"):
            return int(arr.shape[0]) if len(arr.shape) > 0 else int(arr.size)
    if "name" in sample:
        arr = sample["name"]
        if hasattr(arr, "shape"):
            return int(arr.shape[0]) if len(arr.shape) > 0 else int(arr.size)
    return 0


def sample_classes(sample):
    names = sample.get("name", None)
    if names is None:
        return []
    if isinstance(names, np.ndarray):
        return [ensure_str(x) for x in names.tolist()]
    if isinstance(names, (list, tuple)):
        return [ensure_str(x) for x in names]
    return [ensure_str(names)]


def sample_scores(sample):
    scores = sample.get("score", None)
    if scores is None:
        return np.array([], dtype=np.float32)
    if isinstance(scores, np.ndarray):
        return scores.astype(np.float32)
    return np.array(scores, dtype=np.float32)


def summarize_sample(sample):
    frame_id = normalize_frame_id(sample.get("frame_id", None))
    n = sample_num_preds(sample)
    names = sample_classes(sample)
    scores = sample_scores(sample)

    cls_counter = Counter(names)
    mean_score = float(scores.mean()) if len(scores) > 0 else 0.0
    max_score = float(scores.max()) if len(scores) > 0 else 0.0

    return {
        "frame_id": frame_id,
        "num_preds": n,
        "class_counter": dict(cls_counter),
        "class_set": sorted(list(cls_counter.keys())),
        "mean_score": mean_score,
        "max_score": max_score,
    }


def result_pkl_path(root, model_name, epoch):
    return os.path.join(
        root, model_name, "default", "eval", "eval_with_train",
        f"epoch_{epoch}", "val", "result.pkl"
    )


def build_model_frame_map(root, model_name, epoch):
    pkl_path = result_pkl_path(root, model_name, epoch)
    data = load_pkl(pkl_path)

    frame_map = {}
    missing_frame_idx = []

    for idx, sample in enumerate(data):
        info = summarize_sample(sample)
        fid = info["frame_id"]
        if fid is None:
            missing_frame_idx.append(idx)
            continue
        frame_map[fid] = info

    return frame_map, pkl_path, missing_frame_idx


def safe_get_num(info_dict, key):
    if info_dict is None:
        return 0
    return info_dict.get(key, 0)


def class_diff_score(base_cls, other_cls):
    """
    简单类别差异分数：对称差大小
    """
    s1 = set(base_cls)
    s2 = set(other_cls)
    return len(s1.symmetric_difference(s2))


def aggregate_candidate_score(row):
    """
    给每个 frame 打一个启发式分数，越大越值得看
    图5.8 倾向于：
    - baseline 与 fusion 差异大
    - fusion 比两个单模块更“有内容”
    - 多目标 / 分数更高更稳定
    """
    base = row["CasA-V"]
    scaf = row["SCAFNet"]
    fgsp = row["CasA_V_fgsp_V1"]
    fusion = row["SCAF-FGSPNet"]

    # 预测框数差异
    diff_fusion_vs_base = abs(fusion["num_preds"] - base["num_preds"])
    diff_scaf_vs_base = abs(scaf["num_preds"] - base["num_preds"])
    diff_fgsp_vs_base = abs(fgsp["num_preds"] - base["num_preds"])

    # 类别集合差异
    cls_diff_fusion_base = class_diff_score(base["class_set"], fusion["class_set"])
    cls_diff_scaf_base = class_diff_score(base["class_set"], scaf["class_set"])
    cls_diff_fgsp_base = class_diff_score(base["class_set"], fgsp["class_set"])

    # 多目标帧偏好
    multi_obj_bonus = min(fusion["num_preds"], 6)

    # fusion 相对单模块的额外变化
    fusion_vs_singles = abs(fusion["num_preds"] - scaf["num_preds"]) + abs(fusion["num_preds"] - fgsp["num_preds"])

    # 分数稳定性
    score_bonus = fusion["mean_score"] + fusion["max_score"]

    total = (
        3.0 * diff_fusion_vs_base +
        1.5 * diff_scaf_vs_base +
        1.5 * diff_fgsp_vs_base +
        2.0 * cls_diff_fusion_base +
        1.0 * cls_diff_scaf_base +
        1.0 * cls_diff_fgsp_base +
        1.2 * fusion_vs_singles +
        0.8 * multi_obj_bonus +
        2.0 * score_bonus
    )
    return float(total)


def row_to_printable(fid, row, score):
    return {
        "frame_id": fid,
        "candidate_score": round(score, 4),
        "CasA-V": row["CasA-V"],
        "SCAFNet": row["SCAFNet"],
        "CasA_V_fgsp_V1": row["CasA_V_fgsp_V1"],
        "SCAF-FGSPNet": row["SCAF-FGSPNet"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models"
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="fig58_candidates.json"
    )
    parser.add_argument(
        "--save_txt",
        type=str,
        default="fig58_candidates_top50.txt"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    save_json = os.path.abspath(args.save_json)
    save_txt = os.path.abspath(args.save_txt)

    model_maps = {}
    for model, epoch in MODEL_CONFIG.items():
        fmap, pkl_path, missing = build_model_frame_map(root, model, epoch)
        print("=" * 100)
        print(f"[MODEL] {model} | epoch={epoch}")
        print(f"result: {pkl_path}")
        print(f"valid frame_ids: {len(fmap)}")
        if missing:
            print(f"missing frame_id entries: {len(missing)}")
        model_maps[model] = fmap

    common_ids = set(model_maps["CasA-V"].keys())
    for model in MODEL_CONFIG:
        common_ids &= set(model_maps[model].keys())

    common_ids = sorted(common_ids)
    print("=" * 100)
    print(f"Common frame_ids across 4 models: {len(common_ids)}")

    candidates = []
    for fid in common_ids:
        row = {
            model: model_maps[model][fid]
            for model in MODEL_CONFIG
        }

        # 过滤过于平淡的样本：四模型预测数量完全一样且类别集合一样
        nums = [row[m]["num_preds"] for m in MODEL_CONFIG]
        cls_sets = [tuple(row[m]["class_set"]) for m in MODEL_CONFIG]

        if len(set(nums)) == 1 and len(set(cls_sets)) == 1:
            continue

        score = aggregate_candidate_score(row)
        candidates.append((fid, row, score))

    candidates.sort(key=lambda x: x[2], reverse=True)

    topk = candidates[:args.topk]

    # 保存 json
    json_out = {
        "model_epochs": MODEL_CONFIG,
        "num_common_ids": len(common_ids),
        "num_candidates_after_filter": len(candidates),
        "top_candidates": [
            row_to_printable(fid, row, score)
            for fid, row, score in topk
        ]
    }

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # 保存 txt，方便命令行看
    with open(save_txt, "w", encoding="utf-8") as f:
        f.write("Top candidates for Figure 5.8\n")
        f.write(f"Model epochs: {MODEL_CONFIG}\n")
        f.write(f"Common frame_ids: {len(common_ids)}\n")
        f.write(f"Candidates after filter: {len(candidates)}\n\n")

        for rank, (fid, row, score) in enumerate(topk, start=1):
            f.write("=" * 100 + "\n")
            f.write(f"Rank {rank:02d} | frame_id={fid} | candidate_score={score:.4f}\n")
            for model in MODEL_CONFIG:
                info = row[model]
                f.write(
                    f"  [{model}] "
                    f"num_preds={info['num_preds']}, "
                    f"class_set={info['class_set']}, "
                    f"mean_score={info['mean_score']:.4f}, "
                    f"max_score={info['max_score']:.4f}\n"
                )

    print("=" * 100)
    print(f"Saved JSON: {save_json}")
    print(f"Saved TXT : {save_txt}")
    print("=" * 100)

    print("\nTop 10 candidates:")
    for rank, (fid, row, score) in enumerate(topk[:10], start=1):
        print("-" * 100)
        print(f"Rank {rank:02d} | frame_id={fid} | candidate_score={score:.4f}")
        for model in MODEL_CONFIG:
            info = row[model]
            print(
                f"  [{model}] "
                f"num_preds={info['num_preds']}, "
                f"class_set={info['class_set']}, "
                f"mean_score={info['mean_score']:.4f}, "
                f"max_score={info['max_score']:.4f}"
            )


if __name__ == "__main__":
    main()
