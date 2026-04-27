#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


TARGET_MODELS = [
    "CasA-V",
    "SCAFNet",
    "CasA_V_fgsp_V1",
    "SCAF-FGSPNet",
]


def safe_print(msg=""):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))


def format_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.2f}{u}"
        size /= 1024.0
    return f"{num_bytes}B"


def tree_dir(root, max_depth=3, max_entries_per_dir=20):
    """
    返回目录树字符串列表
    """
    root = Path(root)
    lines = [str(root)]
    if not root.exists():
        lines.append("  [NOT FOUND]")
        return lines

    def _walk(cur, prefix="", depth=0):
        if depth >= max_depth:
            return
        try:
            entries = sorted(list(cur.iterdir()), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            lines.append(prefix + "[Permission Denied]")
            return

        if len(entries) > max_entries_per_dir:
            shown = entries[:max_entries_per_dir]
            omitted = len(entries) - max_entries_per_dir
        else:
            shown = entries
            omitted = 0

        for i, p in enumerate(shown):
            connector = "└── " if i == len(shown) - 1 and omitted == 0 else "├── "
            tag = "/" if p.is_dir() else ""
            lines.append(prefix + connector + p.name + tag)
            if p.is_dir():
                next_prefix = prefix + ("    " if connector == "└── " else "│   ")
                _walk(p, next_prefix, depth + 1)

        if omitted > 0:
            lines.append(prefix + f"└── ... ({omitted} more entries)")

    _walk(root)
    return lines


def summarize_files(model_dir):
    """
    统计文件数量、后缀、重点文件
    """
    suffix_counter = Counter()
    file_list = []
    candidate_files = []

    keywords = [
        "result", "results",
        "pred", "prediction", "predictions",
        "bbox", "boxes",
        "proposal", "proposals",
        "roi", "response", "heatmap",
        "support", "foreground", "fgsp",
        "vis", "visual", "plot",
        "eval", "metric"
    ]

    for root, _, files in os.walk(model_dir):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, model_dir)
            suffix = Path(f).suffix.lower() or "[no_suffix]"
            suffix_counter[suffix] += 1
            file_list.append(path)

            lower_rel = rel.lower()
            if any(k in lower_rel for k in keywords):
                candidate_files.append(path)

    return suffix_counter, file_list, sorted(candidate_files)


def inspect_txt_file(path, max_lines=20):
    info = {"type": "txt", "preview": []}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                info["preview"].append(line.rstrip("\n"))
    except Exception as e:
        info["error"] = str(e)
    return info


def inspect_json_file(path):
    info = {"type": "json"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        info["python_type"] = type(obj).__name__
        info["summary"] = summarize_object(obj, depth=2)
    except Exception as e:
        info["error"] = str(e)
    return info


def inspect_npy_file(path):
    info = {"type": "npy"}
    if np is None:
        info["error"] = "numpy not installed"
        return info
    try:
        arr = np.load(path, allow_pickle=True)
        info["python_type"] = type(arr).__name__
        if isinstance(arr, np.ndarray):
            info["shape"] = arr.shape
            info["dtype"] = str(arr.dtype)
            if arr.dtype == object and arr.size > 0:
                try:
                    sample = arr.flat[0]
                    info["sample_summary"] = summarize_object(sample, depth=2)
                except Exception as e:
                    info["sample_summary_error"] = str(e)
        else:
            info["summary"] = summarize_object(arr, depth=2)
    except Exception as e:
        info["error"] = str(e)
    return info


def inspect_pkl_file(path):
    info = {"type": "pkl"}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        info["python_type"] = type(obj).__name__
        info["summary"] = summarize_object(obj, depth=3)
    except Exception as e:
        info["error"] = str(e)
    return info


def summarize_object(obj, depth=2):
    """
    尽量概括字段结构，不打印太多内容
    """
    if depth < 0:
        return f"<{type(obj).__name__}>"

    if isinstance(obj, dict):
        result = {
            "__type__": "dict",
            "__len__": len(obj),
            "keys": list(obj.keys())[:20]
        }
        sample_detail = {}
        for k in list(obj.keys())[:10]:
            try:
                sample_detail[str(k)] = summarize_object(obj[k], depth - 1)
            except Exception as e:
                sample_detail[str(k)] = f"<error: {e}>"
        result["sample_detail"] = sample_detail
        return result

    if isinstance(obj, (list, tuple)):
        result = {
            "__type__": type(obj).__name__,
            "__len__": len(obj)
        }
        if len(obj) > 0:
            result["first_item"] = summarize_object(obj[0], depth - 1)
        return result

    if np is not None and isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": obj.shape,
            "dtype": str(obj.dtype)
        }

    # 常见 mmengine / mmdet3d 结果对象
    if hasattr(obj, "__dict__"):
        attrs = list(vars(obj).keys())[:20]
        result = {
            "__type__": type(obj).__name__,
            "attrs": attrs
        }
        sample_detail = {}
        for k in attrs[:10]:
            try:
                sample_detail[k] = summarize_object(getattr(obj, k), depth - 1)
            except Exception as e:
                sample_detail[k] = f"<error: {e}>"
        result["sample_detail"] = sample_detail
        return result

    return repr(obj)[:200]


def inspect_file(path):
    suffix = Path(path).suffix.lower()
    size = os.path.getsize(path)
    info = {
        "path": path,
        "size": format_size(size)
    }

    if suffix in [".txt", ".log", ".md", ".csv"]:
        info["content"] = inspect_txt_file(path)
    elif suffix == ".json":
        info["content"] = inspect_json_file(path)
    elif suffix in [".npy", ".npz"]:
        info["content"] = inspect_npy_file(path)
    elif suffix in [".pkl", ".pickle"]:
        info["content"] = inspect_pkl_file(path)
    else:
        info["content"] = {"type": suffix or "[no_suffix]", "note": "binary or unsupported preview"}
    return info


def choose_files_for_deep_inspection(file_list, candidate_files, max_files=12):
    """
    优先检查更可能是结果文件的内容
    """
    preferred_suffix = {".pkl", ".json", ".txt", ".npy", ".npz"}
    picked = []

    # 先选关键词命中的
    for p in candidate_files:
        if Path(p).suffix.lower() in preferred_suffix:
            picked.append(p)

    # 再补充常见结构文件
    if len(picked) < max_files:
        for p in file_list:
            if Path(p).suffix.lower() in preferred_suffix and p not in picked:
                picked.append(p)
            if len(picked) >= max_files:
                break

    return picked[:max_files]


def inspect_model(model_dir, output_stream):
    model_name = os.path.basename(model_dir.rstrip("/"))
    safe_print("=" * 100)
    safe_print(f"[MODEL] {model_name}")
    safe_print("=" * 100)

    output_stream.write("=" * 100 + "\n")
    output_stream.write(f"[MODEL] {model_name}\n")
    output_stream.write("=" * 100 + "\n")

    # 1. 目录树
    tree_lines = tree_dir(model_dir, max_depth=3, max_entries_per_dir=25)
    safe_print("\n[1] Directory Tree")
    for line in tree_lines:
        safe_print(line)
        output_stream.write(line + "\n")

    # 2. 文件统计
    suffix_counter, file_list, candidate_files = summarize_files(model_dir)
    safe_print("\n[2] File Type Statistics")
    output_stream.write("\n[2] File Type Statistics\n")
    for suf, cnt in suffix_counter.most_common():
        line = f"  {suf:<10} : {cnt}"
        safe_print(line)
        output_stream.write(line + "\n")

    # 3. 候选结果文件
    safe_print("\n[3] Candidate Result Files")
    output_stream.write("\n[3] Candidate Result Files\n")
    if candidate_files:
        for p in candidate_files[:30]:
            rel = os.path.relpath(p, model_dir)
            line = f"  {rel}"
            safe_print(line)
            output_stream.write(line + "\n")
        if len(candidate_files) > 30:
            more_line = f"  ... ({len(candidate_files) - 30} more)"
            safe_print(more_line)
            output_stream.write(more_line + "\n")
    else:
        line = "  [No obvious candidate files found by keyword matching]"
        safe_print(line)
        output_stream.write(line + "\n")

    # 4. 深度检查文件内容
    picked = choose_files_for_deep_inspection(file_list, candidate_files, max_files=12)
    safe_print("\n[4] Deep Inspection")
    output_stream.write("\n[4] Deep Inspection\n")

    if not picked:
        line = "  [No parsable files selected]"
        safe_print(line)
        output_stream.write(line + "\n")
        return

    for p in picked:
        rel = os.path.relpath(p, model_dir)
        safe_print(f"\n--- {rel} ---")
        output_stream.write(f"\n--- {rel} ---\n")
        info = inspect_file(p)

        try:
            pretty = json.dumps(info, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pretty = str(info)

        safe_print(pretty)
        output_stream.write(pretty + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect result directory structure and field schema for multiple KITTI models."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models",
        help="Root directory containing model folders."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=TARGET_MODELS,
        help="Model folder names to inspect."
    )
    parser.add_argument(
        "--save",
        type=str,
        default="inspect_results_report.txt",
        help="Path to save report."
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    report_path = os.path.abspath(args.save)

    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write(f"Root: {root}\n")
        fout.write(f"Models: {args.models}\n\n")

        for model in args.models:
            model_dir = os.path.join(root, model)
            if not os.path.exists(model_dir):
                msg = f"[WARNING] Model directory not found: {model_dir}"
                safe_print(msg)
                fout.write(msg + "\n")
                continue
            inspect_model(model_dir, fout)

    safe_print("\n" + "=" * 100)
    safe_print(f"Report saved to: {report_path}")
    safe_print("=" * 100)


if __name__ == "__main__":
    main()
