import os
import json
import pickle
import numpy as np
from pathlib import Path

ROOT = Path("/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models")

MODEL_NAMES = [
    "CasA-V",
    "SCAFNet",
    "CasA_V_fgsp_V1",   # 注意你的目录名是下划线
    "SCAF-FGSPNet",
]

RESULT_EXTS = {".pkl", ".pickle", ".npy", ".npz", ".json"}
TARGET_KEYS = [
    "rois", "roi_boxes", "proposal_boxes", "proposals", "stage1_boxes",
    "batch_box_preds", "rpn_box_preds", "box_preds",
    "roi_scores", "cls_preds",
    "pred_boxes", "pred_scores", "pred_labels", "boxes_3d", "bboxes_3d"
]

MAX_FILES_PER_MODEL = 3   # 每个模型最多只看 3 个结果文件
MAX_DEPTH = 3             # 最多递归 3 层
MAX_ITEMS = 3             # list 里最多看前 3 个元素


def safe_load(path: Path):
    suf = path.suffix.lower()
    try:
        if suf in [".pkl", ".pickle"]:
            with open(path, "rb") as f:
                return pickle.load(f)
        elif suf == ".npy":
            return np.load(path, allow_pickle=True)
        elif suf == ".npz":
            return dict(np.load(path, allow_pickle=True))
        elif suf == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        return {"__load_error__": str(e)}
    return None


def shape_str(x):
    try:
        if isinstance(x, np.ndarray):
            return f"shape={x.shape}, dtype={x.dtype}"
        if hasattr(x, "shape"):
            return f"shape={x.shape}"
        if isinstance(x, (list, tuple)):
            return f"len={len(x)}"
        if isinstance(x, dict):
            return f"dict_keys={list(x.keys())[:8]}"
        return ""
    except:
        return ""


def scan_obj(obj, prefix="root", depth=0, hits=None):
    if hits is None:
        hits = []
    if depth > MAX_DEPTH:
        return hits

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in TARGET_KEYS:
                hits.append((f"{prefix}.{k}", type(v).__name__, shape_str(v)))
            scan_obj(v, f"{prefix}.{k}", depth + 1, hits)

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj[:MAX_ITEMS]):
            scan_obj(item, f"{prefix}[{i}]", depth + 1, hits)

    elif isinstance(obj, np.ndarray) and obj.dtype == object and obj.size > 0:
        flat = obj.reshape(-1)
        for i, item in enumerate(flat[:MAX_ITEMS]):
            scan_obj(item, f"{prefix}[{i}]", depth + 1, hits)

    return hits


def judge(hits):
    names = [x[0].split(".")[-1] for x in hits]
    strong = any(k in names for k in [
        "rois", "roi_boxes", "proposal_boxes", "proposals", "stage1_boxes", "rpn_box_preds"
    ])
    final_only = any(k in names for k in [
        "pred_boxes", "pred_scores", "pred_labels", "boxes_3d", "bboxes_3d"
    ]) and not strong

    if strong:
        return "很可能有第一阶段候选框"
    elif final_only:
        return "更像最终检测结果"
    elif hits:
        return "有相关字段，但需人工确认"
    else:
        return "没看到明显可用字段"


def print_basic(obj):
    print(f"    类型: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"    keys: {list(obj.keys())[:15]}")
    elif isinstance(obj, (list, tuple)):
        print(f"    len: {len(obj)}")
        if len(obj) > 0:
            print(f"    first type: {type(obj[0]).__name__}")
            if isinstance(obj[0], dict):
                print(f"    first keys: {list(obj[0].keys())[:15]}")
    elif isinstance(obj, np.ndarray):
        print(f"    ndarray: shape={obj.shape}, dtype={obj.dtype}")


def find_candidate_files(eval_dir: Path):
    files = []
    for p in eval_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in RESULT_EXTS:
            files.append(p)
    files = sorted(files)

    # 优先挑名字里像结果文件的
    priority = []
    others = []
    for p in files:
        name = p.name.lower()
        if any(k in name for k in ["result", "eval", "pred", "epoch", "val", "test"]):
            priority.append(p)
        else:
            others.append(p)

    picked = priority[:MAX_FILES_PER_MODEL]
    if len(picked) < MAX_FILES_PER_MODEL:
        picked += others[:MAX_FILES_PER_MODEL - len(picked)]
    return picked


def main():
    for model_name in MODEL_NAMES:
        print("=" * 100)
        print(f"[模型] {model_name}")

        model_dir = ROOT / model_name / "default" / "eval"
        if not model_dir.exists():
            print("  没找到 default/eval 目录")
            continue

        cand_files = find_candidate_files(model_dir)
        if not cand_files:
            print("  eval 下没找到 pkl/npy/npz/json 文件")
            continue

        print(f"  抽查文件数: {len(cand_files)}")

        for fp in cand_files:
            print("-" * 100)
            print(f"  文件: {fp.relative_to(ROOT)}")

            obj = safe_load(fp)
            if obj is None:
                print("    无法识别")
                continue
            if isinstance(obj, dict) and "__load_error__" in obj:
                print(f"    读取失败: {obj['__load_error__']}")
                continue

            print_basic(obj)
            hits = scan_obj(obj)

            if hits:
                print("    命中字段:")
                for h in hits[:20]:
                    print(f"      {h[0]} | {h[1]} | {h[2]}")
            else:
                print("    命中字段: 无")

            print(f"    判断: {judge(hits)}")


if __name__ == "__main__":
    main()