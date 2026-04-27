import pickle
from pathlib import Path
import numpy as np

ROOT = Path("/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/stage1_vis")

MODEL_DIRS = {
    "baseline": ROOT / "CasA-V_epoch76",
    "scafnet": ROOT / "SCAFNet_epoch120",
    "fgsp": ROOT / "CasA_V_fgsp_V1_epoch72",
    "fusion": ROOT / "SCAF-FGSPNet_epoch119",
}

def bev_iou(box1, box2):
    # box: [x, y, z, dx, dy, dz, yaw]
    # 这里先用轴对齐近似，够做初筛
    x1, y1, dx1, dy1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, dx2, dy2 = box2[0], box2[1], box2[3], box2[4]

    b1 = [x1 - dx1 / 2, y1 - dy1 / 2, x1 + dx1 / 2, y1 + dy1 / 2]
    b2 = [x2 - dx2 / 2, y2 - dy2 / 2, x2 + dx2 / 2, y2 + dy2 / 2]

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def best_iou(rois, gt_box):
    if rois is None or len(rois) == 0:
        return 0.0
    return max(bev_iou(r[:7], gt_box[:7]) for r in rois)


def load_sample(model_dir, frame_id):
    fp = model_dir / f"{frame_id}.pkl"
    with open(fp, "rb") as f:
        return pickle.load(f)


def main():
    ids = sorted(set(p.stem for p in MODEL_DIRS["baseline"].glob("*.pkl")))
    rows = []

    for frame_id in ids:
        samples = {k: load_sample(v, frame_id) for k, v in MODEL_DIRS.items()}
        gt_boxes = samples["baseline"]["gt_boxes"]

        if gt_boxes is None or len(gt_boxes) == 0:
            continue

        base_scores = []
        scaf_scores = []
        fgsp_scores = []
        fusion_scores = []

        for gt in gt_boxes:
            base_scores.append(best_iou(samples["baseline"]["rois"], gt))
            scaf_scores.append(best_iou(samples["scafnet"]["rois"], gt))
            fgsp_scores.append(best_iou(samples["fgsp"]["rois"], gt))
            fusion_scores.append(best_iou(samples["fusion"]["rois"], gt))

        row = {
            "frame_id": frame_id,
            "num_gt": len(gt_boxes),
            "baseline_mean": float(np.mean(base_scores)),
            "scafnet_mean": float(np.mean(scaf_scores)),
            "fgsp_mean": float(np.mean(fgsp_scores)),
            "fusion_mean": float(np.mean(fusion_scores)),
        }
        row["scaf_gain"] = row["scafnet_mean"] - row["baseline_mean"]
        row["fgsp_gain"] = row["fgsp_mean"] - row["baseline_mean"]
        row["fusion_gain"] = row["fusion_mean"] - row["baseline_mean"]
        rows.append(row)

    # 1) SCAFNet 相对 Baseline 提升明显
    rows_scaf = sorted(rows, key=lambda x: x["scaf_gain"], reverse=True)

    # 2) FGSP 与 Baseline 差异不大
    rows_fgsp_close = sorted(rows, key=lambda x: abs(x["fgsp_gain"]))

    # 3) Fusion 也有提升
    rows_fusion = sorted(rows, key=lambda x: x["fusion_gain"], reverse=True)

    print("=" * 100)
    print("Top 20 frames where SCAFNet improves most over Baseline")
    for r in rows_scaf[:20]:
        print(r)

    print("=" * 100)
    print("Top 20 frames where FGSP is closest to Baseline")
    for r in rows_fgsp_close[:20]:
        print(r)

    print("=" * 100)
    print("Top 20 frames where Fusion improves most over Baseline")
    for r in rows_fusion[:20]:
        print(r)


if __name__ == "__main__":
    main()