import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


PROJECT_ROOT = Path("/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet")
STAGE1_ROOT = PROJECT_ROOT / "output" / "stage1_vis"
OUTPUT_ROOT = PROJECT_ROOT / "visualization" / "fig5_6" / "output" / "fig5_6_final_single"

MODEL_DIRS = {
    "CasA-V": STAGE1_ROOT / "CasA-V_epoch76",
    "SCAFNet": STAGE1_ROOT / "SCAFNet_epoch120",
    "CasA_V_fgsp_V1": STAGE1_ROOT / "CasA_V_fgsp_V1_epoch72",
    "SCAF-FGSPNet": STAGE1_ROOT / "SCAF-FGSPNet_epoch119",
}

PANEL_TITLES = {
    "CasA-V": "(a) CasA-V",
    "SCAFNet": "(b) SCAFNet",
    "CasA_V_fgsp_V1": "(c) CasA-V-FGSP",
    "SCAF-FGSPNet": "(d) SCAF-FGSPNet",
}

# 全局字体设置
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_kitti_points(data_root: Path, frame_id: str):
    bin_path = data_root / "velodyne" / f"{frame_id}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {bin_path}")
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts


def rot_mat_2d(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def box_to_bev_corners(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
    ], dtype=np.float32)
    rot = rot_mat_2d(yaw)
    corners = local @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_box(ax, box, color="g", lw=2.0, alpha=1.0, linestyle="-", zorder=3):
    corners = box_to_bev_corners(box)
    poly = Polygon(
        corners, closed=True, fill=False, edgecolor=color,
        linewidth=lw, alpha=alpha, linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(poly)

    center = np.array([box[0], box[1]], dtype=np.float32)
    front_mid = (corners[0] + corners[1]) / 2.0
    ax.plot(
        [center[0], front_mid[0]], [center[1], front_mid[1]],
        color=color, linewidth=lw, alpha=alpha, zorder=zorder
    )


def bev_iou_axis_aligned(box1, box2):
    x1, y1, dx1, dy1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, dx2, dy2 = box2[0], box2[1], box2[3], box2[4]

    a = [x1 - dx1 / 2, y1 - dy1 / 2, x1 + dx1 / 2, y1 + dy1 / 2]
    b = [x2 - dx2 / 2, y2 - dy2 / 2, x2 + dx2 / 2, y2 + dy2 / 2]

    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area2 = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def select_topk_rois(rois, scores, topk=10):
    if rois is None or len(rois) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if scores is None or len(scores) != len(rois):
        return rois[:min(topk, len(rois))]
    order = np.argsort(scores)[::-1][:topk]
    return rois[order]


def best_roi_indices_for_each_gt(rois, gt_boxes):
    best_indices = []
    if rois is None or len(rois) == 0 or gt_boxes is None or len(gt_boxes) == 0:
        return best_indices
    for gt in gt_boxes:
        ious = [bev_iou_axis_aligned(r[:7], gt[:7]) for r in rois]
        best_indices.append(int(np.argmax(ious)))
    return best_indices


def filter_points_for_bev(points, x_range=(0, 80), y_range=(-40, 40), z_range=(-3, 2)):
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def compute_plot_range(gt_boxes, sample_rois, default_margin=5.0):
    xs, ys = [], []
    if gt_boxes is not None and len(gt_boxes) > 0:
        xs.extend(gt_boxes[:, 0].tolist())
        ys.extend(gt_boxes[:, 1].tolist())

    for rois in sample_rois:
        if rois is not None and len(rois) > 0:
            xs.extend(rois[:, 0].tolist())
            ys.extend(rois[:, 1].tolist())

    if len(xs) == 0:
        return 0, 80, -40, 40

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_pad = max(default_margin, 0.12 * (x_max - x_min + 1e-6))
    y_pad = max(default_margin, 0.12 * (y_max - y_min + 1e-6))

    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def style_axis(ax, x_min, x_max, y_min, y_max, panel_text):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 横向矩形
    ax.set_box_aspect(1 / 1.5)

    ax.grid(False)
    ax.tick_params(labelsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 正常保留坐标轴
    # ax.set_xlabel("x", fontsize=11)
    # ax.set_ylabel("y", fontsize=11)

    # 子图标号放正下方
    ax.text(
        0.5, -0.15, panel_text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11
    )


def render_single_panel(data_root: Path, frame_id: str, model_name: str, topk: int,
                        x_min: float, x_max: float, y_min: float, y_max: float,
                        gt_boxes: np.ndarray, points: np.ndarray,
                        rois: np.ndarray, topk_rois: np.ndarray, best_indices):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    ax.scatter(
        points[:, 0], points[:, 1],
        s=0.6, c="gray", alpha=0.75, linewidths=0, zorder=0
    )

    if topk_rois is not None and len(topk_rois) > 0:
        for roi in topk_rois:
            draw_box(ax, roi[:7], color="red", lw=0.6, alpha=0.20, zorder=1)

    used = set()
    for idx in best_indices:
        if idx in used:
            continue
        used.add(idx)
        if rois is not None and 0 <= idx < len(rois):
            draw_box(ax, rois[idx][:7], color="#f39c12", lw=1.2, alpha=0.95, zorder=4)

    if gt_boxes is not None and len(gt_boxes) > 0:
        for gt in gt_boxes:
            draw_box(ax, gt[:7], color="green", lw=1.4, alpha=1.0, zorder=5)

    style_axis(ax, x_min, x_max, y_min, y_max, PANEL_TITLES[model_name])

    out_dir = OUTPUT_ROOT / frame_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_id}_{model_name}.png"

    plt.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.22)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved to: {out_path}")


def render_one_frame_all_models(data_root: Path, frame_id: str, topk: int):
    points = load_kitti_points(data_root, frame_id)
    points = filter_points_for_bev(points)

    samples = {}
    for model_name, model_dir in MODEL_DIRS.items():
        p = model_dir / f"{frame_id}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Stage1 file not found: {p}")
        samples[model_name] = load_pkl(p)

    gt_boxes = samples["CasA-V"]["gt_boxes"]
    if gt_boxes is None:
        gt_boxes = np.zeros((0, 8), dtype=np.float32)

    topk_cache = {}
    best_idx_cache = {}
    all_topk_rois = []

    for model_name, data in samples.items():
        rois = data["rois"]
        scores = data["roi_scores"]
        topk_rois = select_topk_rois(rois, scores, topk=topk)
        topk_cache[model_name] = topk_rois
        best_idx_cache[model_name] = best_roi_indices_for_each_gt(rois, gt_boxes)
        all_topk_rois.append(topk_rois)

    x_min, x_max, y_min, y_max = compute_plot_range(
        gt_boxes[:, :7] if len(gt_boxes) > 0 else gt_boxes,
        all_topk_rois
    )

    for model_name in MODEL_DIRS.keys():
        data = samples[model_name]
        render_single_panel(
            data_root=data_root,
            frame_id=frame_id,
            model_name=model_name,
            topk=topk,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            gt_boxes=gt_boxes,
            points=points,
            rois=data["rois"],
            topk_rois=topk_cache[model_name],
            best_indices=best_idx_cache[model_name]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--frame_list", type=str, nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    for frame_id in args.frame_list:
        render_one_frame_all_models(data_root, frame_id, args.topk)


if __name__ == "__main__":
    main()