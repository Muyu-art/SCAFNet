#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python export_fig56_single_bev_panels.py \
#   --velodyne_dir /home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/data/kitti/training/velodyne \
#   --frame_id 004615 007279 006577 004291 001291 007274 007235 001450 006913 002577 006249 005739 006762 004173 005077 005434 003667 006650 003126 005785 007120 005994 001442 003432 004644 000981 007136 001627 005342 000378 \
#   --topk 15

import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager as fm


MODEL_STAGE1_DIRS = {
    "CasA-V": "CasA-V_epoch76",
    "SCAFNet": "SCAFNet_epoch120",
    "CasA_V_fgsp_V1": "CasA_V_fgsp_V1_epoch72",
    "SCAF-FGSPNet": "SCAF-FGSPNet_epoch119",
}

PANEL_TITLES = {
    "CasA-V": "(a) CasA-V",
    "SCAFNet": "(b) SCAFNet",
    "CasA_V_fgsp_V1": "(c) CasA-V-FGSP",
    "SCAF-FGSPNet": "(d) SCAF-FGSPNet",
}

DEFAULT_FRAME_IDS = ["006577", "004291", "006913", "004173", "006714", "005785"]

# 全局可调参数
TITLE_FONT_SIZE = 18
TICK_FONT_SIZE = 13


def setup_times_font():
    font_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/microsoft/Times New Roman.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]

    chosen_font = None
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                chosen_font = fm.FontProperties(fname=fp).get_name()
                break
            except Exception:
                continue

    if chosen_font is not None:
        plt.rcParams["font.family"] = chosen_font
    else:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "FreeSerif"]

    plt.rcParams["axes.unicode_minus"] = False


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_frame_id(frame_id):
    s = str(frame_id).strip()
    if s.isdigit():
        return f"{int(s):06d}"
    return s


def stage1_pkl_path(stage1_root, model_name, frame_id):
    subdir = MODEL_STAGE1_DIRS[model_name]
    return os.path.join(stage1_root, subdir, f"{frame_id}.pkl")


def load_velodyne_points(velodyne_dir, frame_id):
    bin_path = os.path.join(velodyne_dir, f"{frame_id}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Velodyne file not found: {bin_path}")
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts


def rot_mat_2d(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def boxes_to_bev_corners(boxes3d):
    if boxes3d is None or len(boxes3d) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)

    boxes3d = np.asarray(boxes3d, dtype=np.float32)
    corners = []

    for box in boxes3d:
        x, y, z, dx, dy, dz, yaw = box[:7]
        local = np.array([
            [ dx / 2,  dy / 2],
            [ dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2,  dy / 2],
        ], dtype=np.float32)

        rot = rot_mat_2d(yaw)
        bev = local @ rot.T
        bev[:, 0] += x
        bev[:, 1] += y
        corners.append(bev)

    return np.stack(corners, axis=0)


def draw_bev_box(ax, corners, color="red", linewidth=1.2, linestyle="-", alpha=1.0, zorder=1):
    order = [0, 1, 2, 3, 0]
    xs = corners[order, 0]
    ys = corners[order, 1]
    ax.plot(
        xs, ys,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder
    )

    center = corners.mean(axis=0)
    front_mid = (corners[0] + corners[1]) / 2.0
    ax.plot(
        [center[0], front_mid[0]],
        [center[1], front_mid[1]],
        color=color,
        linewidth=max(0.8, linewidth - 0.2),
        alpha=alpha,
        zorder=zorder
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


def filter_front_view_points(points, x_range=(0.0, 70.4), y_range=(-40.0, 40.0), z_range=(-3.0, 2.0)):
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


def parse_stage1_sample(sample):
    if sample is None:
        return {
            "frame_id": None,
            "rois": np.zeros((0, 7), dtype=np.float32),
            "roi_scores": np.zeros((0,), dtype=np.float32),
            "gt_boxes": np.zeros((0, 8), dtype=np.float32),
        }

    frame_id = ensure_frame_id(sample.get("frame_id", ""))
    rois = sample.get("rois", np.zeros((0, 7), dtype=np.float32))
    roi_scores = sample.get("roi_scores", np.zeros((0,), dtype=np.float32))
    gt_boxes = sample.get("gt_boxes", np.zeros((0, 8), dtype=np.float32))

    if not isinstance(rois, np.ndarray):
        rois = np.array(rois, dtype=np.float32)
    if not isinstance(roi_scores, np.ndarray):
        roi_scores = np.array(roi_scores, dtype=np.float32)
    if not isinstance(gt_boxes, np.ndarray):
        gt_boxes = np.array(gt_boxes, dtype=np.float32)

    return {
        "frame_id": frame_id,
        "rois": rois,
        "roi_scores": roi_scores,
        "gt_boxes": gt_boxes,
    }


def save_single_bev_panel(
    save_file,
    points,
    gt_boxes,
    rois,
    roi_scores,
    panel_title,
    x_range=(0.0, 70.4),
    y_range=(-40.0, 40.0),
    z_range=(-3.0, 2.0),
    point_size=0.18,
    point_color="#5f6f7f",
    topk=10,
    figsize=(6.0, 4.0),
):
    pts = filter_front_view_points(points, x_range=x_range, y_range=y_range, z_range=z_range)
    topk_rois = select_topk_rois(rois, roi_scores, topk=topk)
    best_indices = best_roi_indices_for_each_gt(rois, gt_boxes)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.Axes(fig, [0.12, 0.14, 0.83, 0.78])
    fig.add_axes(ax)

    # 点云
    if len(pts) > 0:
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=point_size, c=point_color, alpha=0.80, linewidths=0, zorder=0
        )

    # 1) 先画 GT
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_arr = np.array([g[:7] for g in gt_boxes], dtype=np.float32)
        gt_corners = boxes_to_bev_corners(gt_arr)
        for corners in gt_corners:
            draw_bev_box(ax, corners, color="green", linewidth=1.5, alpha=1.0, zorder=1)

    # 2) 再画 top-k proposal
    if topk_rois is not None and len(topk_rois) > 0:
        pred_corners = boxes_to_bev_corners(topk_rois)
        for corners in pred_corners:
            draw_bev_box(ax, corners, color="red", linewidth=0.8, alpha=0.20, zorder=2)

    # 3) 最后画 best proposal，置于最顶层
    used = set()
    for idx in best_indices:
        if idx in used:
            continue
        used.add(idx)
        if 0 <= idx < len(rois):
            best_corners = boxes_to_bev_corners(np.array([rois[idx]], dtype=np.float32))
            draw_bev_box(ax, best_corners[0], color="#f39c12", linewidth=1.3, alpha=0.95, zorder=3)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect("equal")
    ax.grid(False)

    ax.tick_params(labelsize=TICK_FONT_SIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.text(
        0.5, -0.16, panel_title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=TITLE_FONT_SIZE
    )

    plt.savefig(save_file, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def main():
    setup_times_font()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1_root",
        type=str,
        default="/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/stage1_vis",
        help="Root directory of stage1 exported pkls"
    )
    parser.add_argument(
        "--velodyne_dir",
        type=str,
        required=True,
        help="KITTI velodyne directory"
    )
    parser.add_argument(
        "--frame_ids",
        nargs="+",
        default=DEFAULT_FRAME_IDS,
        help="Frame IDs to export"
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=75.0)
    parser.add_argument("--y_min", type=float, default=-25.0)
    parser.add_argument("--y_max", type=float, default=25.0)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--point_size", type=float, default=0.18)
    parser.add_argument("--point_color", type=str, default="#5f6f7f")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/visualization/fig5_6/output/fig5_6_final_single",
        help="Root output dir"
    )
    args = parser.parse_args()

    stage1_root = os.path.abspath(args.stage1_root)
    velodyne_dir = os.path.abspath(args.velodyne_dir)
    output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    frame_ids = [ensure_frame_id(fid) for fid in args.frame_ids]
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    z_range = (args.z_min, args.z_max)

    saved_count = 0

    for frame_id in frame_ids:
        print(f"[INFO] Processing frame_id={frame_id}")
        points = load_velodyne_points(velodyne_dir, frame_id)

        frame_out_dir = os.path.join(output_dir, frame_id)
        os.makedirs(frame_out_dir, exist_ok=True)

        for model_name in MODEL_STAGE1_DIRS.keys():
            pkl_path = stage1_pkl_path(stage1_root, model_name, frame_id)
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Stage1 pkl not found: {pkl_path}")

            sample = parse_stage1_sample(load_pkl(pkl_path))

            save_path = os.path.join(frame_out_dir, f"{frame_id}_{model_name}.png")
            save_single_bev_panel(
                save_file=save_path,
                points=points,
                gt_boxes=sample["gt_boxes"],
                rois=sample["rois"],
                roi_scores=sample["roi_scores"],
                panel_title=PANEL_TITLES[model_name],
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                point_size=args.point_size,
                point_color=args.point_color,
                topk=args.topk,
                figsize=(6.0, 4.0),
            )
            print(f"[SAVE] {save_path}")
            saved_count += 1

    print("=" * 100)
    print(f"Saved single BEV panels to: {output_dir}")
    print(f"Total saved panels: {saved_count}")
    print("=" * 100)


if __name__ == "__main__":
    main()