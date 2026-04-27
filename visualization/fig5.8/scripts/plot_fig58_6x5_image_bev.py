#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


MODEL_CONFIG = {
    "CasA-V": 72,
    "SCAFNet": 119,
    "CasA_V_fgsp_V1": 72,
    "SCAF-FGSPNet": 113,
}

TARGET_FRAME_IDS = ["003047", "004943", "005878"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def normalize_frame_id(frame_id):
    if isinstance(frame_id, np.ndarray):
        if frame_id.size == 0:
            return None
        frame_id = frame_id.reshape(-1)[0]

    s = ensure_str(frame_id).strip()
    if s == "":
        return None
    if s.isdigit():
        return f"{int(s):06d}"
    return s


def result_pkl_path(root, model_name, epoch):
    return os.path.join(
        root, model_name, "default", "eval", "eval_with_train",
        f"epoch_{epoch}", "val", "result.pkl"
    )


def build_pred_frame_map(result_pkl):
    data = load_pkl(result_pkl)
    frame_map = {}
    for sample in data:
        fid = normalize_frame_id(sample.get("frame_id", None))
        if fid is not None:
            frame_map[fid] = sample
    return frame_map


def parse_pred_sample(sample):
    if sample is None:
        return []

    names = sample.get("name", np.array([]))
    bboxes2d = sample.get("bbox", np.zeros((0, 4), dtype=np.float32))
    scores = sample.get("score", np.array([]))
    boxes_lidar = sample.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32))

    if not isinstance(names, np.ndarray):
        names = np.array(names)
    if not isinstance(bboxes2d, np.ndarray):
        bboxes2d = np.array(bboxes2d, dtype=np.float32)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores, dtype=np.float32)
    if not isinstance(boxes_lidar, np.ndarray):
        boxes_lidar = np.array(boxes_lidar, dtype=np.float32)

    n = len(names)
    preds = []
    for i in range(n):
        bbox2d = bboxes2d[i].astype(float).tolist() if i < len(bboxes2d) else None
        box_lidar = boxes_lidar[i].astype(float).tolist() if i < len(boxes_lidar) else None
        score = float(scores[i]) if i < len(scores) else 0.0
        preds.append({
            "name": ensure_str(names[i]),
            "bbox2d": bbox2d,
            "box_lidar": box_lidar,
            "score": score,
        })
    return preds


def load_kitti_image(image_dir, frame_id):
    img_path = os.path.join(image_dir, f"{frame_id}.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_velodyne_points(velodyne_dir, frame_id):
    bin_path = os.path.join(velodyne_dir, f"{frame_id}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Velodyne file not found: {bin_path}")
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts


def get_info_frame_id(info):
    # 兼容不同 info 结构
    if "image" in info and isinstance(info["image"], dict):
        if "image_idx" in info["image"]:
            return normalize_frame_id(info["image"]["image_idx"])
        if "image_path" in info["image"]:
            stem = Path(info["image"]["image_path"]).stem
            return normalize_frame_id(stem)

    if "point_cloud" in info and isinstance(info["point_cloud"], dict):
        if "lidar_idx" in info["point_cloud"]:
            return normalize_frame_id(info["point_cloud"]["lidar_idx"])

    if "sample_idx" in info:
        return normalize_frame_id(info["sample_idx"])

    return None


def build_gt_info_map(info_pkl_path):
    infos = load_pkl(info_pkl_path)
    frame_map = {}
    for info in infos:
        fid = get_info_frame_id(info)
        if fid is not None:
            frame_map[fid] = info
    return frame_map


def parse_gt_from_info(info):
    """
    从 kitti_infos_val.pkl 中读取 GT:
    - 2D bbox: annos['bbox']
    - class: annos['name']
    - lidar 3D box: annos['gt_boxes_lidar']
    """
    if info is None or "annos" not in info:
        return []

    annos = info["annos"]
    names = annos.get("name", np.array([]))
    bboxes2d = annos.get("bbox", np.zeros((0, 4), dtype=np.float32))
    gt_boxes_lidar = annos.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32))

    if not isinstance(names, np.ndarray):
        names = np.array(names)
    if not isinstance(bboxes2d, np.ndarray):
        bboxes2d = np.array(bboxes2d, dtype=np.float32)
    if not isinstance(gt_boxes_lidar, np.ndarray):
        gt_boxes_lidar = np.array(gt_boxes_lidar, dtype=np.float32)

    gts = []
    for i in range(len(names)):
        cls_name = ensure_str(names[i])
        if cls_name not in ["Car", "Pedestrian", "Cyclist"]:
            continue
        bbox2d = bboxes2d[i].astype(float).tolist() if i < len(bboxes2d) else None
        box_lidar = gt_boxes_lidar[i].astype(float).tolist() if i < len(gt_boxes_lidar) else None
        gts.append({
            "name": cls_name,
            "bbox2d": bbox2d,
            "box_lidar": box_lidar,
        })
    return gts


def draw_image_panel(ax, image, gt_boxes=None, pred_boxes=None, title=""):
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    # GT: green
    if gt_boxes is not None:
        for gt in gt_boxes:
            if gt["bbox2d"] is None:
                continue
            x1, y1, x2, y2 = gt["bbox2d"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(0, y1 - 4),
                f"GT:{gt['name']}",
                color="lime", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )

    # Pred: red
    if pred_boxes is not None:
        for pred in pred_boxes:
            if pred["bbox2d"] is None:
                continue
            x1, y1, x2, y2 = pred["bbox2d"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.8, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, min(image.shape[0] - 5, y1 + 10),
                f"{pred['name']}:{pred['score']:.2f}",
                color="red", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.65, pad=1)
            )


def boxes_to_bev_corners(boxes3d):
    """
    boxes3d: [N, 7] with [x, y, z, dx, dy, dz, yaw]
    返回 [N, 4, 2] 的 BEV corners
    """
    if boxes3d is None or len(boxes3d) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)

    boxes3d = np.asarray(boxes3d, dtype=np.float32)
    corners = []

    for box in boxes3d:
        x, y, z, dx, dy, dz, yaw = box
        local = np.array([
            [ dx / 2,  dy / 2],
            [ dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2,  dy / 2],
        ], dtype=np.float32)

        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        bev = local @ rot.T
        bev[:, 0] += x
        bev[:, 1] += y
        corners.append(bev)

    return np.stack(corners, axis=0)


def draw_bev_box(ax, corners, color="red", linewidth=1.8, linestyle="-", alpha=1.0):
    """
    corners: [4,2]
    """
    order = [0, 1, 2, 3, 0]
    xs = corners[order, 0]
    ys = corners[order, 1]
    ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

    # 车头方向线：从中心指向前边中点（0-1 边）
    center = corners.mean(axis=0)
    front_mid = (corners[0] + corners[1]) / 2.0
    ax.plot(
        [center[0], front_mid[0]],
        [center[1], front_mid[1]],
        color=color,
        linewidth=max(1.0, linewidth - 0.3),
        alpha=alpha
    )


def filter_front_view_points(points, x_range=(0.0, 70.4), y_range=(-40.0, 40.0)):
    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    return points[mask]


def draw_bev_panel(
    ax,
    points,
    gt_boxes=None,
    pred_boxes=None,
    title="",
    x_range=(0.0, 70.4),
    y_range=(-40.0, 40.0),
    point_size=0.15,
    point_color="white",
):
    ax.set_facecolor("black")

    pts = filter_front_view_points(points, x_range=x_range, y_range=y_range)

    # 点云，白色/浅色
    if len(pts) > 0:
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=point_size, c=point_color, alpha=0.9, linewidths=0
        )

    # GT: green
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_arr = np.array([g["box_lidar"] for g in gt_boxes if g["box_lidar"] is not None], dtype=np.float32)
        if len(gt_arr) > 0:
            gt_corners = boxes_to_bev_corners(gt_arr)
            for corners in gt_corners:
                draw_bev_box(ax, corners, color="lime", linewidth=2.0)

    # Pred: red
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_arr = np.array([p["box_lidar"] for p in pred_boxes if p["box_lidar"] is not None], dtype=np.float32)
        if len(pred_arr) > 0:
            pred_corners = boxes_to_bev_corners(pred_arr)
            for corners in pred_corners:
                draw_bev_box(ax, corners, color="red", linewidth=1.6)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, color="white")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel("X (forward)", color="white", fontsize=9)
    ax.set_ylabel("Y (left/right)", color="white", fontsize=9)
    ax.grid(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=str,
        default="/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models",
        help="Root directory of model outputs"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="KITTI image_2 directory"
    )
    parser.add_argument(
        "--velodyne_dir",
        type=str,
        required=True,
        help="KITTI velodyne directory"
    )
    parser.add_argument(
        "--info_pkl",
        type=str,
        required=True,
        help="KITTI info pkl, e.g. kitti_infos_val.pkl"
    )
    parser.add_argument(
        "--frame_ids",
        nargs="+",
        default=TARGET_FRAME_IDS,
        help="Frame IDs to plot"
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshold for predicted boxes"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Keep top-k predictions per panel after score filtering"
    )
    parser.add_argument(
        "--x_min",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--x_max",
        type=float,
        default=70.4
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=-40.0
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=40.0
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.15
    )
    parser.add_argument(
        "--point_color",
        type=str,
        default="white"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="fig58_6x5_image_bev_compare.png"
    )
    args = parser.parse_args()

    result_root = os.path.abspath(args.result_root)
    image_dir = os.path.abspath(args.image_dir)
    velodyne_dir = os.path.abspath(args.velodyne_dir)
    info_pkl = os.path.abspath(args.info_pkl)
    save_path = os.path.abspath(args.save_path)

    frame_ids = [f"{int(fid):06d}" if str(fid).isdigit() else str(fid) for fid in args.frame_ids]
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    # 1) 读取四模型预测
    model_frame_maps = {}
    for model_name, epoch in MODEL_CONFIG.items():
        pkl_path = result_pkl_path(result_root, model_name, epoch)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Result pkl not found: {pkl_path}")
        print(f"[INFO] Loading predictions: {model_name} <- {pkl_path}")
        model_frame_maps[model_name] = build_pred_frame_map(pkl_path)

    # 2) 读取 GT infos
    print(f"[INFO] Loading GT infos: {info_pkl}")
    gt_info_map = build_gt_info_map(info_pkl)

    # 3) 创建画布：每个样本两行
    n_samples = len(frame_ids)
    nrows = n_samples * 2
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(26, 4.8 * n_samples * 2))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Input + GT",
        "CasA-V",
        "SCAFNet",
        "CasA_V_fgsp_V1",
        "SCAF-FGSPNet"
    ]

    for i, frame_id in enumerate(frame_ids):
        print(f"[INFO] Plotting frame_id={frame_id}")
        image = load_kitti_image(image_dir, frame_id)
        points = load_velodyne_points(velodyne_dir, frame_id)

        info = gt_info_map.get(frame_id, None)
        gt_boxes = parse_gt_from_info(info)

        img_row = 2 * i
        bev_row = 2 * i + 1

        # 第一列：输入 + GT
        draw_image_panel(
            axes[img_row, 0],
            image=image,
            gt_boxes=gt_boxes,
            pred_boxes=None,
            title=f"{col_titles[0]} (Image)\nframe_id={frame_id}"
        )
        draw_bev_panel(
            axes[bev_row, 0],
            points=points,
            gt_boxes=gt_boxes,
            pred_boxes=None,
            title=f"{col_titles[0]} (BEV)",
            x_range=x_range,
            y_range=y_range,
            point_size=args.point_size,
            point_color=args.point_color,
        )

        # 后四列：四模型预测
        for j, model_name in enumerate(MODEL_CONFIG.keys(), start=1):
            sample = model_frame_maps[model_name].get(frame_id, None)
            preds = parse_pred_sample(sample)
            preds = [p for p in preds if p["score"] >= args.score_thr]
            preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:args.topk]

            draw_image_panel(
                axes[img_row, j],
                image=image,
                gt_boxes=gt_boxes,
                pred_boxes=preds,
                title=f"{col_titles[j]} (Image)\n#pred={len(preds)}"
            )
            draw_bev_panel(
                axes[bev_row, j],
                points=points,
                gt_boxes=gt_boxes,
                pred_boxes=preds,
                title=f"{col_titles[j]} (BEV)\n#pred={len(preds)}",
                x_range=x_range,
                y_range=y_range,
                point_size=args.point_size,
                point_color=args.point_color,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    print("=" * 100)
    print(f"Saved figure to: {save_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
