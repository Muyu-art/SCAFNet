import numpy as np
import cv2
import pickle
import os
from pathlib import Path


# --------------------------- 帧选择开关（新增） ---------------------------
# True：只处理下面指定的帧
# False：使用原逻辑（例如前 20 帧）
USE_SELECTED_FRAMES = True

# 推荐你选的 4 张包含 Car / Ped / Cyc 的帧
SELECTED_FRAMES = [3728, 3763, 3765, 3793]


# --------------------------- 1. 数据加载模块 ---------------------------

def load_pkl_result(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_calib(calib_path):
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calib file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        lines = f.readlines()

    P2 = np.array([float(x) for x in lines[2].strip().split()[1:]]).reshape(3, 4)
    V2C = np.array([float(x) for x in lines[5].strip().split()[1:]]).reshape(3, 4)
    V2C_ext = np.vstack((V2C, [0, 0, 0, 1]))
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = np.array([float(x) for x in lines[4].strip().split()[1:]]).reshape(3, 3)
    return P2, V2C_ext, R0_rect


# --------------------------- 2. 投影模块 ---------------------------

def project_lidar_to_image(points_lidar, P2, V2C_ext, R0_rect):
    N = points_lidar.shape[0]
    lidar_hom = np.hstack((points_lidar[:, :3], np.ones((N, 1))))

    cam_pts = (V2C_ext @ lidar_hom.T).T
    rect_pts = (R0_rect @ np.hstack((cam_pts[:, :3], np.ones((N, 1)))).T).T
    img_pts = (P2 @ rect_pts[:, :4].T).T

    valid = img_pts[:, 2] > 1e-3
    img_pts[valid, 0] /= img_pts[valid, 2]
    img_pts[valid, 1] /= img_pts[valid, 2]

    return img_pts[:, :2], rect_pts[:, 2]


def compute_box_3d(center, size, yaw):
    l, w, h = size
    x_c = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_c = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_c = [h,h,h,h,0,0,0,0]

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    corners = R @ np.vstack([x_c, y_c, z_c])
    corners = corners.T
    corners[:, 0] += center[0]
    corners[:, 1] += center[1]
    corners[:, 2] += center[2]

    return corners


# --------------------------- 3. 绘图模块 ---------------------------

def draw_projected_box(image, corners_2d, color=(0,255,0), label=None,
                       thickness=1):

    corners_2d = corners_2d.astype(int)
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    for s, e in lines:
        pt1, pt2 = tuple(corners_2d[s]), tuple(corners_2d[e])
        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

    if label:
        x, y = corners_2d[0]
        y -= int(12 + thickness * 2)

        font_scale = 0.45 + 0.05 * thickness
        inner_thick = max(1, thickness - 1)
        outline_thick = thickness

        cv2.putText(image, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,0,0), outline_thick, cv2.LINE_AA)

        cv2.putText(image, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, inner_thick, cv2.LINE_AA)

    return image


# --------------------------- 可视化模块 ---------------------------

def visualize_projected_boxes(img, calib_path, boxes_lidar, names, scores, score_thresh=0.3):

    P2, V2C_ext, R0_rect = load_calib(calib_path)

    color_map = {
        "Car": (0, 0, 255),
        "Pedestrian": (0, 255, 0),
        "Cyclist": (255, 0, 0),
    }

    img = img.copy()

    for i, score in enumerate(scores):
        if score < score_thresh:
            continue

        name = names[i]
        if name not in color_map:
            continue

        box = boxes_lidar[i]
        corners_3d = compute_box_3d(box[:3], box[3:6], box[6])
        corners_2d, depths = project_lidar_to_image(corners_3d, P2, V2C_ext, R0_rect)

        if np.mean(depths) < 0:
            continue

        img = draw_projected_box(
            img, corners_2d,
            color=color_map[name],
            label=f"{name} {score:.2f}",
            thickness=1
        )

    return img


# --------------------------- 4. 主流程 ---------------------------

def main():
    img_dir = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/data/kitti/training/image_2'
    calib_dir = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/data/kitti/training/calib'
    pkl_path = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models/SCAFNet-Lab1/default/eval/eval_with_train/epoch_160/val/result.pkl'
    out_root = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output_vis_img'

    clean_dir = f"{out_root}/clean"
    box_dir   = f"{out_root}/boxes"

    Path(clean_dir).mkdir(parents=True, exist_ok=True)
    Path(box_dir).mkdir(parents=True, exist_ok=True)

    results = load_pkl_result(pkl_path)

    # =====================================================
    # ★ 关键逻辑：选择处理哪些帧
    # =====================================================
    if USE_SELECTED_FRAMES:
        # 只选你指定的帧（不会超过数据集范围）
        iter_results = [r for r in results if int(r["frame_id"]) in SELECTED_FRAMES]
    else:
        # 默认只处理前 20 帧
        iter_results = results[:20]
    # =====================================================

    for result in iter_results:

        frame_id = int(result.get("frame_id", -1))

        img_path = f"{img_dir}/{frame_id:06d}.png"
        calib_path = f"{calib_dir}/{frame_id:06d}.txt"

        img = cv2.imread(img_path)
        if img is None:
            print(f"图像不存在：{img_path}")
            continue

        # 保存原图
        clean_save = f"{clean_dir}/{frame_id:06d}.png"
        cv2.imwrite(clean_save, img)

        # 保存带框图
        boxes = result["boxes_lidar"]
        names = result["name"]
        scores = result["score"]

        img_vis = visualize_projected_boxes(img, calib_path, boxes, names, scores)
        box_save = f"{box_dir}/{frame_id:06d}.png"
        cv2.imwrite(box_save, img_vis)

        print(f"保存 clean：{clean_save}")
        print(f"保存 boxes：{box_save}")

    print("全部图像处理完成。")



if __name__ == '__main__':
    main()
