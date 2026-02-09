import os
import pickle
import numpy as np
import open3d as o3d
from open3d.visualization import rendering


# --------------------------- 帧选择开关 ---------------------------
USE_SELECTED_FRAMES = True            # True=只处理 SELECTED_FRAMES
SELECTED_FRAMES = [3728, 3763, 3765, 3793]


# =========================
#  KITTI 3 类颜色配置
# =========================
BOX_COLOR = {
    'Car':        [1.00, 0.00, 0.00],
    'Pedestrian': [0.00, 1.00, 0.00],
    'Cyclist':    [0.00, 0.30, 1.00],
}

# =========================
# 纯黑点云
# =========================
PCD_COLOR = [0.0, 0.0, 0.0]


# =========================
# FOV 裁剪（前方 ±90°）
# =========================
def load_point_cloud(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]

    angle = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    mask = (
        (xyz[:, 0] > 0) &
        (angle > -90) & (angle < 90)
    )

    xyz = xyz[mask]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.paint_uniform_color(PCD_COLOR)
    return pcd


# =========================
# 加载 PKL
# =========================
def load_pkl_result(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


# =========================
# 粗线框（Cylinder）
# =========================
def create_thick_box(center, size, yaw, color, thickness=0.15):
    l, w, h = size

    x = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z = [h,h,h,h,0,0,0,0]

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    corners = (R @ np.vstack([x, y, z])) + np.array(center).reshape(3, 1)
    corners = corners.T

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    cylinders = []

    for a, b in edges:
        p1, p2 = corners[a], corners[b]
        length = np.linalg.norm(p2 - p1)

        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=thickness, height=length)
        cyl.paint_uniform_color(color)

        direction = (p2 - p1) / length
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

        if np.linalg.norm(axis) > 1e-6:
            Rm = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
            cyl.rotate(Rm, center=(0, 0, 0))

        cyl.translate((p1 + p2) / 2)
        cylinders.append(cyl)

    return cylinders


# =========================
# 渲染（纯白背景）
# =========================
def render_frame(pcd, boxes, save_path):

    renderer = rendering.OffscreenRenderer(3840, 2160)
    scene = renderer.scene

    # ----------------------
    # ★ 强制纯白背景 ★
    # ----------------------
    scene.set_background([1, 1, 1, 1])  # 白色

    # 关闭光照、HDR、天空盒（兼容不同版本 Open3D）
    try:
        scene.scene.enable_sun_light(False)
        scene.scene.enable_indirect_light(False)
        scene.scene.set_indirect_light_intensity(0.0)
    except:
        pass

    try:
        scene.scene.show_skybox(False)
    except:
        pass

    # 再次强制白背景
    scene.set_background([1, 1, 1, 1])


    # ----------------------
    # 添加点云
    # ----------------------
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 4.0

    scene.add_geometry("pcd", pcd, mat_pcd)

    all_pts = np.asarray(pcd.points)

    # ----------------------
    # 添加 3D 粗线框
    # ----------------------
    for obj in boxes:
        scene.add_geometry(str(id(obj)), obj, rendering.MaterialRecord())
        all_pts = np.vstack((all_pts, np.asarray(obj.vertices)))


    # ----------------------
    # 自动取视角
    # ----------------------
    lo = np.quantile(all_pts, 0.02, axis=0)
    hi = np.quantile(all_pts, 0.98, axis=0)

    center = (lo + hi) / 2
    center[1] = 0.0  # 左右居中

    max_extent = max((hi - lo)[0], (hi - lo)[1]) * 1.2

    eye = center + np.array([
        -max_extent * 1.0,
        0,
        max_extent * 0.65
    ])

    scene.camera.look_at(center, eye, [0, 0, 1])
    scene.camera.set_projection(30, 3840 / 2160, 0.1, 500,
                                rendering.Camera.FovType.Vertical)

    # ----------------------
    # 输出图像
    # ----------------------
    img = renderer.render_to_image()
    o3d.io.write_image(save_path, img)
    print("Saved:", save_path)



# =========================
# 单帧可视化
# =========================
def visualize_one(bin_file, result, clean_dir, box_dir, score_thresh=0.3):

    pcd = load_point_cloud(bin_file)

    boxes = []
    for i, score in enumerate(result["score"]):
        if score < score_thresh:
            continue

        cls = result["name"][i]
        if cls not in BOX_COLOR:
            continue

        x, y, z, dx, dy, dz, yaw = result["boxes_lidar"][i]

        boxes += create_thick_box((x, y, z), (dx, dy, dz), yaw,
                                  BOX_COLOR[cls],
                                  thickness=0.03)  # 可调整粗细

    fid = int(result["frame_id"])

    render_frame(pcd, [], os.path.join(clean_dir, f"{fid:06d}.png"))
    render_frame(pcd, boxes, os.path.join(box_dir, f"{fid:06d}.png"))



# =========================
# 主程序
# =========================
if __name__ == "__main__":

    bin_dir = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/data/kitti/training/velodyne'
    pkl_file = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output/kitti_models/SCAFNet-Lab1/default/eval/eval_with_train/epoch_160/val/result.pkl'
    out_root = '/home/fmm/MMDet3D/mmdetection3d/projects/SCAFNet/output_vis'

    clean_dir = os.path.join(out_root, "clean")
    box_dir = os.path.join(out_root, "boxes")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)

    results = load_pkl_result(pkl_file)

    # 非选帧模式 → 只处理前 20 张
    iter_results = results[:20] if not USE_SELECTED_FRAMES else results

    for result in iter_results:
        fid = int(result["frame_id"])

        if USE_SELECTED_FRAMES and fid not in SELECTED_FRAMES:
            continue

        bin_file = os.path.join(bin_dir, f"{fid:06d}.bin")

        print("\n📦 处理帧:", fid)
        visualize_one(bin_file, result, clean_dir, box_dir)

    print("\n🎉 完成!")
