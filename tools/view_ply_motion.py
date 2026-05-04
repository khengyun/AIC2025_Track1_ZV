#!/usr/bin/env python3
"""
Visualize AIC2025 fused multi-camera .ply point clouds with Open3D.

Features:
1. View one .ply frame.
2. View .ply sequence as motion.
3. Overlay GT 3D boxes/classes from ground_truth.json if available.
4. Visualize camera positions/frustums from calibration.json.

Examples:

# 1) View one PLY
python tools/view_ply_motion.py \
  --mode single \
  --ply dataset/pcd_dataset/test/pcd/Warehouse_017_00000.ply \
  --voxel 0.15 \
  --axis

# 2) View motion sequence
python tools/view_ply_motion.py \
  --mode sequence \
  --folder dataset/pcd_dataset/test/pcd \
  --scene Warehouse_017 \
  --voxel 0.20 \
  --sleep 0.05 \
  --max-files 300 \
  --axis

# 3) View camera positions/frustums
python tools/view_ply_motion.py \
  --mode single \
  --ply dataset/pcd_dataset/test/pcd/Warehouse_017_00000.ply \
  --calibration-json /home/niran/Documents/data/aic2025/test/Warehouse_017/calibration.json \
  --show-cameras \
  --camera-scale 2.0 \
  --camera-axis \
  --voxel 0.20 \
  --axis

# 4) View Lab with GT boxes if ground_truth.json exists
python tools/view_ply_motion.py \
  --mode single \
  --ply dataset/pcd_dataset/test/pcd/Lab_000_00000.ply \
  --gt-json /home/niran/Documents/data/aic2025/test/Lab_000/ground_truth.json \
  --show-boxes \
  --show-label-text \
  --calibration-json /home/niran/Documents/data/aic2025/test/Lab_000/calibration.json \
  --show-cameras \
  --voxel 0.15 \
  --axis
"""

import argparse
import glob
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import open3d as o3d
try:
    import cv2
except ImportError:
    cv2 = None


CLASS_COLORS = {
    "Person": (1.0, 0.1, 0.1),
    "Forklift": (1.0, 0.55, 0.0),
    "NovaCarter": (0.0, 0.45, 1.0),
    "Transporter": (0.0, 0.8, 0.25),
    "FourierGR1T2": (0.75, 0.2, 1.0),
    "AgilityDigit": (1.0, 0.0, 0.75),
    "Unknown": (1.0, 1.0, 1.0),
}


def natural_key(path: str):
    """Sort paths naturally by all numbers in filename."""
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    return [int(n) for n in nums] if nums else [stem]


def extract_frame_id_from_ply_name(path: str) -> int:
    """
    Extract final number from names like:
      Warehouse_017_00000.ply -> 0
      Lab_000_01521.ply       -> 1521
    """
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Cannot extract frame id from {path}")
    return int(nums[-1])


def load_pcd(path: str, voxel: float = 0.0):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    pcd = o3d.io.read_point_cloud(path)

    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {path}")

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)

    return pcd


def print_pcd_info(path: str, pcd):
    print("=" * 80)
    print("PLY:", path)
    print("points:", len(pcd.points))
    print("has_colors:", pcd.has_colors())
    print("has_normals:", pcd.has_normals())

    bbox = pcd.get_axis_aligned_bounding_box()
    print("min_bound:", bbox.min_bound)
    print("max_bound:", bbox.max_bound)
    print("=" * 80)


# ============================================================
# GT / object box visualization
# ============================================================

def load_gt_json(path: str | None):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        data = json.load(f)

    print(f"[GT] loaded: {path}")
    print(f"[GT] num frame keys: {len(data)}")
    print(f"[GT] first keys: {list(data.keys())[:10]}")
    return data


def normalize_frame_key(frame_id: int, gt_data: dict):
    """
    Try multiple key styles because JSON keys may be:
      "0", "00000", "000000"
    """
    candidates = [
        str(frame_id),
        f"{frame_id:05d}",
        f"{frame_id:06d}",
    ]
    for k in candidates:
        if k in gt_data:
            return k
    return None


def rot_matrix_z(yaw: float):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_oriented_box(center, size, yaw, color):
    """
    Create oriented 3D box.

    AIC fields:
      center = "3d location"             [x, y, z]
      size   = "3d bounding box scale"   [sx, sy, sz]
      yaw    = z rotation
    """
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    R = rot_matrix_z(float(yaw))

    box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=size)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    line_set.paint_uniform_color(color)
    return line_set


def make_aabb_box(center, size, color):
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)

    min_bound = center - size / 2.0
    max_bound = center + size / 2.0

    box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=min_bound,
        max_bound=max_bound,
    )
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    line_set.paint_uniform_color(color)
    return line_set


def make_label_marker(position, color, radius=0.15):
    """
    Old Open3D Visualizer does not render text directly.
    This marker is a small sphere above each box.
    Label text is printed to terminal.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(np.asarray(position, dtype=np.float64))
    sphere.paint_uniform_color(color)
    return sphere


def gt_objects_to_geometries(
    gt_data: dict | None,
    frame_id: int,
    show_boxes: bool,
    show_label_text: bool,
    box_mode: str,
):
    if gt_data is None or not show_boxes:
        return []

    key = normalize_frame_key(frame_id, gt_data)
    if key is None:
        print(f"[GT] no labels for frame {frame_id}")
        return []

    objects = gt_data[key]
    geometries = []

    print(f"[GT] frame={frame_id}, key={key}, objects={len(objects)}")

    for idx, obj in enumerate(objects):
        obj_type = obj.get("object type", "Unknown")
        obj_id = obj.get("object id", idx)

        center = obj.get("3d location", None)
        size = obj.get("3d bounding box scale", None)
        rot = obj.get("3d bounding box rotation", [0.0, 0.0, 0.0])

        if center is None or size is None:
            continue

        yaw = float(rot[2]) if len(rot) >= 3 else 0.0
        color = CLASS_COLORS.get(obj_type, CLASS_COLORS["Unknown"])

        if box_mode == "aabb":
            geom = make_aabb_box(center, size, color)
        else:
            geom = make_oriented_box(center, size, yaw, color)

        geometries.append(geom)

        if show_label_text:
            label_pos = np.asarray(center, dtype=np.float64).copy()
            label_pos[2] += float(size[2]) / 2.0 + 0.3
            geometries.append(make_label_marker(label_pos, color))

            print(
                f"  [{idx}] id={obj_id} class={obj_type} "
                f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
                f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f}) "
                f"yaw={yaw:.2f}"
            )

    return geometries


def remove_dynamic_geometries(vis, geoms):
    for g in geoms:
        try:
            vis.remove_geometry(g, reset_bounding_box=False)
        except Exception:
            pass


def add_dynamic_geometries(vis, geoms):
    for g in geoms:
        vis.add_geometry(g, reset_bounding_box=False)


# ============================================================
# Camera calibration visualization
# ============================================================

def normalize_camera_name_from_calib(camera_name: str) -> str:
    """
    Match naming convention used by tools/generate_pcd_data.py:
      Camera_1 -> Camera_0001
      Camera   -> Camera_0000
    """
    parts = camera_name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return parts[0] + "_" + f"{int(parts[1]):04d}"
    return parts[0] + "_0000"


def load_camera_calibration(calibration_json: str):
    if not os.path.exists(calibration_json):
        raise FileNotFoundError(calibration_json)

    with open(calibration_json, "r") as f:
        data = json.load(f)

    cameras = {}

    for calib in data["sensors"]:
        raw_name = calib["id"]
        cam_name = normalize_camera_name_from_calib(raw_name)

        K = np.array(calib["intrinsicMatrix"], dtype=np.float64).reshape(3, 3)

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :4] = np.array(
            calib["extrinsicMatrix"],
            dtype=np.float64,
        ).reshape(3, 4)

        # In generate_pcd_data.py, point cloud is transformed by inv(extrinsic).
        # So the camera pose in world coordinate is also inv(extrinsic).
        cam_to_world = np.linalg.inv(extrinsic)

        cameras[cam_name] = {
            "raw_name": raw_name,
            "K": K,
            "extrinsic": extrinsic,
            "cam_to_world": cam_to_world,
        }

    print(f"[CALIB] loaded {len(cameras)} cameras from {calibration_json}")
    for name in sorted(cameras.keys())[:20]:
        pos = cameras[name]["cam_to_world"][:3, 3]
        print(f"  {name}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    return cameras


def make_camera_frustum(
    cam_to_world,
    scale=2.0,
    color=(0.0, 1.0, 1.0),
    flip_z=False,
):
    """
    Draw a simple camera frustum.

    Local camera convention used here:
      origin = camera center
      +Z     = view direction

    If frustums look reversed, run with --camera-flip-z.
    """
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    z = -scale if flip_z else scale

    corners = np.array(
        [
            origin,
            [-0.6 * scale, -0.4 * scale, z],
            [0.6 * scale, -0.4 * scale, z],
            [0.6 * scale, 0.4 * scale, z],
            [-0.6 * scale, 0.4 * scale, z],
        ],
        dtype=np.float64,
    )

    homo = np.concatenate(
        [corners, np.ones((corners.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    world = (cam_to_world @ homo.T).T[:, :3]

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)

    return line_set


def make_camera_axis(cam_to_world, size=1.0):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axis.transform(cam_to_world)
    return axis


def camera_numeric_id(cam_name: str, display_index: int, start_at_one=True) -> str:
    """
    Numeric camera label.
    Default: sequential 1,2,3,... after sorting camera names.
    """
    if start_at_one:
        return str(display_index + 1)

    nums = re.findall(r"\d+", cam_name)
    if nums:
        return str(int(nums[-1]))
    return str(display_index)


def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix that rotates vector a to vector b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if c > 1.0 - 1e-10:
        return np.eye(3)

    if c < -1.0 + 1e-10:
        # 180-degree rotation around any axis orthogonal to a.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)
        vx = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + 2 * vx @ vx

    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def make_cylinder_segment(p0, p1, radius, color):
    """
    Create a thick 3D line segment as a cylinder between p0 and p1.
    This makes camera IDs much bolder than Open3D LineSet.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))

    if length < 1e-8:
        return None

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius),
        height=length,
        resolution=16,
        split=1,
    )

    # Cylinder is created along local +Z. Rotate +Z to segment direction.
    R = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), vec)
    mesh.rotate(R, center=np.array([0.0, 0.0, 0.0]))
    mesh.translate((p0 + p1) / 2.0)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def make_digit_label_3d(
    label: str,
    origin,
    size=1.0,
    radius=0.05,
    color=(1.0, 1.0, 0.0),
):
    """
    Create a bold 3D seven-segment numeric label using cylinders.

    The label is placed near the camera center and visible directly in the
    Open3D window, even without saving a video.
    """
    seg_points = {
        "a": ((0.0, 1.0), (0.7, 1.0)),
        "b": ((0.7, 1.0), (0.7, 0.5)),
        "c": ((0.7, 0.5), (0.7, 0.0)),
        "d": ((0.0, 0.0), (0.7, 0.0)),
        "e": ((0.0, 0.5), (0.0, 0.0)),
        "f": ((0.0, 1.0), (0.0, 0.5)),
        "g": ((0.0, 0.5), (0.7, 0.5)),
    }

    digits = {
        "0": "abcdef",
        "1": "bc",
        "2": "abged",
        "3": "abgcd",
        "4": "fgbc",
        "5": "afgcd",
        "6": "afgecd",
        "7": "abc",
        "8": "abcdefg",
        "9": "abfgcd",
    }

    origin = np.asarray(origin, dtype=np.float64)
    label = str(label)

    merged = o3d.geometry.TriangleMesh()
    x_offset = 0.0

    for ch in label:
        if ch not in digits:
            x_offset += 0.9 * size
            continue

        for seg in digits[ch]:
            q0, q1 = seg_points[seg]
            p0 = origin + np.array([(q0[0] + x_offset) * size, q0[1] * size, 0.0])
            p1 = origin + np.array([(q1[0] + x_offset) * size, q1[1] * size, 0.0])

            cyl = make_cylinder_segment(p0, p1, radius=radius, color=color)
            if cyl is not None:
                merged += cyl

        x_offset += 0.95 * size

    merged.compute_vertex_normals()
    return merged


def render_pcd_to_image(vis, width=1280, height=720):
    """
    Capture current visualizer view as image.
    """
    image = vis.capture_screen_float_buffer()
    image_np = np.asarray(image)
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_np


def set_topdown_view(vis, pcd):
    """
    Set camera to top-down view (looking down at XY plane).
    """
    ctrl = vis.get_view_control()
    
    # Get point cloud bounds
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    # Set view above the center
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720, fx=500, fy=500, cx=640, cy=360
    )
    
    # Position camera above the center, looking down
    height_above = max(extent[0], extent[1], extent[2]) * 1.5
    extrinsic = np.eye(4)
    extrinsic[0, 3] = center[0]
    extrinsic[1, 3] = center[1]
    extrinsic[2, 3] = center[2] + height_above
    
    # Rotate to look down (rotate around X axis by 90 degrees)
    rotation = np.eye(4)
    rotation[1, 1] = 0
    rotation[1, 2] = 1
    rotation[2, 1] = -1
    rotation[2, 2] = 0
    extrinsic[:3, :3] = rotation[:3, :3]
    
    cam_params.extrinsic = extrinsic
    ctrl.convert_from_pinhole_camera_parameters(cam_params)


def camera_geometries_from_calibration(
    calibration_json: str | None,
    show_cameras: bool,
    camera_scale: float,
    camera_axis: bool,
    max_cameras: int,
    camera_flip_z: bool,
    camera_label_size: float = 0.35,
    camera_label_radius: float = 0.035,
    camera_label_z_offset: float = 0.35,
):
    if not show_cameras:
        return []

    if not calibration_json:
        raise ValueError("--calibration-json is required when --show-cameras is enabled")

    cameras = load_camera_calibration(calibration_json)

    names = sorted(cameras.keys())
    if max_cameras > 0:
        names = names[:max_cameras]

    geoms = []

    for cam_idx, name in enumerate(names):
        cam_to_world = cameras[name]["cam_to_world"]

        frustum = make_camera_frustum(
            cam_to_world=cam_to_world,
            scale=camera_scale,
            color=(0.0, 1.0, 1.0),
            flip_z=camera_flip_z,
        )
        geoms.append(frustum)

        # Replace the old yellow sphere with a bold numeric camera ID.
        cam_center = cam_to_world[:3, 3]
        label = camera_numeric_id(name, cam_idx, start_at_one=True)

        # Place label slightly beside/above camera center.
        label_origin = cam_center + np.array(
            [
                camera_scale * 0.16,
                camera_scale * 0.16,
                camera_scale * camera_label_z_offset,
            ],
            dtype=np.float64,
        )

        label_mesh = make_digit_label_3d(
            label=label,
            origin=label_origin,
            size=camera_scale * camera_label_size,
            radius=camera_scale * camera_label_radius,
            color=(1.0, 1.0, 0.0),
        )
        geoms.append(label_mesh)

        if camera_axis:
            geoms.append(make_camera_axis(cam_to_world, size=camera_scale * 0.5))

    print(f"[CALIB] visualizing {len(names)} cameras")
    print("[CALIB] camera numeric ID mapping:")
    for cam_idx, name in enumerate(names):
        print(f"  {camera_numeric_id(name, cam_idx, start_at_one=True)} -> {name}")

    return geoms

# ============================================================
# View modes
# ============================================================

def view_single(args):
    pcd = load_pcd(args.ply, args.voxel)
    print_pcd_info(args.ply, pcd)

    gt_data = load_gt_json(args.gt_json)

    frame_id = args.frame_id
    if frame_id is None:
        frame_id = extract_frame_id_from_ply_name(args.ply)

    geometries = [pcd]

    if args.axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
        geometries.append(axis)

    camera_geoms = camera_geometries_from_calibration(
        calibration_json=args.calibration_json,
        show_cameras=args.show_cameras,
        camera_scale=args.camera_scale,
        camera_axis=args.camera_axis,
        max_cameras=args.max_cameras,
        camera_flip_z=args.camera_flip_z,
        camera_label_size=args.camera_label_size,
        camera_label_radius=args.camera_label_radius,
        camera_label_z_offset=args.camera_label_z_offset,
    )
    geometries.extend(camera_geoms)

    label_geoms = gt_objects_to_geometries(
        gt_data=gt_data,
        frame_id=frame_id,
        show_boxes=args.show_boxes,
        show_label_text=args.show_label_text,
        box_mode=args.box_mode,
    )
    geometries.extend(label_geoms)

    print("\nControls:")
    print("  Left mouse  : rotate")
    print("  Right mouse : pan")
    print("  Wheel       : zoom")
    print("  P/Home      : reset view")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"AIC2025 PLY Viewer: {Path(args.ply).name}",
        width=args.width,
        height=args.height,
    )


def find_sequence_files(
    folder: str,
    scene: str | None,
    pattern: str | None,
    max_files: int,
    skip_frames: int = 0,
):
    """
    Find PLY sequence files.

    skip_frames means:
      --skip-frames 0  -> use every file
      --skip-frames 1  -> display 1, skip 1, display 1...
      --skip-frames 9  -> display every 10th file

    This operates on the sorted .ply file list, not raw video FPS.
    """
    if pattern is None:
        if scene:
            pattern = f"{scene}_*.ply"
        else:
            pattern = "*.ply"

    files = glob.glob(os.path.join(folder, pattern))
    files = sorted(files, key=natural_key)

    if skip_frames and skip_frames > 0:
        step = skip_frames + 1
        files = files[::step]

    if max_files and max_files > 0:
        files = files[:max_files]

    return files


def view_sequence(args):
    files = find_sequence_files(
        args.folder,
        args.scene,
        args.pattern,
        args.max_files,
        args.skip_frames,
    )

    print("folder:", args.folder)
    print("scene:", args.scene)
    print("pattern:", args.pattern if args.pattern else f"{args.scene}_*.ply")
    print("num_files:", len(files))
    print("skip_frames:", args.skip_frames)
    print("first files:")
    for f in files[:10]:
        print(" ", f)

    if not files:
        raise RuntimeError("No .ply files found. Check --folder, --scene, or --pattern.")

    gt_data = load_gt_json(args.gt_json)

    # Load calibration if available (for camera ID overlay).
    # AIC calibration.json uses data["sensors"], intrinsicMatrix, extrinsicMatrix.
    cameras_dict = {}
    if args.calibration_json and os.path.exists(args.calibration_json):
        print(f"Loading calibration from {args.calibration_json}...")
        try:
            cameras_dict = load_camera_calibration(args.calibration_json)
            print(f"  Loaded {len(cameras_dict)} camera definitions")
        except Exception as e:
            print(f"  [WARN] Failed to load calibration: {e}")

    first_pcd = load_pcd(files[0], args.voxel)
    print_pcd_info(files[0], first_pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"AIC2025 PLY Motion Viewer: {args.scene or args.pattern or '*.ply'}",
        width=args.width,
        height=args.height,
    )

    pcd = first_pcd
    vis.add_geometry(pcd)

    if args.axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
        vis.add_geometry(axis)

    camera_geoms = camera_geometries_from_calibration(
        calibration_json=args.calibration_json,
        show_cameras=args.show_cameras,
        camera_scale=args.camera_scale,
        camera_axis=args.camera_axis,
        max_cameras=args.max_cameras,
        camera_flip_z=args.camera_flip_z,
        camera_label_size=args.camera_label_size,
        camera_label_radius=args.camera_label_radius,
        camera_label_z_offset=args.camera_label_z_offset,
    )
    for g in camera_geoms:
        vis.add_geometry(g)

    dynamic_geoms = []

    vis.poll_events()
    vis.update_renderer()

    print("\nControls:")
    print("  Left mouse  : rotate")
    print("  Right mouse : pan")
    print("  Wheel       : zoom")
    print("  Close window to stop")
    print("\nStarting sequence...")
    time.sleep(args.initial_sleep)

    # Setup top-down view if requested
    view_params = None
    if args.topdown_view:
        print("[INFO] Setting up top-down view...")
        try:
            ctrl = vis.get_view_control()
            cam_params = ctrl.convert_to_pinhole_camera_parameters()
            # Look down from above: position at high Z, looking down
            # Standard perspective view looking down
            cam_params.extrinsic = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 50.0],  # 50 meters up
                [0.0, 0.0, 0.0, 1.0]
            ])
            ctrl.convert_from_pinhole_camera_parameters(cam_params)
            view_params = cam_params
        except Exception as e:
            print(f"[WARN] Could not set top-down view: {e}")
            try:
                view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            except Exception:
                pass
    else:
        try:
            view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        except Exception:
            pass

    # Setup video writer if needed
    video_writer = None
    frame_count = 0
    if args.save_video and cv2 is not None:
        video_path = args.save_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        fps = max(1.0 / args.sleep, 1.0)  # Frames per second
        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            (args.width, args.height)
        )
        if not video_writer.isOpened():
            print(f"[ERROR] Could not open video writer for {video_path}")
            video_writer = None
        else:
            print(f"[INFO] Recording video to {video_path} at {fps:.1f} FPS")

    try:
        for i, path in enumerate(files):
            frame_id = extract_frame_id_from_ply_name(path)
            status_str = f"[{i + 1}/{len(files)}] {Path(path).name} frame={frame_id}"

            new_pcd = load_pcd(path, args.voxel)

            pcd.points = new_pcd.points
            if new_pcd.has_colors():
                pcd.colors = new_pcd.colors
            if new_pcd.has_normals():
                pcd.normals = new_pcd.normals

            vis.update_geometry(pcd)

            remove_dynamic_geometries(vis, dynamic_geoms)
            dynamic_geoms = []

            dynamic_geoms = gt_objects_to_geometries(
                gt_data=gt_data,
                frame_id=frame_id,
                show_boxes=args.show_boxes,
                show_label_text=args.show_label_text,
                box_mode=args.box_mode,
            )
            add_dynamic_geometries(vis, dynamic_geoms)

            if view_params is not None:
                try:
                    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)
                except Exception:
                    pass

            alive = vis.poll_events()
            vis.update_renderer()

            # Capture frame for video if recording
            if video_writer is not None:
                try:
                    # Render current scene to image
                    image = vis.capture_screen_float_buffer(do_render=True)
                    # Convert from float to uint8 and RGB to BGR for OpenCV
                    image_uint8 = (np.asarray(image) * 255).astype(np.uint8)
                    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
                    
                    # Add camera IDs if requested
                    if args.show_camera_ids and cameras_dict:
                        image_bgr = add_camera_ids_to_frame(
                            image_bgr, cameras_dict, vis
                        )
                    
                    # Add frame info overlay
                    overlay_text = f"Frame: {frame_id}"
                    cv2.putText(
                        image_bgr,
                        overlay_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    video_writer.write(image_bgr)
                    frame_count += 1
                    status_str += f" -> recorded"
                except Exception as e:
                    status_str += f" [WARN video failed: {e}]"

            print(status_str)

            if not alive:
                break

            time.sleep(args.sleep)

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"\n[SUCCESS] Video saved to {args.save_video} ({frame_count} frames)")

        vis.destroy_window()


# ============================================================
# CLI
# ============================================================

def add_camera_ids_to_frame(frame, cameras_dict, vis):
    """
    Project camera positions to 2D frame and add bold numeric labels.
    This is for exported video frames. The Open3D window itself uses
    bold 3D cylinder digits near each camera center.
    """
    if cv2 is None:
        return frame

    try:
        cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        K = cam_params.intrinsic.intrinsic_matrix
        extrinsic = cam_params.extrinsic

        for cam_idx, cam_name in enumerate(sorted(cameras_dict.keys())):
            cam_data = cameras_dict[cam_name]
            cam_pos = cam_data["cam_to_world"][:3, 3]

            homo = np.append(cam_pos, 1.0)
            cam_frame = extrinsic @ homo

            if cam_frame[2] <= 0:
                continue

            proj = K @ cam_frame[:3]
            proj = proj / cam_frame[2]
            x, y = int(proj[0]), int(proj[1])

            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                label = camera_numeric_id(cam_name, cam_idx, start_at_one=True)

                cv2.circle(frame, (x, y), 18, (0, 255, 255), -1)
                cv2.circle(frame, (x, y), 20, (0, 0, 0), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                thickness = 3
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                tx = x - tw // 2
                ty = y + th // 2

                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

    except Exception as e:
        print(f"[WARN] Could not project cameras: {e}")

    return frame

def main():
    parser = argparse.ArgumentParser(
        description="Visualize AIC2025 fused .ply point clouds with Open3D."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "sequence"],
        default="single",
        help="single: view one .ply; sequence: play multiple .ply files.",
    )

    # Single mode
    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Path to one .ply file. Required for --mode single.",
    )

    # Sequence mode
    parser.add_argument(
        "--folder",
        type=str,
        default="dataset/pcd_dataset/test/pcd",
        help="Folder containing .ply files.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene prefix, e.g. Lab_000, Warehouse_017.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help='Custom glob pattern, e.g. "Lab_000_*.ply". Overrides --scene pattern.',
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=300,
        help="Max number of sequence frames. 0 means all.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help=(
            "Sequence mode only: after displaying one PLY, skip this many PLY files. "
            "Example: --skip-frames 9 displays every 10th file."
        ),
    )

    # Visualization
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.08,
        help="Voxel downsample size in meters. Larger = faster/lower detail. 0 disables.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Delay between sequence frames.",
    )
    parser.add_argument(
        "--initial-sleep",
        type=float,
        default=1.0,
        help="Pause before sequence starts.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--axis",
        action="store_true",
        help="Show world XYZ coordinate axis.",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=2.0,
        help="World axis size.",
    )

    # GT box options
    parser.add_argument(
        "--gt-json",
        type=str,
        default=None,
        help="Path to AIC ground_truth.json for this scene.",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Overlay 3D object boxes from ground_truth.json.",
    )
    parser.add_argument(
        "--show-label-text",
        action="store_true",
        help="Print class/id and show small colored marker above each box.",
    )
    parser.add_argument(
        "--box-mode",
        choices=["oriented", "aabb"],
        default="oriented",
        help="oriented uses yaw rotation; aabb ignores rotation.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Override frame id for single mode. Default: parse from .ply filename.",
    )

    # Camera calibration visualization
    parser.add_argument(
        "--calibration-json",
        type=str,
        default=None,
        help="Path to scene calibration.json. Required if --show-cameras.",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Visualize camera frustums from calibration.json.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=2.0,
        help="Camera frustum size in world units/meters.",
    )
    parser.add_argument(
        "--camera-label-size",
        type=float,
        default=0.35,
        help="Camera numeric ID size relative to --camera-scale. Larger = bigger/bolder-looking numbers.",
    )
    parser.add_argument(
        "--camera-label-radius",
        type=float,
        default=0.035,
        help="Camera numeric ID tube radius relative to --camera-scale. Larger = thicker/bolder numbers.",
    )
    parser.add_argument(
        "--camera-label-z-offset",
        type=float,
        default=0.35,
        help="Camera numeric ID z offset relative to --camera-scale.",
    )
    parser.add_argument(
        "--camera-axis",
        action="store_true",
        help="Show local XYZ axis for each camera.",
    )
    parser.add_argument(
        "--camera-flip-z",
        action="store_true",
        help="Flip camera frustum viewing direction if frustums look reversed.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=0,
        help="Max number of cameras to visualize. 0 means all.",
    )

    # Video and view options
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save sequence as MP4 video. Requires OpenCV (cv2).",
    )
    parser.add_argument(
        "--topdown-view",
        action="store_true",
        help="Start with top-down view (looking down at XY plane).",
    )
    parser.add_argument(
        "--show-camera-ids",
        action="store_true",
        help="Show camera IDs projected onto the scene. Requires --calibration-json.",
    )

    args = parser.parse_args()

    if args.save_video and cv2 is None:
        print("OpenCV (cv2) is required for video export. Install with: pip install opencv-python")
        return

    if args.show_camera_ids and not args.calibration_json:
        print("[WARN] --show-camera-ids requires --calibration-json")

    if args.mode == "single":
        if not args.ply:
            raise ValueError("--ply is required when --mode single")
        view_single(args)
    else:
        view_sequence(args)


if __name__ == "__main__":
    main()