#!/usr/bin/env python3
import argparse
import os

import numpy as np

o3d = None

CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "PalletTruck",
}


def load_open3d():
    global o3d
    if o3d is None:
        import open3d as open3d_module
        o3d = open3d_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove precomputed static/background points from one PLY frame and visualize dynamic points."
    )
    parser.add_argument("--pcd-root", required=True,
                        help="Root point cloud folder.")
    parser.add_argument("--split", default="train",
                        help="Split name.")
    parser.add_argument("--scene-name", required=True,
                        help="Scene name, e.g. Warehouse_000.")
    parser.add_argument("--frame-id", type=int, required=True,
                        help="Frame id to load.")
    parser.add_argument("--static-voxel-size", type=float, default=0.05,
                        help="Voxel size used in static PLY filename.")
    parser.add_argument("--compare-voxel-size", type=float, default=0.05,
                        help="Voxel size for downsampling before distance comparison.")
    parser.add_argument("--distance-threshold", type=float, default=0.10,
                        help="Frame points farther than this distance from static cloud are considered dynamic.")
    parser.add_argument("--save", action="store_true",
                        help="Save dynamic-only PLY.")
    parser.add_argument("--no-view", action="store_true",
                        help="Do not open Open3D visualizer.")
    parser.add_argument("--paint", action="store_true",
                        help="Paint dynamic points green.")
    parser.add_argument("--show-static", action="store_true",
                        help="Visualize original frame gray, static red, and dynamic green.")
    parser.add_argument("--output-dir", default=None,
                        help="Optional output directory. Defaults to <pcd-root>/<split>/<scene>/dynamic.")
    parser.add_argument("--show-gt", action="store_true",
                        help="Visualize GT 3D bounding boxes.")
    parser.add_argument("--gt-path", default=None,
                        help="Optional custom GT txt path.")
    parser.add_argument("--gt-color-by-class", action="store_true",
                        help="Color GT boxes by class instead of using red for all boxes.")
    parser.add_argument("--gt-line-width", type=float, default=2.0,
                        help="GT box line width.")
    parser.add_argument("--hide-dynamic", action="store_true",
                        help="Hide dynamic points when visualizing.")
    parser.add_argument("--color-by-height", action="store_true",
                        help="Color dynamic points by coordinate height.")
    parser.add_argument("--height-axis", choices=["x", "y", "z"], default="z",
                        help="Axis used for height coloring.")
    parser.add_argument("--height-colormap", choices=["jet", "turbo", "viridis", "plasma"], default="jet",
                        help="Matplotlib colormap used for height coloring.")
    parser.add_argument("--height-min", type=float, default=None,
                        help="Optional minimum axis value for height normalization.")
    parser.add_argument("--height-max", type=float, default=None,
                        help="Optional maximum axis value for height normalization.")
    return parser.parse_args()


def is_ply_file(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            first_line = f.readline().rstrip(b"\r\n")
    except OSError:
        return False
    return first_line == b"ply"


def require_valid_ply(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} PLY does not exist: {path}")
    if not is_ply_file(path):
        raise ValueError(f"{label} file is not a valid PLY: {path}")


def voxel_mm(voxel_size):
    return int(voxel_size * 1000)


def threshold_mm(distance_threshold):
    return int(distance_threshold * 1000)


def build_frame_path(pcd_root, split, scene_name, frame_id):
    return os.path.join(
        pcd_root,
        split,
        scene_name,
        "pcd",
        f"{scene_name}_{frame_id:05d}.ply",
    )


def build_static_path(pcd_root, split, scene_name, static_voxel_size):
    return os.path.join(
        pcd_root,
        split,
        scene_name,
        "static",
        f"{scene_name}_static_voxel{voxel_mm(static_voxel_size):03d}.ply",
    )


def build_output_path(args):
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            args.pcd_root,
            args.split,
            args.scene_name,
            "dynamic",
        )
    return os.path.join(
        output_dir,
        f"{args.scene_name}_{args.frame_id:05d}_dynamic_thr{threshold_mm(args.distance_threshold):03d}.ply",
    )


def build_gt_path(pcd_root, split, scene_name, frame_id):
    return os.path.join(
        pcd_root,
        split,
        scene_name,
        "gt",
        f"{scene_name}_{frame_id:05d}.txt",
    )


def load_point_cloud(path, label):
    require_valid_ply(path, label)
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"{label} point cloud is empty: {path}")
    return pcd


def copy_colors_or_zeros(source_pcd, mask):
    if source_pcd.has_colors():
        colors = np.asarray(source_pcd.colors)
        return colors[mask]
    return np.zeros((int(np.count_nonzero(mask)), 3), dtype=np.float64)


def make_dynamic_cloud(frame_ds, mask, paint):
    points = np.asarray(frame_ds.points)[mask]
    dynamic_pcd = o3d.geometry.PointCloud()
    dynamic_pcd.points = o3d.utility.Vector3dVector(points)

    if paint:
        colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(points), 1))
    else:
        colors = copy_colors_or_zeros(frame_ds, mask)
    dynamic_pcd.colors = o3d.utility.Vector3dVector(colors)
    return dynamic_pcd


def apply_height_coloring(pcd, axis, colormap_name, height_min, height_max):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("Warning: dynamic cloud is empty; skipping height coloring.")
        return

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    values = points[:, axis_index]
    value_min = float(np.min(values)) if height_min is None else height_min
    value_max = float(np.max(values)) if height_max is None else height_max
    if value_max <= value_min:
        print(
            f"Warning: invalid height range [{value_min}, {value_max}]; "
            "using zeros for normalized height."
        )
        normalized = np.zeros_like(values, dtype=np.float64)
    else:
        normalized = np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0)

    try:
        from matplotlib import colormaps
        cmap = colormaps[colormap_name]
    except (ImportError, AttributeError):
        from matplotlib import cm
        cmap = cm.get_cmap(colormap_name)

    colors = np.asarray(cmap(normalized))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(
        f"height coloring: axis={axis}, colormap={colormap_name}, "
        f"height_min={value_min}, height_max={value_max}"
    )


def make_painted_copy(pcd, color):
    painted = o3d.geometry.PointCloud(pcd)
    painted.paint_uniform_color(color)
    return painted


def load_gt_boxes(gt_path):
    boxes = []
    with open(gt_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 11:
                print(f"Warning: skipping malformed GT line {line_idx}: {stripped}")
                continue

            try:
                label = int(parts[0])
                object_id = int(parts[1])
            except ValueError:
                label = int(float(parts[0]))
                object_id = parts[1]

            try:
                values = [float(value) for value in parts[2:]]
            except ValueError:
                print(f"Warning: skipping non-numeric GT line {line_idx}: {stripped}")
                continue

            center = np.asarray(values[0:3], dtype=np.float64)
            extent = np.asarray(values[3:6], dtype=np.float64)
            rotation = np.asarray(values[6:9], dtype=np.float64)
            boxes.append({
                "label": label,
                "object_id": object_id,
                "center": center,
                "extent": extent,
                "rotation": rotation,
                "yaw": rotation[2],
            })
    return boxes


def create_o3d_box(center, extent, yaw, color):
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, yaw))
    box = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    lineset.paint_uniform_color(color)
    return lineset


def get_class_color(label):
    colors = {
        0: [0.0, 1.0, 0.0],
        1: [1.0, 0.55, 0.0],
        2: [0.0, 0.35, 1.0],
        3: [0.65, 0.0, 1.0],
        4: [0.0, 1.0, 1.0],
        5: [1.0, 0.0, 0.0],
        6: [1.0, 1.0, 0.0],
    }
    return colors.get(label, [1.0, 1.0, 1.0])


def summarize_gt_boxes(gt_boxes):
    distribution = {}
    for box in gt_boxes:
        label = box["label"]
        name = CLASS_NAMES.get(label, f"unknown_{label}")
        distribution[name] = distribution.get(name, 0) + 1
    return distribution


def print_gt_summary(gt_path, gt_boxes):
    print(f"gt_path: {gt_path}")
    print(f"number of GT boxes: {len(gt_boxes)}")
    print(f"class distribution: {summarize_gt_boxes(gt_boxes)}")
    print("first few boxes:")
    for box in gt_boxes[:5]:
        label = box["label"]
        label_name = CLASS_NAMES.get(label, "Unknown")
        center = box["center"].tolist()
        extent = box["extent"].tolist()
        rotation = box["rotation"].tolist()
        print(
            f"  label={label}({label_name}) object_id={box['object_id']} "
            f"center={center} extent={extent} rotation={rotation}"
        )


def draw_geometries(geometries, window_name, line_width):
    if line_width <= 0:
        raise ValueError("--gt-line-width must be positive.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for geometry in geometries:
        vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.line_width = float(line_width)
    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()

    if args.static_voxel_size <= 0:
        raise ValueError("--static-voxel-size must be positive.")
    if args.compare_voxel_size <= 0:
        raise ValueError("--compare-voxel-size must be positive.")
    if args.distance_threshold < 0:
        raise ValueError("--distance-threshold must be non-negative.")
    if args.gt_line_width <= 0:
        raise ValueError("--gt-line-width must be positive.")
    if args.height_min is not None and args.height_max is not None and args.height_max <= args.height_min:
        raise ValueError("--height-max must be greater than --height-min when both are provided.")

    frame_path = build_frame_path(args.pcd_root, args.split, args.scene_name, args.frame_id)
    static_path = build_static_path(args.pcd_root, args.split, args.scene_name, args.static_voxel_size)
    gt_path = args.gt_path if args.gt_path else build_gt_path(
        args.pcd_root,
        args.split,
        args.scene_name,
        args.frame_id,
    )
    output_path = build_output_path(args)

    print(f"frame path: {frame_path}")
    print(f"static path: {static_path}")
    if args.show_gt:
        print(f"gt_path: {gt_path}")

    load_open3d()

    frame = load_point_cloud(frame_path, "frame")
    static = load_point_cloud(static_path, "static")

    frame_points_before = len(frame.points)
    frame_ds = frame.voxel_down_sample(args.compare_voxel_size)
    static_ds = static.voxel_down_sample(args.compare_voxel_size)

    frame_points_after = len(frame_ds.points)
    static_points_after = len(static_ds.points)
    if frame_points_after == 0:
        raise RuntimeError("Frame point cloud is empty after downsampling.")
    if static_points_after == 0:
        raise RuntimeError("Static point cloud is empty after downsampling.")

    distances = np.asarray(frame_ds.compute_point_cloud_distance(static_ds))
    dynamic_mask = distances > args.distance_threshold
    dynamic_points = int(np.count_nonzero(dynamic_mask))
    removed_points = int(len(dynamic_mask) - dynamic_points)
    dynamic_ratio = dynamic_points / len(dynamic_mask)

    dynamic_pcd = make_dynamic_cloud(
        frame_ds,
        dynamic_mask,
        (args.paint or args.show_static) and not args.color_by_height,
    )
    if args.color_by_height:
        apply_height_coloring(
            dynamic_pcd,
            args.height_axis,
            args.height_colormap,
            args.height_min,
            args.height_max,
        )

    print(f"frame points before downsample: {frame_points_before}")
    print(f"frame points after downsample: {frame_points_after}")
    print(f"static points after downsample: {static_points_after}")
    print(f"dynamic points: {dynamic_points}")
    print(f"removed/static-like points: {removed_points}")
    print(f"dynamic ratio: {dynamic_ratio:.6f}")

    if args.save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_ok = o3d.io.write_point_cloud(output_path, dynamic_pcd, format="ply")
        if not write_ok:
            raise RuntimeError(f"Failed to write dynamic PLY: {output_path}")
        print(f"output path: {output_path}")

    gt_boxes = []
    if args.show_gt:
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file does not exist: {gt_path}")
        gt_boxes = load_gt_boxes(gt_path)
        print_gt_summary(gt_path, gt_boxes)
        print(f"gt_line_width: {args.gt_line_width}")

    if not args.no_view:
        geometries = []
        if args.show_static:
            frame_gray = make_painted_copy(frame_ds, [0.55, 0.55, 0.55])
            static_red = make_painted_copy(static_ds, [1.0, 0.0, 0.0])
            geometries.extend([frame_gray, static_red])
            if not args.hide_dynamic:
                if args.color_by_height:
                    geometries.append(dynamic_pcd)
                else:
                    geometries.append(make_painted_copy(dynamic_pcd, [0.0, 1.0, 0.0]))
        else:
            if not args.hide_dynamic:
                geometries.append(dynamic_pcd)

        if args.show_gt:
            for box in gt_boxes:
                color = get_class_color(box["label"]) if args.gt_color_by_class else [1.0, 0.0, 0.0]
                geometries.append(create_o3d_box(box["center"], box["extent"], box["yaw"], color))

        if not geometries:
            print("Warning: no geometries to visualize.")
            return

        window_name = "Dynamic foreground points"
        if args.show_static and args.show_gt:
            window_name = "Frame gray + Static red + Dynamic green + GT boxes"
        elif args.show_static:
            window_name = "Frame gray + Static red + Dynamic green"
        elif args.show_gt:
            window_name = "Dynamic foreground points + GT boxes"

        if args.show_gt:
            draw_geometries(geometries, window_name, args.gt_line_width)
        else:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
            )


if __name__ == "__main__":
    main()
