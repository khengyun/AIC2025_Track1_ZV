#!/usr/bin/env python3
import argparse
import json
import os
import time

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
    parser.add_argument("--processing-root", default=None,
                        help="Optional processing root. Uses point_cloud, point_cloud_background, and point_cloud_object below it by default.")
    parser.add_argument("--data-root", default=None,
                        help="Optional raw dataset root for ground_truth.json.")
    parser.add_argument("--pcd-root", default=None,
                        help="Root point cloud folder.")
    parser.add_argument("--split", default="train",
                        help="Split name for GT/raw metadata when needed.")
    parser.add_argument("--scene-name", default=None,
                        help="Scene name, e.g. Warehouse_000.")
    parser.add_argument("--frame-id", type=int, default=None,
                        help="Frame id to load.")
    parser.add_argument("--all-scenes", action="store_true",
                        help="Process all Warehouse_* scenes under <pcd-root>.")
    parser.add_argument("--all-frames", action="store_true",
                        help="Process all frame PLY files under each selected scene.")
    parser.add_argument("--frame-stride-dynamic", type=int, default=1,
                        help="Frame stride for --all-frames batch processing.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional maximum number of frames per scene in batch mode.")
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
    parser.add_argument("--viewer", choices=["classic", "gui"], default="classic",
                        help="Viewer backend. classic uses Open3D Visualizer; gui uses SceneWidget with 3D labels.")
    parser.add_argument("--paint", action="store_true",
                        help="Paint dynamic points green.")
    parser.add_argument("--show-static", action="store_true",
                        help="Visualize original frame, static background, and dynamic points.")
    parser.add_argument("--static-color", default="0.45,0.45,0.45",
                        help='Static/background RGB color as "r,g,b" in 0..1 or 0..255 when --show-static is enabled.')
    parser.add_argument("--static-point-size", type=float, default=1.0,
                        help="Point size for static/background points in GUI mode.")
    parser.add_argument("--frame-color", default="0.25,0.25,0.25",
                        help='Full/frame RGB color as "r,g,b" in 0..1 or 0..255 when --show-static is enabled.')
    parser.add_argument("--dynamic-color", default="0,1,0",
                        help='Dynamic point RGB color as "r,g,b" in 0..1 or 0..255 when --paint or --show-static is enabled.')
    parser.add_argument("--dynamic-point-size", type=float, default=2.0,
                        help="Point size for dynamic points in GUI mode.")
    parser.add_argument("--clip-z-min", type=float, default=None,
                        help="Optional minimum Z for visualization-only point cloud clipping.")
    parser.add_argument("--clip-z-max", type=float, default=None,
                        help="Optional maximum Z for visualization-only point cloud clipping.")
    parser.add_argument("--clip-apply-to",
                        choices=["all", "frame", "static", "dynamic", "static_in_box"],
                        nargs="*",
                        default=["all"],
                        help="Point cloud visualization layers to Z-clip.")
    parser.add_argument("--output-dir", default=None,
                        help="Optional output directory. Defaults to <object-root>/<scene> when available.")
    parser.add_argument("--background-root", default=None,
                        help="Optional root for static/background PLY files.")
    parser.add_argument("--object-root", default=None,
                        help="Optional root for saved dynamic/object PLY files.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip dynamic output files that already exist when --save is enabled.")
    parser.add_argument("--show-gt", action="store_true",
                        help="Visualize GT 3D bounding boxes.")
    parser.add_argument("--show-gt-labels", action="store_true",
                        help="Show class names for GT boxes where supported and in saved previews.")
    parser.add_argument("--show-gt-ids", action="store_true",
                        help="Show object IDs for GT boxes where supported and in saved previews.")
    parser.add_argument("--gt-path", default=None,
                        help="Optional custom GT txt path.")
    parser.add_argument("--gt-source", choices=["auto", "json", "txt"], default="auto",
                        help="GT source. Auto prefers ground_truth.json then per-frame TXT.")
    parser.add_argument("--gt-box-scale", type=float, default=1.0,
                        help="Scale factor applied to GT box extents for visualization.")
    parser.add_argument("--gt-yaw-only", action="store_true",
                        help="Use only yaw (rz) from GT box rotations.")
    parser.add_argument("--gt-color-by-class", action="store_true",
                        help="Color GT boxes by class instead of using red for all boxes.")
    parser.add_argument("--gt-line-width", type=float, default=2.0,
                        help="GT box line width.")
    parser.add_argument("--show-gt-id-labels", action="store_true",
                        help="Show object IDs above GT boxes with Open3D GUI Label3D labels.")
    parser.add_argument("--gt-id-label-scale", type=float, default=1.0,
                        help="Open3D GUI GT object ID label scale.")
    parser.add_argument("--gt-id-label-color", default="1,1,0",
                        help='Open3D GUI GT object ID label RGB color as "r,g,b" in 0..1 or 0..255.')
    parser.add_argument("--gt-label-size", type=float, default=2.0,
                        help="3D GT text label size.")
    parser.add_argument("--gt-label-z-offset", type=float, default=2.0,
                        help="Vertical offset above each GT box for 3D text labels.")
    parser.add_argument("--gt-label-x-offset", type=float, default=0.0,
                        help="X offset for 3D GT text labels.")
    parser.add_argument("--gt-label-y-offset", type=float, default=0.0,
                        help="Y offset for 3D GT text labels.")
    parser.add_argument("--gt-label-color", default="1,1,1",
                        help='GT text label RGB color as "r,g,b" in 0..1 or 0..255.')
    parser.add_argument("--gt-label-orientation", choices=["xy", "xz", "yz"], default="xy",
                        help="Plane orientation for Open3D GT text labels.")
    parser.add_argument("--show-gt-id-mesh", action="store_true",
                        help="Show object IDs as robust 7-segment cylinder meshes above GT boxes.")
    parser.add_argument("--hide-gt-id-mesh", action="store_true",
                        help="Disable 3D digit mesh labels even if --show-gt-id-mesh is passed.")
    parser.add_argument("--gt-id-display", choices=["none", "index", "object_id"], default="index",
                        help="Digit mesh content: compact view index, full object ID, or none.")
    parser.add_argument("--gt-id-max-digits", type=int, default=2,
                        help="Maximum object ID digits to render before falling back to view index.")
    parser.add_argument("--gt-id-size", type=float, default=0.35,
                        help="Digit mesh object ID label size.")
    parser.add_argument("--gt-id-radius", type=float, default=0.025,
                        help="Cylinder radius for digit mesh object ID labels.")
    parser.add_argument("--gt-id-z-offset", type=float, default=0.8,
                        help="Vertical offset above each GT box for digit mesh object ID labels.")
    parser.add_argument("--gt-id-color", default="1,1,0",
                        help='Digit mesh object ID label RGB color as "r,g,b" in 0..1 or 0..255.')
    parser.add_argument("--focus-object-id", type=int, default=None,
                        help="Optional GT object ID to highlight.")
    parser.add_argument("--focus-class-name", default=None,
                        help="Optional GT class name to highlight.")
    parser.add_argument("--focus-label-text", default=None,
                        help="Optional focused label text override.")
    parser.add_argument("--show-only-focused-gt", action="store_true",
                        help="Only draw the focused GT object.")
    parser.add_argument("--focus-bbox-color", default="1,0,0",
                        help='Focused bbox RGB color as "r,g,b" in 0..1 or 0..255.')
    parser.add_argument("--focus-label-color", default="1,1,0",
                        help='Focused label RGB color as "r,g,b" in 0..1 or 0..255.')
    parser.add_argument("--focus-label-size", type=float, default=0.6,
                        help="Focused full-label mesh text size.")
    parser.add_argument("--focus-label-radius", type=float, default=0.03,
                        help="Focused full-label mesh cylinder radius.")
    parser.add_argument("--focus-label-z-offset", type=float, default=0.8,
                        help="Focused full-label vertical offset above bbox.")
    parser.add_argument("--show-static-in-gt-boxes", action="store_true",
                        help="Highlight downsampled static/background points that fall inside GT boxes.")
    parser.add_argument("--static-in-box-color", default="1,0,0",
                        help='Static-in-GT-box point RGB color as "r,g,b" in 0..1 or 0..255.')
    parser.add_argument("--static-in-box-point-size", type=float, default=4.0,
                        help="Point size for static/background points inside GT boxes in GUI mode.")
    parser.add_argument("--static-in-box-gt-box-scale", type=float, default=None,
                        help="Optional GT box scale used only for static-in-box point matching.")
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
    parser.add_argument("--save-gt-overlay", action="store_true",
                        help="Save GT overlay metadata JSON next to dynamic outputs.")
    parser.add_argument("--save-preview-png", action="store_true",
                        help="Save a top-view PNG of dynamic points and projected GT boxes.")
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


def resolve_roots(args):
    if args.pcd_root is None:
        if args.processing_root is None:
            raise ValueError("Either --pcd-root or --processing-root must be provided.")
        args.pcd_root = os.path.join(args.processing_root, "point_cloud")
    if args.processing_root is not None and args.background_root is None:
        args.background_root = os.path.join(args.processing_root, "point_cloud_background")
    if args.processing_root is not None and args.object_root is None:
        args.object_root = os.path.join(args.processing_root, "point_cloud_object")


def get_scene_dirs(pcd_root, scene_name=None, all_scenes=False):
    if scene_name and not all_scenes:
        scene_dir = os.path.join(pcd_root, scene_name)
        return [scene_dir] if os.path.isdir(scene_dir) else []
    if not os.path.isdir(pcd_root):
        return []
    return [
        os.path.join(pcd_root, name)
        for name in sorted(os.listdir(pcd_root))
        if name.startswith("Warehouse_") and os.path.isdir(os.path.join(pcd_root, name))
    ]


def parse_frame_id(path, scene_name):
    filename = os.path.basename(path)
    prefix = f"{scene_name}_"
    suffix = ".ply"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Invalid frame PLY filename for scene {scene_name}: {filename}")
    return int(filename[len(prefix):-len(suffix)])


def get_scene_frame_ids(pcd_root, scene_name, all_frames, frame_id, frame_stride_dynamic, max_frames):
    if not all_frames:
        if frame_id is None:
            raise ValueError("--frame-id is required unless --all-frames is passed.")
        return [frame_id]

    pcd_dir = os.path.join(pcd_root, scene_name, "pcd")
    frame_ids = []
    if not os.path.isdir(pcd_dir):
        return frame_ids
    for name in os.listdir(pcd_dir):
        if not name.endswith(".ply"):
            continue
        path = os.path.join(pcd_dir, name)
        try:
            parsed_frame_id = parse_frame_id(path, scene_name)
        except ValueError:
            continue
        if parsed_frame_id % frame_stride_dynamic == 0:
            frame_ids.append(parsed_frame_id)
    frame_ids.sort()
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def build_frame_path(pcd_root, scene_name, frame_id):
    return os.path.join(
        pcd_root,
        scene_name,
        "pcd",
        f"{scene_name}_{frame_id:05d}.ply",
    )


def build_static_path(pcd_root, scene_name, static_voxel_size, background_root=None):
    if background_root is not None:
        return os.path.join(
            background_root,
            scene_name,
            f"{scene_name}_static_voxel{voxel_mm(static_voxel_size):03d}.ply",
        )
    return os.path.join(
        pcd_root,
        scene_name,
        "static",
        f"{scene_name}_static_voxel{voxel_mm(static_voxel_size):03d}.ply",
    )


def build_output_path(args, scene_name=None, frame_id=None):
    scene_name = args.scene_name if scene_name is None else scene_name
    frame_id = args.frame_id if frame_id is None else frame_id
    output_dir = args.output_dir
    if args.object_root is not None:
        output_dir = os.path.join(args.object_root, scene_name)
    if output_dir is None:
        output_dir = os.path.join(
            args.pcd_root,
            scene_name,
            "dynamic",
        )
    return os.path.join(
        output_dir,
        f"{scene_name}_{frame_id:05d}_dynamic_thr{threshold_mm(args.distance_threshold):03d}.ply",
    )


def build_overlay_json_path(args, scene_name, frame_id):
    output_dir = args.object_root if args.object_root is not None else args.pcd_root
    return os.path.join(
        output_dir,
        scene_name,
        f"{scene_name}_{frame_id:05d}_gt_overlay.json",
    )


def build_static_inside_gt_path(args, scene_name, frame_id):
    output_dir = args.object_root if args.object_root is not None else args.pcd_root
    return os.path.join(
        output_dir,
        scene_name,
        f"{scene_name}_{frame_id:05d}_static_inside_gt_thr{threshold_mm(args.distance_threshold):03d}.ply",
    )


def build_preview_png_path(args, scene_name, frame_id):
    output_dir = args.object_root if args.object_root is not None else args.pcd_root
    return os.path.join(
        output_dir,
        scene_name,
        f"{scene_name}_{frame_id:05d}_dynamic_gt_topview.png",
    )


def build_gt_path(pcd_root, scene_name, frame_id):
    return os.path.join(
        pcd_root,
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


def make_dynamic_cloud(frame_ds, mask, paint, color=(0.0, 1.0, 0.0)):
    points = np.asarray(frame_ds.points)[mask]
    dynamic_pcd = o3d.geometry.PointCloud()
    dynamic_pcd.points = o3d.utility.Vector3dVector(points)

    if paint:
        colors = np.tile(np.asarray(color, dtype=np.float64), (len(points), 1))
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


def clip_point_cloud_by_z(pcd, z_min=None, z_max=None):
    points = np.asarray(pcd.points)
    before_count = len(points)
    if before_count == 0 or (z_min is None and z_max is None):
        print(f"clip_point_cloud_by_z: before={before_count} after={before_count}")
        return o3d.geometry.PointCloud(pcd)

    mask = np.ones(before_count, dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max

    clipped = o3d.geometry.PointCloud()
    clipped.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        clipped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        clipped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    print(f"clip_point_cloud_by_z: before={before_count} after={len(clipped.points)}")
    return clipped


def z_clipping_enabled(args):
    return args.clip_z_min is not None or args.clip_z_max is not None


def clip_target_enabled(args, target):
    targets = set(args.clip_apply_to or [])
    return "all" in targets or target in targets


def maybe_clip_visual_cloud(name, pcd, args):
    if pcd is None:
        if z_clipping_enabled(args):
            print(f"{name} before/after: 0 -> 0")
        return None
    before_count = len(pcd.points)
    if z_clipping_enabled(args) and clip_target_enabled(args, name):
        clipped = clip_point_cloud_by_z(pcd, args.clip_z_min, args.clip_z_max)
    else:
        clipped = o3d.geometry.PointCloud(pcd)
    if z_clipping_enabled(args):
        print(f"{name} before/after: {before_count} -> {len(clipped.points)}")
    return clipped


def prepare_visualization_clouds(args, frame_ds, static_ds, dynamic_pcd, static_in_box_pcd):
    if z_clipping_enabled(args):
        print("z clipping enabled:")
        print(f"  z_min={args.clip_z_min}")
        print(f"  z_max={args.clip_z_max}")
        print(f"  apply_to={args.clip_apply_to}")

    frame_view = maybe_clip_visual_cloud("frame", frame_ds, args) if args.show_static else frame_ds
    static_view = maybe_clip_visual_cloud("static", static_ds, args) if args.show_static else static_ds
    dynamic_view = maybe_clip_visual_cloud("dynamic", dynamic_pcd, args)
    static_in_box_view = maybe_clip_visual_cloud("static_in_box", static_in_box_pcd, args)
    return frame_view, static_view, dynamic_view, static_in_box_view


def make_gt_text(class_name, object_id, show_labels=True, show_ids=True):
    parts = []
    if show_labels:
        parts.append(class_name)
    if show_ids:
        parts.append(f"#{object_id}")
    return " ".join(parts)


def get_gt_view_mappings(gt_boxes):
    mappings = []
    for view_index, box in enumerate(gt_boxes, start=1):
        class_label = int(box["label"])
        mappings.append(
            {
                "view_index": view_index,
                "class_label": class_label,
                "class_name": CLASS_NAMES.get(class_label, f"unknown_{class_label}"),
                "object_id": box["object_id"],
                "center": np.asarray(box["center"], dtype=np.float64),
                "extent": np.asarray(box["extent"], dtype=np.float64),
                "rotation": np.asarray(box["rotation"], dtype=np.float64),
                "box": box,
            }
        )
    return mappings


def gt_mesh_display_label(mapping, args):
    if args.gt_id_display == "none":
        return None
    object_id_text = str(mapping["object_id"])
    if args.gt_id_display == "object_id" and len(object_id_text) <= args.gt_id_max_digits:
        return object_id_text
    return str(mapping["view_index"])


def object_id_matches(object_id, focus_object_id):
    if focus_object_id is None:
        return True
    try:
        return int(object_id) == int(focus_object_id)
    except (TypeError, ValueError):
        return str(object_id) == str(focus_object_id)


def is_focused_mapping(mapping, args):
    if args.focus_object_id is None and args.focus_class_name is None:
        return False
    if not object_id_matches(mapping["object_id"], args.focus_object_id):
        return False
    if args.focus_class_name is not None and mapping["class_name"] != args.focus_class_name:
        return False
    return True


def focused_label_text(mapping, args):
    if args.focus_label_text:
        return args.focus_label_text
    return make_gt_text(mapping["class_name"], mapping["object_id"], show_labels=True, show_ids=True)


def focused_object_dict(mapping, text):
    return {
        "class_name": mapping["class_name"],
        "object_id": mapping["object_id"],
        "text": text,
        "center": mapping["center"].tolist(),
        "extent": mapping["extent"].tolist(),
        "rotation": mapping["rotation"].tolist(),
    }


def make_gt_box(label, object_id, center, extent, rotation):
    class_name = CLASS_NAMES.get(label, f"unknown_{label}")
    return {
        "label": int(label),
        "class_label": int(label),
        "class_name": class_name,
        "object_id": object_id,
        "center": np.asarray(center, dtype=np.float64),
        "extent": np.asarray(extent, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
        "yaw": float(rotation[2]),
        "text": make_gt_text(class_name, object_id),
    }


def load_gt_boxes_from_txt(gt_path):
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
                try:
                    label = int(float(parts[0]))
                except ValueError:
                    print(f"Warning: skipping non-numeric GT label on line {line_idx}: {stripped}")
                    continue
                object_id = parts[1]

            try:
                values = [float(value) for value in parts[2:]]
            except ValueError:
                print(f"Warning: skipping non-numeric GT line {line_idx}: {stripped}")
                continue

            center = np.asarray(values[0:3], dtype=np.float64)
            extent = np.asarray(values[3:6], dtype=np.float64)
            rotation = np.asarray(values[6:9], dtype=np.float64)
            boxes.append(make_gt_box(label, object_id, center, extent, rotation))
    return boxes


def load_gt_boxes_from_json(gt_path, frame_id):
    with open(gt_path, "r") as f:
        data = json.load(f)

    frame_key = str(frame_id)
    if frame_key not in data:
        return None

    frame_objects = data.get(frame_key, [])
    if not isinstance(frame_objects, list):
        return []

    class_to_label = {name: label for label, name in CLASS_NAMES.items()}
    boxes = []
    for obj in frame_objects:
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("object type")
        if class_name not in class_to_label:
            continue
        try:
            boxes.append(
                make_gt_box(
                    class_to_label[class_name],
                    obj.get("object id", "unknown"),
                    obj["3d location"],
                    obj["3d bounding box scale"],
                    obj["3d bounding box rotation"],
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return boxes


def load_gt_boxes_for_frame(args, scene_name, frame_id):
    if args.gt_path:
        return args.gt_path, load_gt_boxes_from_txt(args.gt_path)

    json_path = None
    if args.data_root is not None:
        json_path = os.path.join(args.data_root, args.split, scene_name, "ground_truth.json")
    txt_path = build_gt_path(args.pcd_root, scene_name, frame_id)

    if args.gt_source in {"auto", "json"} and json_path is not None and os.path.exists(json_path):
        json_boxes = load_gt_boxes_from_json(json_path, frame_id)
        if json_boxes is not None or args.gt_source == "json":
            return json_path, json_boxes or []
    if args.gt_source == "json":
        return json_path, []
    if os.path.exists(txt_path):
        return txt_path, load_gt_boxes_from_txt(txt_path)
    return txt_path, []


def get_box_rotation(rotation, yaw_only):
    if yaw_only:
        return o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, float(rotation[2])))
    return o3d.geometry.get_rotation_matrix_from_xyz(tuple(np.asarray(rotation, dtype=np.float64).tolist()))


def create_o3d_box(box_data, color, scale, yaw_only):
    rotation = get_box_rotation(box_data["rotation"], yaw_only)
    extent = np.asarray(box_data["extent"], dtype=np.float64) * scale
    box = o3d.geometry.OrientedBoundingBox(box_data["center"], rotation, extent)
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    lineset.paint_uniform_color(color)
    return lineset


def create_o3d_obb(box_data, scale, yaw_only):
    rotation = get_box_rotation(box_data["rotation"], yaw_only)
    extent = np.asarray(box_data["extent"], dtype=np.float64) * scale
    return o3d.geometry.OrientedBoundingBox(box_data["center"], rotation, extent)


def static_in_box_scale(args):
    if args.static_in_box_gt_box_scale is not None:
        return args.static_in_box_gt_box_scale
    return args.gt_box_scale


def find_static_points_inside_gt_boxes(static_ds, gt_boxes, args):
    stats = {}
    empty = o3d.geometry.PointCloud()
    if not args.show_static_in_gt_boxes:
        return empty, stats
    if not gt_boxes:
        print("Warning: --show-static-in-gt-boxes requested but no GT boxes were loaded.")
        return empty, stats

    static_points = np.asarray(static_ds.points)
    if len(static_points) == 0:
        print("Warning: static downsampled cloud is empty; no static-in-GT points to highlight.")
        return empty, stats

    matched_mask = np.zeros(len(static_points), dtype=bool)
    box_counts = {}
    box_scale = static_in_box_scale(args)
    for mapping in get_gt_view_mappings(gt_boxes):
        obb = create_o3d_obb(mapping["box"], box_scale, args.gt_yaw_only)
        indices = np.asarray(obb.get_point_indices_within_bounding_box(static_ds.points), dtype=np.int64)
        count = int(len(indices))
        box_counts[mapping["view_index"]] = count
        if count > 0:
            matched_mask[indices] = True

    matched_points = static_points[matched_mask]
    static_in_box_pcd = o3d.geometry.PointCloud()
    static_in_box_pcd.points = o3d.utility.Vector3dVector(matched_points)
    if len(matched_points) > 0:
        colors = np.tile(args.static_in_box_color_rgb, (len(matched_points), 1))
        static_in_box_pcd.colors = o3d.utility.Vector3dVector(colors)

    boxes_with_points = sum(1 for count in box_counts.values() if count > 0)
    stats = {
        "box_counts": box_counts,
        "total_points": int(len(matched_points)),
        "boxes_with_points": int(boxes_with_points),
        "box_scale": float(box_scale),
    }
    print(f"static_points_inside_any_gt_box: {stats['total_points']}")
    print(f"gt_boxes_with_static_points: {boxes_with_points} / {len(gt_boxes)}")
    print("first 10 boxes:")
    for mapping in get_gt_view_mappings(gt_boxes)[:10]:
        print(
            f"  {mapping['class_name']} {mapping['object_id']} "
            f"{box_counts.get(mapping['view_index'], 0)}"
        )
    return static_in_box_pcd, stats


def gt_label_position(mapping, scale):
    center = np.asarray(mapping["center"], dtype=np.float64)
    extent = np.asarray(mapping["extent"], dtype=np.float64) * scale
    return center + np.asarray([0.0, 0.0, extent[2] / 2.0 + 0.5], dtype=np.float64)


def collect_gt_draw_mappings(gt_boxes, args):
    mappings = get_gt_view_mappings(gt_boxes)
    focused_mappings = [mapping for mapping in mappings if is_focused_mapping(mapping, args)]
    focused_mapping = focused_mappings[0] if focused_mappings else None
    return mappings, focused_mapping


def log_focus_status(focused_mapping, args):
    if args.focus_object_id is None and args.focus_class_name is None:
        return
    print(
        f"Focused object requested: class={args.focus_class_name} "
        f"object_id={args.focus_object_id}"
    )
    if focused_mapping is None:
        print("Focused object not found.")
        return
    focus_text = focused_label_text(focused_mapping, args)
    print(
        f"Focused object found: {focus_text} "
        f"center={focused_mapping['center'].tolist()}"
    )
    print(f"Focused object extent: {focused_mapping['extent'].tolist()}")
    print(f"Focused label text: {focus_text}")


def mapping_is_drawn(mapping, focused_mapping, args):
    is_focus = focused_mapping is not None and mapping["view_index"] == focused_mapping["view_index"]
    if args.show_only_focused_gt and not is_focus:
        return False, is_focus
    return True, is_focus


def get_gt_box_color(mapping, is_focus, args):
    if is_focus:
        return args.focus_bbox_color_rgb
    if args.gt_color_by_class:
        return get_class_color(mapping["class_label"])
    return [1.0, 0.0, 0.0]


def should_label_mapping(mapping, is_focus, args):
    if not args.show_gt_id_labels:
        return False
    focus_requested = args.focus_object_id is not None or args.focus_class_name is not None
    if focus_requested and not is_focus:
        return False
    return True


def log_gui_label_summary(label_examples, labels_added, gt_boxes_count, args):
    print(f"viewer mode: {args.viewer}")
    print(f"number of GT boxes: {gt_boxes_count}")
    print(f"number of ID labels added: {labels_added}")
    if not label_examples:
        return
    print("first 10 label mappings:")
    for example in label_examples[:10]:
        print(
            f"  id={example['object_id']} class={example['class_name']} "
            f"frame={example['frame_id']} pos={example['position'].tolist()}"
        )


def make_rendering_material(shader="defaultUnlit", color=None, point_size=None, line_width=None, rendering_module=None):
    rendering_api = rendering_module if rendering_module is not None else o3d.visualization.rendering
    material = rendering_api.MaterialRecord()
    material.shader = shader
    if color is not None:
        rgba = np.asarray(color, dtype=np.float64)
        if rgba.shape[0] == 3:
            rgba = np.concatenate([rgba, np.asarray([1.0])])
        material.base_color = np.clip(rgba, 0.0, 1.0).tolist()
    if point_size is not None and hasattr(material, "point_size"):
        material.point_size = float(point_size)
    if line_width is not None and hasattr(material, "line_width"):
        material.line_width = float(line_width)
    return material


def combined_bounds(dynamic_pcd, gt_boxes, args, static_in_box_pcd=None, static_pcd=None):
    bounds = []
    if len(dynamic_pcd.points) > 0:
        points = np.asarray(dynamic_pcd.points)
        bounds.extend([np.min(points, axis=0), np.max(points, axis=0)])
    if static_pcd is not None and len(static_pcd.points) > 0:
        points = np.asarray(static_pcd.points)
        bounds.extend([np.min(points, axis=0), np.max(points, axis=0)])
    if static_in_box_pcd is not None and len(static_in_box_pcd.points) > 0:
        points = np.asarray(static_in_box_pcd.points)
        bounds.extend([np.min(points, axis=0), np.max(points, axis=0)])
    for box in gt_boxes:
        rotation = get_box_rotation(box["rotation"], args.gt_yaw_only)
        extent = np.asarray(box["extent"], dtype=np.float64) * args.gt_box_scale
        obb = o3d.geometry.OrientedBoundingBox(box["center"], rotation, extent)
        corners = np.asarray(obb.get_box_points())
        bounds.extend([np.min(corners, axis=0), np.max(corners, axis=0)])

    if not bounds:
        return o3d.geometry.AxisAlignedBoundingBox(
            np.asarray([-1.0, -1.0, -1.0], dtype=np.float64),
            np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        )

    stacked = np.vstack(bounds)
    min_bound = np.min(stacked, axis=0)
    max_bound = np.max(stacked, axis=0)
    if np.linalg.norm(max_bound - min_bound) < 1e-6:
        min_bound = min_bound - 1.0
        max_bound = max_bound + 1.0
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def visualize_frame_gui(args, frame_ds, static_ds, dynamic_pcd, gt_boxes, static_in_box_pcd=None):
    del frame_ds
    try:
        from open3d.visualization import gui, rendering
    except (ImportError, RuntimeError, AttributeError) as e:
        raise RuntimeError("Open3D GUI viewer is not available in this Open3D installation.") from e

    app = gui.Application.instance
    app.initialize()
    window = app.create_window("Dynamic foreground points + GT boxes", 1280, 800)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])
    window.add_child(scene_widget)
    window.set_on_layout(lambda ctx: setattr(scene_widget, "frame", window.content_rect))

    if hasattr(scene_widget, "enable_scene_caching"):
        scene_widget.enable_scene_caching(True)
    if hasattr(scene_widget, "set_view_controls"):
        scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    geometry_count = 0
    static_background_points_added = 0
    dynamic_points_added = 0
    static_in_gt_box_points_added = 0
    if args.show_static:
        static_background = make_painted_copy(static_ds, args.static_color_rgb)
        static_material = make_rendering_material(
            "defaultUnlit",
            point_size=args.static_point_size,
            rendering_module=rendering,
        )
        scene_widget.scene.add_geometry("static_background", static_background, static_material)
        static_background_points_added = len(static_background.points)
        geometry_count += 1
    if not args.hide_dynamic:
        dynamic_for_view = dynamic_pcd
        if args.show_static and not args.color_by_height:
            dynamic_for_view = make_painted_copy(dynamic_pcd, args.dynamic_color_rgb)
        point_material = make_rendering_material(
            "defaultUnlit",
            point_size=args.dynamic_point_size,
            rendering_module=rendering,
        )
        scene_widget.scene.add_geometry("dynamic_points", dynamic_for_view, point_material)
        dynamic_points_added = len(dynamic_for_view.points)
        geometry_count += 1
    if static_in_box_pcd is not None and len(static_in_box_pcd.points) > 0:
        static_in_box_material = make_rendering_material(
            "defaultUnlit",
            point_size=args.static_in_box_point_size,
            rendering_module=rendering,
        )
        scene_widget.scene.add_geometry("static_points_inside_gt_boxes", static_in_box_pcd, static_in_box_material)
        static_in_gt_box_points_added = len(static_in_box_pcd.points)
        geometry_count += 1

    labels_added = 0
    label_examples = []
    if args.show_gt:
        mappings, focused_mapping = collect_gt_draw_mappings(gt_boxes, args)
        log_focus_status(focused_mapping, args)
        for mapping in mappings:
            should_draw, is_focus = mapping_is_drawn(mapping, focused_mapping, args)
            if not should_draw:
                continue
            color = get_gt_box_color(mapping, is_focus, args)
            box = create_o3d_box(mapping["box"], color, args.gt_box_scale, args.gt_yaw_only)
            line_width = args.gt_line_width * (2.0 if is_focus else 1.0)
            line_material = make_rendering_material(
                "unlitLine",
                color=color,
                line_width=line_width,
                rendering_module=rendering,
            )
            scene_widget.scene.add_geometry(f"gt_box_{mapping['view_index']}", box, line_material)
            geometry_count += 1
            if should_label_mapping(mapping, is_focus, args):
                label_pos = gt_label_position(mapping, args.gt_box_scale)
                label = scene_widget.add_3d_label(label_pos.astype(np.float32), str(mapping["object_id"]))
                if label is not None:
                    label.color = gui.Color(
                        float(args.gt_id_label_color_rgb[0]),
                        float(args.gt_id_label_color_rgb[1]),
                        float(args.gt_id_label_color_rgb[2]),
                        1.0,
                    )
                    label.scale = float(args.gt_id_label_scale)
                labels_added += 1
                if len(label_examples) < 10:
                    label_examples.append(
                        {
                            "object_id": mapping["object_id"],
                            "class_name": mapping["class_name"],
                            "frame_id": args.frame_id,
                            "position": label_pos,
                        }
                    )

    bounds = combined_bounds(
        dynamic_pcd,
        gt_boxes if args.show_gt else [],
        args,
        static_in_box_pcd,
        static_ds if args.show_static else None,
    )
    center = bounds.get_center()
    scene_widget.setup_camera(60.0, bounds, center)
    print(f"Open3D GUI geometries added: {geometry_count}")
    print(f"GUI show_static: {args.show_static}")
    print(f"GUI static background points added: {static_background_points_added}")
    print(f"GUI dynamic points added: {dynamic_points_added}")
    print(f"GUI static_in_gt_box points added: {static_in_gt_box_points_added}")
    log_gui_label_summary(label_examples, labels_added, len(gt_boxes), args)
    app.run()


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
    R = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), vec)
    mesh.rotate(R, center=np.array([0.0, 0.0, 0.0]))
    mesh.translate((p0 + p1) / 2.0)
    mesh.paint_uniform_color(np.clip(np.asarray(color, dtype=np.float64), 0.0, 1.0))
    mesh.compute_vertex_normals()
    return mesh


def make_digit_label_3d(label, origin, size=1.0, radius=0.05, color=(1.0, 1.0, 0.0)):
    """
    Create a bold 3D seven-segment numeric label using cylinders.
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


def get_stroke_segments(ch):
    strokes = {
        "A": [((0.0, 0.0), (0.0, 1.0)), ((0.7, 0.0), (0.7, 1.0)), ((0.0, 1.0), (0.7, 1.0)), ((0.0, 0.5), (0.7, 0.5))],
        "C": [((0.7, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.7, 0.0))],
        "D": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.6, 0.85)), ((0.6, 0.85), (0.7, 0.5)), ((0.7, 0.5), (0.6, 0.15)), ((0.6, 0.15), (0.0, 0.0))],
        "E": [((0.7, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.5), (0.6, 0.5)), ((0.0, 0.0), (0.7, 0.0))],
        "F": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.7, 1.0)), ((0.0, 0.5), (0.6, 0.5))],
        "G": [((0.7, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.7, 0.0)), ((0.7, 0.0), (0.7, 0.5)), ((0.7, 0.5), (0.4, 0.5))],
        "I": [((0.0, 1.0), (0.7, 1.0)), ((0.35, 1.0), (0.35, 0.0)), ((0.0, 0.0), (0.7, 0.0))],
        "K": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 0.5), (0.7, 1.0)), ((0.0, 0.5), (0.7, 0.0))],
        "L": [((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.7, 0.0))],
        "N": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.7, 0.0)), ((0.7, 0.0), (0.7, 1.0))],
        "O": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.7, 1.0)), ((0.7, 1.0), (0.7, 0.0)), ((0.7, 0.0), (0.0, 0.0))],
        "P": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.7, 1.0)), ((0.7, 1.0), (0.7, 0.5)), ((0.7, 0.5), (0.0, 0.5))],
        "R": [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.7, 1.0)), ((0.7, 1.0), (0.7, 0.5)), ((0.7, 0.5), (0.0, 0.5)), ((0.0, 0.5), (0.7, 0.0))],
        "S": [((0.7, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.5)), ((0.0, 0.5), (0.7, 0.5)), ((0.7, 0.5), (0.7, 0.0)), ((0.7, 0.0), (0.0, 0.0))],
        "T": [((0.0, 1.0), (0.7, 1.0)), ((0.35, 1.0), (0.35, 0.0))],
        "U": [((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.7, 0.0)), ((0.7, 0.0), (0.7, 1.0))],
        "V": [((0.0, 1.0), (0.35, 0.0)), ((0.35, 0.0), (0.7, 1.0))],
        "Y": [((0.0, 1.0), (0.35, 0.5)), ((0.7, 1.0), (0.35, 0.5)), ((0.35, 0.5), (0.35, 0.0))],
        "#": [((0.2, 1.0), (0.1, 0.0)), ((0.6, 1.0), (0.5, 0.0)), ((0.0, 0.65), (0.7, 0.65)), ((0.0, 0.35), (0.7, 0.35))],
    }
    digit_segments = {
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
    digit_points = {
        "a": ((0.0, 1.0), (0.7, 1.0)),
        "b": ((0.7, 1.0), (0.7, 0.5)),
        "c": ((0.7, 0.5), (0.7, 0.0)),
        "d": ((0.0, 0.0), (0.7, 0.0)),
        "e": ((0.0, 0.5), (0.0, 0.0)),
        "f": ((0.0, 1.0), (0.0, 0.5)),
        "g": ((0.0, 0.5), (0.7, 0.5)),
    }
    if ch in digit_segments:
        return [digit_points[seg] for seg in digit_segments[ch]]
    return strokes.get(ch, [])


def stroke_text_width(text, size):
    width = 0.0
    for ch in str(text).upper():
        width += (0.45 if ch == " " else 0.95) * size
    return max(width - 0.25 * size, 0.0)


def make_stroke_text_label_3d(text, origin, size=0.6, radius=0.03, color=(1.0, 1.0, 0.0)):
    origin = np.asarray(origin, dtype=np.float64)
    text = str(text).upper()
    merged = o3d.geometry.TriangleMesh()
    x_offset = -stroke_text_width(text, size) / 2.0

    for ch in text:
        if ch == " ":
            x_offset += 0.45 * size
            continue
        for q0, q1 in get_stroke_segments(ch):
            p0 = origin + np.array([(q0[0] + x_offset) * size, q0[1] * size, 0.0])
            p1 = origin + np.array([(q1[0] + x_offset) * size, q1[1] * size, 0.0])
            cyl = make_cylinder_segment(p0, p1, radius=radius, color=color)
            if cyl is not None:
                merged += cyl
        x_offset += 0.95 * size

    merged.compute_vertex_normals()
    return merged


def create_text_label(text, position, size, color, orientation):
    def warn_once(message, detail=None):
        if getattr(create_text_label, "_warned_unsupported", False):
            return
        print(message)
        if detail is not None:
            print(f"  text geometry error: {detail}")
        create_text_label._warned_unsupported = True

    try:
        text_mesh_creator = getattr(getattr(o3d, "t", None).geometry.TriangleMesh, "create_text", None)
    except AttributeError:
        text_mesh_creator = None

    if text_mesh_creator is None:
        warn_once("Warning: Open3D text geometry unsupported in this Open3D version.")
        return None

    create_attempts = (
        lambda: text_mesh_creator(text, depth=0.01),
        lambda: text_mesh_creator(text),
    )
    text_mesh = None
    last_error = None
    for create_attempt in create_attempts:
        try:
            text_mesh = create_attempt()
            break
        except (TypeError, RuntimeError, AttributeError) as e:
            last_error = e

    if text_mesh is None:
        warn_once("Warning: Open3D text geometry unsupported in this Open3D version.", last_error)
        return None

    try:
        if hasattr(text_mesh, "to_legacy"):
            text_mesh = text_mesh.to_legacy()
        text_mesh.compute_vertex_normals()
        text_mesh.paint_uniform_color(np.clip(np.asarray(color, dtype=np.float64), 0.0, 1.0))
        text_mesh.scale(size, center=(0.0, 0.0, 0.0))
        if orientation == "xz":
            R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2.0, 0.0, 0.0))
            text_mesh.rotate(R, center=(0.0, 0.0, 0.0))
        elif orientation == "yz":
            R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2.0, 0.0, np.pi / 2.0))
            text_mesh.rotate(R, center=(0.0, 0.0, 0.0))
        bbox = text_mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        anchor = np.asarray(
            [
                (min_bound[0] + max_bound[0]) / 2.0,
                (min_bound[1] + max_bound[1]) / 2.0,
                min_bound[2],
            ],
            dtype=np.float64,
        )
        text_mesh.translate(np.asarray(position, dtype=np.float64) - anchor)
        return text_mesh
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        warn_once("Warning: Open3D text geometry unsupported in this Open3D version.", e)
        return None


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


def parse_rgb_color(value, name="gt_label_color"):
    if isinstance(value, str):
        raw_parts = value.split(",")
    else:
        raw_parts = value
    try:
        parts = np.asarray([float(part) for part in raw_parts], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"--{name.replace('_', '-')} must be an RGB triplet like 1,1,1 or 255,255,255.") from e
    if len(parts) != 3:
        raise ValueError(f"--{name.replace('_', '-')} must have exactly three values.")
    if np.any(parts > 1.0):
        parts = parts / 255.0
    parts = np.clip(parts, 0.0, 1.0).astype(np.float64)
    print(f"{name} parsed: {parts.tolist()}")
    return parts


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


def validate_args(args):
    if args.static_voxel_size <= 0:
        raise ValueError("--static-voxel-size must be positive.")
    if args.compare_voxel_size <= 0:
        raise ValueError("--compare-voxel-size must be positive.")
    if args.distance_threshold < 0:
        raise ValueError("--distance-threshold must be non-negative.")
    if args.clip_z_min is not None and args.clip_z_max is not None and args.clip_z_max < args.clip_z_min:
        raise ValueError("--clip-z-max must be greater than or equal to --clip-z-min.")
    if args.gt_line_width <= 0:
        raise ValueError("--gt-line-width must be positive.")
    if args.gt_id_label_scale <= 0:
        raise ValueError("--gt-id-label-scale must be positive.")
    args.gt_id_label_color_rgb = parse_rgb_color(args.gt_id_label_color, "gt_id_label_color")
    if args.gt_label_size <= 0:
        raise ValueError("--gt-label-size must be positive.")
    if args.gt_label_z_offset < 0:
        raise ValueError("--gt-label-z-offset must be non-negative.")
    args.gt_label_color_rgb = parse_rgb_color(args.gt_label_color, "gt_label_color")
    if args.gt_id_size <= 0:
        raise ValueError("--gt-id-size must be positive.")
    if args.gt_id_radius <= 0:
        raise ValueError("--gt-id-radius must be positive.")
    if args.gt_id_z_offset < 0:
        raise ValueError("--gt-id-z-offset must be non-negative.")
    if args.gt_id_max_digits <= 0:
        raise ValueError("--gt-id-max-digits must be positive.")
    args.gt_id_color_rgb = parse_rgb_color(args.gt_id_color, "gt_id_color")
    if args.focus_label_size <= 0:
        raise ValueError("--focus-label-size must be positive.")
    if args.focus_label_radius <= 0:
        raise ValueError("--focus-label-radius must be positive.")
    if args.focus_label_z_offset < 0:
        raise ValueError("--focus-label-z-offset must be non-negative.")
    args.focus_bbox_color_rgb = parse_rgb_color(args.focus_bbox_color, "focus_bbox_color")
    args.focus_label_color_rgb = parse_rgb_color(args.focus_label_color, "focus_label_color")
    args.static_color_rgb = parse_rgb_color(args.static_color, "static_color")
    args.frame_color_rgb = parse_rgb_color(args.frame_color, "frame_color")
    args.dynamic_color_rgb = parse_rgb_color(args.dynamic_color, "dynamic_color")
    if args.static_point_size <= 0:
        raise ValueError("--static-point-size must be positive.")
    if args.dynamic_point_size <= 0:
        raise ValueError("--dynamic-point-size must be positive.")
    if args.static_in_box_point_size <= 0:
        raise ValueError("--static-in-box-point-size must be positive.")
    if args.static_in_box_gt_box_scale is not None and args.static_in_box_gt_box_scale <= 0:
        raise ValueError("--static-in-box-gt-box-scale must be positive when provided.")
    args.static_in_box_color_rgb = parse_rgb_color(args.static_in_box_color, "static_in_box_color")
    if args.frame_stride_dynamic <= 0:
        raise ValueError("--frame-stride-dynamic must be positive.")
    if args.gt_box_scale <= 0:
        raise ValueError("--gt-box-scale must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")
    if args.height_min is not None and args.height_max is not None and args.height_max <= args.height_min:
        raise ValueError("--height-max must be greater than --height-min when both are provided.")
    if args.show_gt_labels or args.show_gt_ids or args.show_gt_id_labels or args.show_static_in_gt_boxes:
        args.show_gt = True


def maybe_load_gt(args, scene_name, frame_id, fail_on_missing):
    if not (args.show_gt or args.save_gt_overlay or args.save_preview_png or args.show_static_in_gt_boxes):
        return []
    gt_path, gt_boxes = load_gt_boxes_for_frame(args, scene_name, frame_id)
    print(f"gt_path: {gt_path}")
    if gt_path is None or not os.path.exists(gt_path):
        message = f"GT file does not exist: {gt_path}"
        if fail_on_missing:
            raise FileNotFoundError(message)
        print(f"Warning: {message}")
        return []
    print_gt_summary(gt_path, gt_boxes)
    print(f"gt_line_width: {args.gt_line_width}")
    return gt_boxes


def visualize_frame(args, frame_ds, static_ds, dynamic_pcd, gt_boxes, static_in_box_pcd=None):
    frame_ds, static_ds, dynamic_pcd, static_in_box_pcd = prepare_visualization_clouds(
        args,
        frame_ds,
        static_ds,
        dynamic_pcd,
        static_in_box_pcd,
    )

    if args.viewer == "gui":
        visualize_frame_gui(args, frame_ds, static_ds, dynamic_pcd, gt_boxes, static_in_box_pcd)
        return
    if args.show_gt_id_labels:
        print("Warning: --show-gt-id-labels requires --viewer gui; classic viewer will draw boxes only.")

    geometries = []
    if args.show_static:
        frame_colored = make_painted_copy(frame_ds, args.frame_color_rgb)
        static_colored = make_painted_copy(static_ds, args.static_color_rgb)
        geometries.extend([frame_colored, static_colored])
        if not args.hide_dynamic:
            if args.color_by_height:
                geometries.append(dynamic_pcd)
            else:
                geometries.append(make_painted_copy(dynamic_pcd, args.dynamic_color_rgb))
    else:
        if not args.hide_dynamic:
            geometries.append(dynamic_pcd)
    if static_in_box_pcd is not None and len(static_in_box_pcd.points) > 0:
        geometries.append(static_in_box_pcd)

    if args.show_gt:
        id_mesh_labels_added = 0
        id_mesh_labels_requested = 0
        id_label_examples = []
        mappings = get_gt_view_mappings(gt_boxes)
        focused_mappings = [mapping for mapping in mappings if is_focused_mapping(mapping, args)]
        focused_mapping = focused_mappings[0] if focused_mappings else None
        if args.focus_object_id is not None or args.focus_class_name is not None:
            print(
                f"Focused object requested: class={args.focus_class_name} "
                f"object_id={args.focus_object_id}"
            )
            if focused_mapping is None:
                print("Focused object not found.")
            else:
                focus_text = focused_label_text(focused_mapping, args)
                print(
                    f"Focused object found: {focus_text} "
                    f"center={focused_mapping['center'].tolist()}"
                )
                print(f"Focused object extent: {focused_mapping['extent'].tolist()}")
                print(f"Focused label text: {focus_text}")
        mesh_enabled = args.show_gt_ids and args.show_gt_id_mesh and not args.hide_gt_id_mesh
        if args.show_gt_labels or args.show_gt_ids:
            print("GT view index mapping:")
        for mapping in mappings:
            is_focus = focused_mapping is not None and mapping["view_index"] == focused_mapping["view_index"]
            if args.show_only_focused_gt and not is_focus:
                continue
            box = mapping["box"]
            if is_focus:
                color = args.focus_bbox_color_rgb
            else:
                color = get_class_color(box["label"]) if args.gt_color_by_class else [1.0, 0.0, 0.0]
            geometries.append(create_o3d_box(box, color, args.gt_box_scale, args.gt_yaw_only))
            if args.show_gt_labels or args.show_gt_ids:
                print(
                    f"  [{mapping['view_index']}] {mapping['class_name']} "
                    f"#{mapping['object_id']} center={mapping['center'].tolist()}"
                )
            if mesh_enabled:
                mesh_label = gt_mesh_display_label(mapping, args)
                if mesh_label is None:
                    continue
                id_mesh_labels_requested += 1
                center = mapping["center"]
                extent = mapping["extent"] * args.gt_box_scale
                label_origin = center + np.asarray(
                    [0.0, 0.0, extent[2] / 2.0 + args.gt_id_z_offset],
                    dtype=np.float64,
                )
                if len(id_label_examples) < 5:
                    id_label_examples.append(
                        (
                            mapping["view_index"],
                            mapping["object_id"],
                            mapping["class_name"],
                            mesh_label,
                            label_origin.copy(),
                        )
                    )
                id_mesh = make_digit_label_3d(
                    mesh_label,
                    label_origin,
                    size=args.gt_id_size,
                    radius=args.gt_id_radius,
                    color=args.gt_id_color_rgb,
                )
                if len(id_mesh.vertices) > 0:
                    geometries.append(id_mesh)
                    id_mesh_labels_added += 1

        if mesh_enabled:
            print("GT digit ID label examples:")
            for view_index, object_id, class_name, mesh_label, label_origin in id_label_examples:
                print(
                    f"  view_index={view_index} mesh_label={mesh_label} object_id={object_id} "
                    f"class={class_name} label_origin={label_origin.tolist()} size={args.gt_id_size}"
                )
            print(f"Open3D digit ID labels added: {id_mesh_labels_added}/{id_mesh_labels_requested}")

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


def gt_overlay_dict(frame_id, gt_boxes, args=None, static_in_box_stats=None):
    boxes = []
    focused_object = None
    static_box_counts = {}
    if static_in_box_stats is not None:
        static_box_counts = static_in_box_stats.get("box_counts", {})
    for mapping in get_gt_view_mappings(gt_boxes):
        if args is not None and focused_object is None and is_focused_mapping(mapping, args):
            focused_object = focused_object_dict(mapping, focused_label_text(mapping, args))
        box = mapping["box"]
        boxes.append(
            {
                "view_index": mapping["view_index"],
                "class_label": mapping["class_label"],
                "class_name": mapping["class_name"],
                "object_id": mapping["object_id"],
                "center": mapping["center"].tolist(),
                "extent": mapping["extent"].tolist(),
                "size": mapping["extent"].tolist(),
                "rotation": mapping["rotation"].tolist(),
                "text": make_gt_text(mapping["class_name"], mapping["object_id"]),
                "static_points_inside_box": int(static_box_counts.get(mapping["view_index"], 0)),
            }
        )
    result = {"frame_id": int(frame_id), "boxes": boxes}
    if static_in_box_stats is not None:
        result["static_points_inside_any_gt_box"] = int(static_in_box_stats.get("total_points", 0))
        result["gt_boxes_with_static_points"] = int(static_in_box_stats.get("boxes_with_points", 0))
        default_box_scale = static_in_box_scale(args) if args is not None else None
        box_scale = static_in_box_stats.get("box_scale", default_box_scale)
        if box_scale is not None:
            result["static_in_box_gt_box_scale"] = float(box_scale)
    if focused_object is not None:
        result["focused_object"] = focused_object
    return result


def save_gt_overlay_json(path, frame_id, gt_boxes, args=None, static_in_box_stats=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt_overlay_dict(frame_id, gt_boxes, args, static_in_box_stats), f, indent=2)
        f.write("\n")


def get_box_xy_corners(box, scale, yaw_only):
    rotation = get_box_rotation(box["rotation"], yaw_only)
    extent = np.asarray(box["extent"], dtype=np.float64) * scale
    center = np.asarray(box["center"], dtype=np.float64)
    half_x = extent[0] / 2.0
    half_y = extent[1] / 2.0
    local_corners = np.asarray(
        [
            [-half_x, -half_y, 0.0],
            [half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, half_y, 0.0],
        ],
        dtype=np.float64,
    )
    corners = center + local_corners @ rotation.T
    return corners[:, :2]


def save_preview_png(path, dynamic_pcd, gt_boxes, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    points = np.asarray(dynamic_pcd.points)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    if len(points) > 0:
        sample = points
        if len(points) > 200000:
            indices = np.linspace(0, len(points) - 1, 200000).astype(np.int64)
            sample = points[indices]
        ax.scatter(sample[:, 0], sample[:, 1], s=0.2, c="0.25", alpha=0.6, linewidths=0)

    for box in gt_boxes:
        label = int(box["label"])
        color = get_class_color(label) if args.gt_color_by_class else [1.0, 0.0, 0.0]
        corners = get_box_xy_corners(box, args.gt_box_scale, args.gt_yaw_only)
        polygon = Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(polygon)
        class_name = CLASS_NAMES.get(label, f"unknown_{label}")
        text = make_gt_text(
            class_name,
            box["object_id"],
            show_labels=args.show_gt_labels or not args.show_gt_ids,
            show_ids=args.show_gt_ids or not args.show_gt_labels,
        )
        center = np.asarray(box["center"], dtype=np.float64)
        ax.text(center[0], center[1], text, color=color, fontsize=7, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dynamic points with GT boxes, top view")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def process_dynamic_frame(args, scene_name, frame_id, static_ds, view_enabled=False):
    frame_path = build_frame_path(args.pcd_root, scene_name, frame_id)
    static_path = build_static_path(
        args.pcd_root,
        scene_name,
        args.static_voxel_size,
        args.background_root,
    )
    output_path = build_output_path(args, scene_name, frame_id)

    print(f"\nFrame: {scene_name}/{frame_id:05d}")
    print(f"loaded frame PLY path: {frame_path}")
    print(f"loaded static path: {static_path}")
    if args.save:
        print(f"output path: {output_path}")
    if args.save and args.skip_existing and os.path.exists(output_path):
        print("Skipping existing dynamic output.")
        return {"status": "skipped_existing", "dynamic_ratio": None}

    frame = load_point_cloud(frame_path, "frame")
    frame_points_before = len(frame.points)
    frame_ds = frame.voxel_down_sample(args.compare_voxel_size)

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
        args.paint and not args.color_by_height,
        args.dynamic_color_rgb,
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
        print(f"saved dynamic path: {output_path}")

    gt_boxes = maybe_load_gt(
        args,
        scene_name,
        frame_id,
        fail_on_missing=view_enabled or args.show_static_in_gt_boxes,
    )
    print(f"GT boxes count: {len(gt_boxes)}")
    static_in_box_pcd = None
    static_in_box_stats = None
    if args.show_static_in_gt_boxes:
        static_in_box_pcd, static_in_box_stats = find_static_points_inside_gt_boxes(static_ds, gt_boxes, args)
        if args.save:
            static_inside_gt_path = build_static_inside_gt_path(args, scene_name, frame_id)
            os.makedirs(os.path.dirname(static_inside_gt_path), exist_ok=True)
            write_ok = o3d.io.write_point_cloud(static_inside_gt_path, static_in_box_pcd, format="ply")
            if not write_ok:
                raise RuntimeError(f"Failed to write static-inside-GT PLY: {static_inside_gt_path}")
            print(f"saved static-inside-GT path: {static_inside_gt_path}")
    focus_requested = args.focus_object_id is not None or args.focus_class_name is not None
    overlay_enabled = (
        args.save_gt_overlay
        or focus_requested
        or args.show_static_in_gt_boxes
        or (args.show_gt and (args.show_gt_labels or args.show_gt_ids or args.show_gt_id_labels))
    )
    if overlay_enabled:
        overlay_path = build_overlay_json_path(args, scene_name, frame_id)
        save_gt_overlay_json(overlay_path, frame_id, gt_boxes, args, static_in_box_stats)
        print(f"saved overlay JSON path: {overlay_path}")
    if args.save_preview_png:
        preview_path = build_preview_png_path(args, scene_name, frame_id)
        save_preview_png(preview_path, dynamic_pcd, gt_boxes, args)
        print(f"saved preview PNG path: {preview_path}")
    if view_enabled:
        visualize_frame(args, frame_ds, static_ds, dynamic_pcd, gt_boxes, static_in_box_pcd)

    return {"status": "processed", "dynamic_ratio": dynamic_ratio}


def process_scene(args, scene_name, view_enabled=False):
    start_time = time.time()
    static_path = build_static_path(
        args.pcd_root,
        scene_name,
        args.static_voxel_size,
        args.background_root,
    )
    frame_ids = get_scene_frame_ids(
        args.pcd_root,
        scene_name,
        args.all_frames,
        args.frame_id,
        args.frame_stride_dynamic,
        args.max_frames,
    )

    print(f"\nScene: {args.split}/{scene_name}")
    print(f"  total_frames: {len(frame_ids)}")
    print(f"  static path: {static_path}")

    summary = {
        "total_frames": len(frame_ids),
        "processed_frames": 0,
        "skipped_existing": 0,
        "failed_frames": 0,
        "dynamic_ratios": [],
    }

    static = load_point_cloud(static_path, "static")
    static_ds = static.voxel_down_sample(args.compare_voxel_size)

    for frame_id in frame_ids:
        try:
            result = process_dynamic_frame(args, scene_name, frame_id, static_ds, view_enabled=view_enabled)
        except Exception as e:
            summary["failed_frames"] += 1
            print(f"Warning: failed frame {scene_name}/{frame_id:05d}: {e}")
            continue

        if result["status"] == "skipped_existing":
            summary["skipped_existing"] += 1
        else:
            summary["processed_frames"] += 1
            if result["dynamic_ratio"] is not None:
                summary["dynamic_ratios"].append(result["dynamic_ratio"])

    elapsed_seconds = time.time() - start_time
    if summary["dynamic_ratios"]:
        average_dynamic_ratio = sum(summary["dynamic_ratios"]) / len(summary["dynamic_ratios"])
    else:
        average_dynamic_ratio = 0.0

    print(f"\nSummary for {scene_name}")
    print(f"  total_frames: {summary['total_frames']}")
    print(f"  processed_frames: {summary['processed_frames']}")
    print(f"  skipped_existing: {summary['skipped_existing']}")
    print(f"  failed_frames: {summary['failed_frames']}")
    print(f"  average_dynamic_ratio: {average_dynamic_ratio:.6f}")
    print(f"  elapsed_seconds: {elapsed_seconds:.2f}")
    return summary


def main():
    args = parse_args()
    resolve_roots(args)
    validate_args(args)

    scene_dirs = get_scene_dirs(args.pcd_root, args.scene_name, args.all_scenes)
    if not scene_dirs:
        raise RuntimeError(f"No scenes found under {args.pcd_root}")
    if not args.all_frames and args.frame_id is None:
        raise ValueError("--frame-id is required unless --all-frames is passed.")

    batch_mode = args.all_frames or args.all_scenes or args.scene_name is None or len(scene_dirs) > 1
    view_enabled = not args.no_view and not batch_mode

    print("Dynamic PCD generation/view")
    print(f"  processing_root: {args.processing_root}")
    print(f"  data_root: {args.data_root}")
    print(f"  pcd_root: {args.pcd_root}")
    print(f"  background_root: {args.background_root}")
    print(f"  object_root: {args.object_root}")
    print(f"  split: {args.split}")
    print(f"  selected scenes: {[os.path.basename(path) for path in scene_dirs]}")
    print(f"  all_frames: {args.all_frames}")
    print(f"  frame_stride_dynamic: {args.frame_stride_dynamic}")
    print(f"  max_frames: {args.max_frames}")
    print(f"  save: {args.save}")
    print(f"  skip_existing: {args.skip_existing}")
    print(f"  view_enabled: {view_enabled}")
    print(f"  viewer mode: {args.viewer}")

    load_open3d()

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        try:
            process_scene(args, scene_name, view_enabled=view_enabled)
        except Exception as e:
            print(f"Error processing scene {scene_name}: {e}")


if __name__ == "__main__":
    main()
