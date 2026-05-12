import argparse
import concurrent.futures
import json
import math
import os
import time
from collections import defaultdict

import numpy as np

o3d = None

CLASS_NAMES = [
    "Person",
    "Forklift",
    "NovaCarter",
    "Transporter",
    "FourierGR1T2",
    "AgilityDigit",
    "PalletTruck",
]
CLASS_TO_LABEL = {name: label for label, name in enumerate(CLASS_NAMES)}
CLASS_TO_LABEL_LOWER = {name.lower(): label for name, label in CLASS_TO_LABEL.items()}


def load_open3d():
    global o3d
    if o3d is None:
        import open3d as open3d_module
        o3d = open3d_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one static/background PLY per scene from per-frame point clouds."
    )
    parser.add_argument("--processing-root", default=None,
                        help="Optional processing root. Uses <processing-root>/point_cloud as input by default.")
    parser.add_argument("--data-root", default=None,
                        help="Optional raw dataset root with videos, depth_maps, and calibration.json.")
    parser.add_argument("--pcd-root", default=None,
                        help="Root point cloud folder.")
    parser.add_argument("--background-root", default=None,
                        help="Optional output root for static/background point clouds.")
    parser.add_argument("--source", choices=["pcd", "raw", "auto"], default="auto",
                        help="Source for sampled frames. 'auto' uses existing PLY and generates missing sampled PLYs from raw data when possible.")
    parser.add_argument("--ensure-sampled-pcd", action="store_true",
                        help="Ensure sampled PCD files exist by generating missing/invalid sampled frames from raw data when possible.")
    parser.add_argument("--save-generated-pcd", action="store_true", default=True,
                        help="Save raw-generated sampled PCD files to <pcd-root>/<scene>/pcd.")
    parser.add_argument("--merge-voxel-size", type=float, default=0.05,
                        help="Voxel size used when generating missing sampled PCD files from raw RGB/depth.")
    parser.add_argument("--max-cameras", type=int, default=None,
                        help="Optional maximum number of cameras to use when generating sampled PCD files from raw data.")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Raw data/GT splits to use for selected scenes.")
    parser.add_argument("--scene-name", action="append", default=None,
                        help="Optional scene name filter. Can be passed multiple times.")
    parser.add_argument("--all-scenes", action="store_true",
                        help="Process all Warehouse_* scenes under <pcd-root>.")
    parser.add_argument("--frame-stride-static", type=int, default=100,
                        help="Sample every N frames for static point cloud generation.")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="Voxel size in meters.")
    parser.add_argument("--presence-ratio", type=float, default=0.5,
                        help="Minimum sampled-frame presence ratio for a voxel to be static.")
    parser.add_argument("--min-ply-size-mb", type=float, default=0.1,
                        help="Minimum valid PLY size in MB.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing static PLY and summary files.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenes whose static PLY and summary JSON already exist.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit on sampled frames for quick tests.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of scene worker processes. 1 disables multiprocessing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected frames and output paths without reading or writing PLY files.")
    parser.add_argument("--exclude-labeled-objects", action="store_true",
                        help="Remove GT-labeled object points before accumulating static voxels.")
    parser.add_argument("--exclude-gt-labels", nargs="*", type=int, default=[],
                        help="Optional GT class labels to exclude. If set, only these labels are removed.")
    parser.add_argument("--exclude-gt-classes", nargs="*", default=[],
                        help="Optional GT class names to exclude. If set, only these classes are removed.")
    parser.add_argument("--gt-box-scale", type=float, default=1.05,
                        help="Scale factor applied to GT box extents before point removal.")
    parser.add_argument("--gt-source", choices=["auto", "json", "txt"], default="auto",
                        help="GT source for labeled-object exclusion. Auto prefers ground_truth.json then TXT.")
    parser.add_argument("--gt-yaw-only", action="store_true",
                        help="Use only yaw (rz) from GT box rotations when removing points.")
    return parser.parse_args()


def is_valid_ply(path, min_size_mb):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < min_size_mb * 1024 * 1024:
        return False
    try:
        with open(path, "rb") as f:
            first_line = f.readline().rstrip(b"\r\n")
    except OSError:
        return False
    return first_line == b"ply"


def resolve_roots(args):
    if args.pcd_root is None:
        if args.processing_root is None:
            raise ValueError("Either --pcd-root or --processing-root must be provided.")
        args.pcd_root = os.path.join(args.processing_root, "point_cloud")
    if args.background_root is None and args.processing_root is not None:
        args.background_root = os.path.join(args.processing_root, "point_cloud_background")


def get_scene_dirs(pcd_root, scene_names, all_scenes=False):
    if scene_names and not all_scenes:
        return [
            os.path.join(pcd_root, scene)
            for scene in scene_names
        ]
    if not os.path.isdir(pcd_root):
        return []
    return [
        os.path.join(pcd_root, name)
        for name in sorted(os.listdir(pcd_root))
        if name.startswith("Warehouse_") and os.path.isdir(os.path.join(pcd_root, name))
    ]


def parse_frame_id(path, scene):
    filename = os.path.basename(path)
    prefix = f"{scene}_"
    suffix = ".ply"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Invalid PLY filename for scene {scene}: {filename}")
    return int(filename[len(prefix):-len(suffix)])


def voxel_key(points, voxel_size):
    return np.floor(points / voxel_size).astype(np.int64)


def build_output_paths(scene_dir, scene, voxel_size, background_root=None):
    voxel_mm = int(voxel_size * 1000)
    base_name = f"{scene}_static_voxel{voxel_mm:03d}"
    if background_root is not None:
        static_dir = os.path.join(background_root, scene)
    else:
        static_dir = os.path.join(scene_dir, "static")
    return (
        os.path.join(static_dir, f"{base_name}.ply"),
        os.path.join(static_dir, f"{base_name}_summary.json"),
    )


def build_pcd_path(pcd_root, scene, frame_id):
    return os.path.join(
        pcd_root,
        scene,
        "pcd",
        f"{scene}_{frame_id:05d}.ply",
    )


def build_gt_txt_path(pcd_root, scene, frame_id):
    return os.path.join(
        pcd_root,
        scene,
        "gt",
        f"{scene}_{frame_id:05d}.txt",
    )


def should_exclude_gt(args):
    return (
        args.exclude_labeled_objects
        or bool(args.exclude_gt_labels)
        or bool(args.exclude_gt_classes)
    )


def get_excluded_gt_labels(args):
    labels = set(args.exclude_gt_labels)
    for class_name in args.exclude_gt_classes:
        label = CLASS_TO_LABEL_LOWER.get(class_name.lower())
        if label is None:
            raise ValueError(
                f"Unknown GT class '{class_name}'. Valid classes: {', '.join(CLASS_NAMES)}"
            )
        labels.add(label)
    if labels:
        return labels
    if args.exclude_labeled_objects:
        return set(range(len(CLASS_NAMES)))
    return set()


def make_gt_box(label, object_id, center, extent, rotation):
    return {
        "label": int(label),
        "class_name": CLASS_NAMES[int(label)] if 0 <= int(label) < len(CLASS_NAMES) else str(label),
        "object_id": object_id,
        "center": np.asarray(center, dtype=np.float64),
        "extent": np.asarray(extent, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
    }


def load_gt_boxes_from_json(args, split, scene, frame_id):
    if args.data_root is None:
        return None

    gt_path = os.path.join(args.data_root, split, scene, "ground_truth.json")
    if not os.path.exists(gt_path):
        return None

    cache = getattr(args, "_gt_json_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_gt_json_cache", cache)

    if gt_path not in cache:
        with open(gt_path, "r") as f:
            cache[gt_path] = json.load(f)

    frame_key = str(frame_id)
    if frame_key not in cache[gt_path]:
        return None

    frame_objects = cache[gt_path].get(frame_key, [])
    if not isinstance(frame_objects, list):
        return []

    boxes = []
    for obj in frame_objects:
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("object type")
        if class_name not in CLASS_TO_LABEL:
            continue
        try:
            boxes.append(
                make_gt_box(
                    CLASS_TO_LABEL[class_name],
                    obj.get("object id", "unknown"),
                    obj["3d location"],
                    obj["3d bounding box scale"],
                    obj["3d bounding box rotation"],
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return boxes


def load_gt_boxes_from_txt(args, split, scene, frame_id):
    gt_path = build_gt_txt_path(args.pcd_root, scene, frame_id)
    if not os.path.exists(gt_path):
        return None

    boxes = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 11:
                continue
            try:
                label = int(float(parts[0]))
                object_id = int(parts[1])
            except ValueError:
                try:
                    label = int(float(parts[0]))
                except ValueError:
                    continue
                object_id = parts[1]
            try:
                values = [float(value) for value in parts[2:]]
            except ValueError:
                continue
            boxes.append(make_gt_box(label, object_id, values[0:3], values[3:6], values[6:9]))
    return boxes


def load_gt_boxes_for_frame(args, split, scene, frame_id):
    if args.gt_source in {"auto", "json"}:
        boxes = load_gt_boxes_from_json(args, split, scene, frame_id)
        if boxes is not None or args.gt_source == "json":
            return boxes or []

    boxes = load_gt_boxes_from_txt(args, split, scene, frame_id)
    if boxes is not None:
        return boxes
    return []


def remove_points_inside_gt_boxes(pcd, boxes, gt_box_scale, yaw_only):
    if not boxes or len(pcd.points) == 0:
        return pcd, 0

    point_count = len(pcd.points)
    keep_mask = np.ones(point_count, dtype=bool)
    for box in boxes:
        center = np.asarray(box["center"], dtype=np.float64)
        extent = np.asarray(box["extent"], dtype=np.float64) * gt_box_scale
        rotation = np.asarray(box["rotation"], dtype=np.float64)
        if yaw_only:
            R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, rotation[2]))
        else:
            R = o3d.geometry.get_rotation_matrix_from_xyz(tuple(rotation.tolist()))
        obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
        indices = obb.get_point_indices_within_bounding_box(pcd.points)
        if indices:
            keep_mask[np.asarray(indices, dtype=np.int64)] = False

    excluded_points = int(point_count - np.count_nonzero(keep_mask))
    if excluded_points == 0:
        return pcd, 0

    kept_indices = np.flatnonzero(keep_mask).astype(np.int64).tolist()
    filtered_pcd = pcd.select_by_index(kept_indices)
    return filtered_pcd, excluded_points


def get_scene_ply_files(scene_dir, scene):
    pcd_dir = os.path.join(scene_dir, "pcd")
    if not os.path.isdir(pcd_dir):
        return []

    ply_files = []
    for name in os.listdir(pcd_dir):
        if not name.endswith(".ply"):
            continue
        path = os.path.join(pcd_dir, name)
        try:
            frame_id = parse_frame_id(path, scene)
        except ValueError:
            continue
        ply_files.append((frame_id, path))
    ply_files.sort(key=lambda item: item[0])
    return ply_files


def select_sampled_frame_ids(frame_ids, frame_stride_static, max_frames):
    sampled = [
        frame_id
        for frame_id in sorted(frame_ids)
        if frame_id % frame_stride_static == 0
    ]
    if max_frames is not None:
        sampled = sampled[:max_frames]
    return sampled


def get_selected_static_frame_ids(frame_stride_static, max_frames, max_total_frames=9000):
    frame_ids = list(range(0, max_total_frames, frame_stride_static))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def can_use_raw_source(args):
    return args.source in {"raw", "auto"} and args.data_root is not None


def get_raw_scene_dir(args, split, scene):
    if args.data_root is None:
        return None
    return os.path.join(args.data_root, split, scene)


def get_raw_frame_ids(args, split, scene):
    raw_scene_dir = get_raw_scene_dir(args, split, scene)
    if raw_scene_dir is None or not os.path.isdir(raw_scene_dir):
        return []

    videos_dir = os.path.join(raw_scene_dir, "videos")
    if not os.path.isdir(videos_dir):
        return []

    import cv2

    video_names = sorted(name for name in os.listdir(videos_dir) if name.endswith(".mp4"))
    if args.max_cameras is not None:
        video_names = video_names[:args.max_cameras]
    frame_counts = []
    for video_name in video_names:
        video_path = os.path.join(videos_dir, video_name)
        capture = cv2.VideoCapture(video_path)
        try:
            if capture.isOpened():
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    frame_counts.append(frame_count)
        finally:
            capture.release()
    if not frame_counts:
        return []
    return list(range(min(frame_counts)))


def select_sampled_frames(args, split, scene, ply_files):
    if args.source in {"raw", "auto"} and args.ensure_sampled_pcd:
        raw_scene_dir = get_raw_scene_dir(args, split, scene)
        if args.data_root is None:
            raise ValueError("--data-root is required with --source raw/auto and --ensure-sampled-pcd.")
        if raw_scene_dir is None or not os.path.isdir(raw_scene_dir):
            raise FileNotFoundError(f"{raw_scene_dir} does not exist.")
        return get_selected_static_frame_ids(
            args.frame_stride_static,
            args.max_frames,
            max_total_frames=9000,
        )

    if can_use_raw_source(args):
        raw_frame_ids = get_raw_frame_ids(args, split, scene)
        if raw_frame_ids:
            return select_sampled_frame_ids(raw_frame_ids, args.frame_stride_static, args.max_frames)
        if args.source == "raw":
            return []

    existing_frame_ids = [frame_id for frame_id, _ in ply_files]
    return select_sampled_frame_ids(existing_frame_ids, args.frame_stride_static, args.max_frames)


def import_pcd_generation_functions():
    try:
        from generate_pcd_data import (
            get_calibration_per_camera,
            get_point_cloud,
            normalize_camera_name,
        )
    except ImportError:
        from tools.generate_pcd_data import (
            get_calibration_per_camera,
            get_point_cloud,
            normalize_camera_name,
        )
    return normalize_camera_name, get_calibration_per_camera, get_point_cloud


def write_point_cloud_atomic(output_path, pcd, min_ply_size_mb):
    tmp_output_path = output_path.replace(".ply", ".tmp.ply")
    try:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
        write_ok = o3d.io.write_point_cloud(tmp_output_path, pcd, format="ply")
        if write_ok and is_valid_ply(tmp_output_path, min_ply_size_mb):
            os.replace(tmp_output_path, output_path)
            return True, None
        message = f"Temporary PLY failed validation: {tmp_output_path}"
    except Exception as e:
        message = str(e)

    try:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
    except OSError:
        pass
    return False, message


def generate_pcd_from_raw_frame(args, split, scene, frame_id, output_path):
    if args.data_root is None:
        raise ValueError("--data-root is required to generate missing sampled PLY files from raw data.")

    import cv2
    import h5py

    normalize_camera_name, get_calibration_per_camera, get_point_cloud = import_pcd_generation_functions()

    raw_scene_dir = os.path.join(args.data_root, split, scene)
    calibration_path = os.path.join(raw_scene_dir, "calibration.json")
    videos_dir = os.path.join(raw_scene_dir, "videos")
    depth_dir = os.path.join(raw_scene_dir, "depth_maps")

    camera_params = get_calibration_per_camera(calibration_path)
    camera_file_names = sorted(name.split(".")[0] for name in os.listdir(videos_dir) if name.endswith(".mp4"))
    if args.max_cameras is not None:
        camera_file_names = camera_file_names[:args.max_cameras]

    rgb_image_per_camera = {}
    depth_image_per_camera = {}
    depth_files = {}
    video_captures = {}

    try:
        for camera_file_name in camera_file_names:
            camera_name = normalize_camera_name(camera_file_name)
            video_path = os.path.join(videos_dir, f"{camera_file_name}.mp4")
            depth_path = os.path.join(depth_dir, f"{camera_file_name}.h5")

            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                print(f"  Warning: cannot open video file: {video_path}")
                continue
            video_captures[camera_name] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = capture.read()
            if not ret:
                print(f"  Warning: frame {frame_id:05d} unavailable for {camera_name}")
                continue
            rgb_image_per_camera[camera_name] = frame

            depth_image = None
            if os.path.exists(depth_path):
                depth_file = h5py.File(depth_path, "r")
                depth_files[camera_name] = depth_file
                key = f"distance_to_image_plane_{frame_id:05d}.png"
                if key in depth_file:
                    depth_image = depth_file[key][:]
                else:
                    print(f"  Warning: missing depth key {key} in {depth_path}")
            else:
                print(f"  Warning: depth map file not found: {depth_path}")
            depth_image_per_camera[camera_name] = depth_image

        if not rgb_image_per_camera:
            raise RuntimeError(f"No RGB frames were available for {scene} frame {frame_id:05d}.")

        pcd = get_point_cloud(
            rgb_image_per_camera,
            depth_image_per_camera,
            camera_params,
            voxel_size=args.merge_voxel_size,
        )
        if not args.save_generated_pcd:
            return pcd

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_ok, write_error = write_point_cloud_atomic(output_path, pcd, args.min_ply_size_mb)
        if not write_ok:
            detail = write_error if write_error else "write failed"
            raise RuntimeError(f"Failed to write generated PLY {output_path}: {detail}")
        return pcd
    finally:
        for capture in video_captures.values():
            capture.release()
        for depth_file in depth_files.values():
            depth_file.close()


def accumulate_point_cloud(
    pcd,
    voxel_size,
    voxel_presence_count,
    voxel_xyz_sum,
    voxel_rgb_sum,
    voxel_point_count,
):
    points = np.asarray(pcd.points)
    if points.size == 0:
        return 0, 0

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.zeros_like(points)

    keys = voxel_key(points, voxel_size)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    xyz_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    rgb_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    point_counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.int64)
    np.add.at(xyz_sums, inverse, points)
    np.add.at(rgb_sums, inverse, colors)

    for idx, key_array in enumerate(unique_keys):
        key = tuple(int(value) for value in key_array)
        voxel_presence_count[key] += 1
        voxel_xyz_sum[key] += xyz_sums[idx]
        voxel_rgb_sum[key] += rgb_sums[idx]
        voxel_point_count[key] += int(point_counts[idx])

    return len(points), len(unique_keys)


def accumulate_frame(
    ply_path,
    voxel_size,
    voxel_presence_count,
    voxel_xyz_sum,
    voxel_rgb_sum,
    voxel_point_count,
):
    pcd = o3d.io.read_point_cloud(ply_path)
    return accumulate_point_cloud(
        pcd,
        voxel_size,
        voxel_presence_count,
        voxel_xyz_sum,
        voxel_rgb_sum,
        voxel_point_count,
    )


def build_static_point_cloud(
    voxel_presence_count,
    voxel_xyz_sum,
    voxel_rgb_sum,
    voxel_point_count,
    min_presence_frames,
):
    static_points = []
    static_colors = []

    for key, presence_count in voxel_presence_count.items():
        if presence_count < min_presence_frames:
            continue
        point_count = voxel_point_count[key]
        if point_count <= 0:
            continue
        static_points.append(voxel_xyz_sum[key] / point_count)
        static_colors.append(voxel_rgb_sum[key] / point_count)

    pcd = o3d.geometry.PointCloud()
    if static_points:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(static_points, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(static_colors, dtype=np.float64))
    return pcd


def write_summary(summary_path, summary):
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def scene_result(scene, split, status, output_path, static_points, elapsed_seconds, message=None):
    result = {
        "scene": scene,
        "split": split,
        "status": status,
        "output_path": output_path,
        "static_points": static_points,
        "elapsed_seconds": elapsed_seconds,
    }
    if message:
        result["message"] = message
    return result


def process_scene(args, split, scene_dir):
    scene = os.path.basename(scene_dir)
    start_time = time.time()
    pcd_dir = os.path.join(scene_dir, "pcd")
    output_path, summary_path = build_output_paths(
        scene_dir,
        scene,
        args.voxel_size,
        args.background_root,
    )
    if not args.dry_run:
        load_open3d()

    ply_files = get_scene_ply_files(scene_dir, scene)
    total_ply_files_before = len(ply_files)
    sampled_frame_ids = select_sampled_frames(args, split, scene, ply_files)

    print(f"\nScene: {split}/{scene}")
    print(f"  pcd_dir: {pcd_dir}")
    print(f"  total PLY files before: {total_ply_files_before}")
    print(f"  selected sampled frames count: {len(sampled_frame_ids)}")
    print(f"  first 10 selected frame ids: {sampled_frame_ids[:10]}")
    print(f"  output path: {output_path}")

    if args.dry_run:
        print("  dry-run selected frames:")
        for frame_id in sampled_frame_ids:
            print(f"    {frame_id:05d}: {build_pcd_path(args.pcd_root, scene, frame_id)}")
        return scene_result(scene, split, "processed", output_path, 0, time.time() - start_time)

    if args.skip_existing and os.path.exists(output_path) and os.path.exists(summary_path):
        print("  Skipping existing static PLY and summary JSON.")
        return scene_result(scene, split, "skipped", output_path, None, time.time() - start_time)

    if os.path.exists(output_path) and not args.overwrite:
        print("  Skipping existing output. Use --overwrite to regenerate.")
        return scene_result(scene, split, "skipped", output_path, None, time.time() - start_time)

    voxel_presence_count = defaultdict(int)
    voxel_xyz_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    voxel_rgb_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    voxel_point_count = defaultdict(int)

    exclude_gt = should_exclude_gt(args)
    excluded_label_set = get_excluded_gt_labels(args)
    excluded_gt_boxes_total = 0
    excluded_points_total = 0
    remaining_points_total = 0

    valid_sampled_files = 0
    invalid_sampled_files = 0
    generated_missing_pcd_files = 0
    skipped_invalid_files = 0
    existing_valid_sampled_ply_count = 0

    for sampled_idx, frame_id in enumerate(sampled_frame_ids, start=1):
        ply_path = build_pcd_path(args.pcd_root, scene, frame_id)
        generated_pcd = None
        if is_valid_ply(ply_path, args.min_ply_size_mb):
            existing_valid_sampled_ply_count += 1
        else:
            if can_use_raw_source(args):
                print(f"  Generating missing sampled PLY from raw: {frame_id:05d}")
                try:
                    generated_pcd = generate_pcd_from_raw_frame(args, split, scene, frame_id, ply_path)
                    generated_missing_pcd_files += 1
                except Exception as e:
                    invalid_sampled_files += 1
                    skipped_invalid_files += 1
                    print(f"  Warning: failed to generate sampled PLY frame {frame_id:05d}: {e}")
                    continue
            else:
                invalid_sampled_files += 1
                skipped_invalid_files += 1
                print(f"  Warning: skipping missing/invalid sampled PLY frame {frame_id:05d}: {ply_path}")
                continue

        try:
            if exclude_gt:
                if generated_pcd is not None:
                    pcd = generated_pcd
                else:
                    pcd = o3d.io.read_point_cloud(ply_path)

                gt_boxes = load_gt_boxes_for_frame(args, split, scene, frame_id)
                gt_boxes = [box for box in gt_boxes if box["label"] in excluded_label_set]
                excluded_gt_boxes_total += len(gt_boxes)
                pcd, excluded_points = remove_points_inside_gt_boxes(
                    pcd,
                    gt_boxes,
                    args.gt_box_scale,
                    args.gt_yaw_only,
                )
                excluded_points_total += excluded_points
                point_count, voxel_count = accumulate_point_cloud(
                    pcd,
                    args.voxel_size,
                    voxel_presence_count,
                    voxel_xyz_sum,
                    voxel_rgb_sum,
                    voxel_point_count,
                )
            elif generated_pcd is not None and not args.save_generated_pcd:
                point_count, voxel_count = accumulate_point_cloud(
                    generated_pcd,
                    args.voxel_size,
                    voxel_presence_count,
                    voxel_xyz_sum,
                    voxel_rgb_sum,
                    voxel_point_count,
                )
            else:
                point_count, voxel_count = accumulate_frame(
                    ply_path,
                    args.voxel_size,
                    voxel_presence_count,
                    voxel_xyz_sum,
                    voxel_rgb_sum,
                    voxel_point_count,
                )
        except Exception as e:
            invalid_sampled_files += 1
            skipped_invalid_files += 1
            print(f"  Warning: failed to accumulate PLY frame {frame_id:05d}: {e}")
            continue

        valid_sampled_files += 1
        remaining_points_total += point_count
        if valid_sampled_files % 10 == 0 or sampled_idx == len(sampled_frame_ids):
            print(
                f"  Progress: {sampled_idx}/{len(sampled_frame_ids)} sampled, "
                f"valid={valid_sampled_files}, invalid={invalid_sampled_files}, "
                f"last_points={point_count}, last_voxels={voxel_count}"
            )

    print(f"  invalid files skipped: {invalid_sampled_files}")
    print(f"  existing valid sampled PLY count: {existing_valid_sampled_ply_count}")
    print(f"  generated missing PLY count: {generated_missing_pcd_files}")
    print(f"  valid sampled frames: {valid_sampled_files}")
    print(f"  excluded_gt_boxes_total: {excluded_gt_boxes_total}")
    print(f"  excluded_points_total: {excluded_points_total}")
    print(f"  remaining_points_total: {remaining_points_total}")

    if valid_sampled_files == 0:
        message = f"no valid sampled frames for {split}/{scene}"
        print(f"  Error: {message}; skipping.")
        return scene_result(scene, split, "failed", output_path, 0, time.time() - start_time, message)

    min_presence_frames = int(math.ceil(valid_sampled_files * args.presence_ratio))
    static_pcd = build_static_point_cloud(
        voxel_presence_count,
        voxel_xyz_sum,
        voxel_rgb_sum,
        voxel_point_count,
        min_presence_frames,
    )
    static_points = len(static_pcd.points)
    elapsed_seconds = time.time() - start_time

    print(f"  min_presence_frames: {min_presence_frames}")
    print(f"  static point count: {static_points}")

    if static_points == 0:
        print("  Warning: no static voxels found. Try lowering --presence-ratio.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_ok = o3d.io.write_point_cloud(output_path, static_pcd, format="ply")
    if not write_ok:
        message = f"failed to write static PLY: {output_path}"
        print(f"  Error: {message}")
        return scene_result(scene, split, "failed", output_path, static_points, time.time() - start_time, message)

    summary = {
        "scene": scene,
        "split": split,
        "source": args.source,
        "pcd_dir": pcd_dir,
        "output_path": output_path,
        "total_ply_files_before": total_ply_files_before,
        "sampled_files": len(sampled_frame_ids),
        "valid_sampled_files": valid_sampled_files,
        "generated_missing_pcd_files": generated_missing_pcd_files,
        "skipped_invalid_files": skipped_invalid_files,
        "invalid_sampled_files": invalid_sampled_files,
        "voxel_size": args.voxel_size,
        "presence_ratio": args.presence_ratio,
        "min_presence_frames": min_presence_frames,
        "static_points": static_points,
        "exclude_labeled_objects": args.exclude_labeled_objects,
        "exclude_gt_labels": args.exclude_gt_labels,
        "exclude_gt_classes": args.exclude_gt_classes,
        "gt_box_scale": args.gt_box_scale,
        "gt_source": args.gt_source,
        "gt_yaw_only": args.gt_yaw_only,
        "excluded_gt_boxes_total": excluded_gt_boxes_total,
        "excluded_points_total": excluded_points_total,
        "remaining_points_total": remaining_points_total,
        "elapsed_seconds": elapsed_seconds,
    }
    write_summary(summary_path, summary)

    print(f"  summary path: {summary_path}")
    print(f"  elapsed time: {elapsed_seconds:.2f} seconds")
    return scene_result(scene, split, "processed", output_path, static_points, elapsed_seconds)


def main():
    args = parse_args()
    resolve_roots(args)

    if args.frame_stride_static <= 0:
        raise ValueError("--frame-stride-static must be positive.")
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive.")
    if args.merge_voxel_size <= 0:
        raise ValueError("--merge-voxel-size must be positive.")
    if not 0.0 <= args.presence_ratio <= 1.0:
        raise ValueError("--presence-ratio must be between 0 and 1.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")
    if args.max_cameras is not None and args.max_cameras <= 0:
        raise ValueError("--max-cameras must be positive when provided.")
    if args.source == "raw" and args.data_root is None:
        raise ValueError("--data-root is required when --source raw is used.")
    if args.gt_box_scale <= 0:
        raise ValueError("--gt-box-scale must be positive.")
    for label in args.exclude_gt_labels:
        if label < 0 or label >= len(CLASS_NAMES):
            raise ValueError(f"--exclude-gt-labels values must be between 0 and {len(CLASS_NAMES) - 1}.")
    get_excluded_gt_labels(args)
    if args.num_workers < 1:
        args.num_workers = 1

    print("Static PCD generation")
    print(f"  processing_root: {args.processing_root}")
    print(f"  data_root: {args.data_root}")
    print(f"  pcd_root: {args.pcd_root}")
    print(f"  background_root: {args.background_root}")
    print(f"  source: {args.source}")
    print(f"  ensure_sampled_pcd: {args.ensure_sampled_pcd}")
    print(f"  save_generated_pcd: {args.save_generated_pcd}")
    print(f"  splits: {args.splits}")
    print(f"  scene filters: {'Warehouse_*' if args.all_scenes or not args.scene_name else args.scene_name}")
    print(f"  frame_stride_static: {args.frame_stride_static}")
    print(f"  voxel_size: {args.voxel_size}")
    print(f"  merge_voxel_size: {args.merge_voxel_size}")
    print(f"  presence_ratio: {args.presence_ratio}")
    print(f"  min_ply_size_mb: {args.min_ply_size_mb}")
    print(f"  max_cameras: {args.max_cameras}")
    print(f"  skip_existing: {args.skip_existing}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  exclude_labeled_objects: {args.exclude_labeled_objects}")
    print(f"  exclude_gt_labels: {args.exclude_gt_labels}")
    print(f"  exclude_gt_classes: {args.exclude_gt_classes}")
    print(f"  gt_box_scale: {args.gt_box_scale}")
    print(f"  gt_source: {args.gt_source}")
    print(f"  gt_yaw_only: {args.gt_yaw_only}")

    global_start = time.time()
    global_summary = {
        "total_scenes": 0,
        "processed_scenes": 0,
        "skipped_existing": 0,
        "failed_scenes": 0,
    }

    scene_tasks = []
    for split in args.splits:
        scene_dirs = get_scene_dirs(args.pcd_root, args.scene_name, args.all_scenes)
        print(f"\nSelected scenes for split {split}: {[os.path.basename(path) for path in scene_dirs]}")
        if not scene_dirs:
            print(f"Warning: no scenes found under {args.pcd_root}")
            continue
        for scene_dir in scene_dirs:
            scene_tasks.append((split, scene_dir))

    global_summary["total_scenes"] = len(scene_tasks)

    def handle_result(result):
        status = result.get("status", "failed") if result else "failed"
        if status == "processed":
            global_summary["processed_scenes"] += 1
        elif status == "skipped":
            global_summary["skipped_existing"] += 1
        else:
            global_summary["failed_scenes"] += 1
        print(
            "Per-scene result: "
            f"scene={result.get('scene') if result else 'unknown'}, "
            f"status={status}, "
            f"output_path={result.get('output_path') if result else None}, "
            f"static_points={result.get('static_points') if result else None}, "
            f"elapsed_seconds={result.get('elapsed_seconds', 0.0) if result else 0.0:.2f}"
        )
        if result and result.get("message"):
            print(f"  message: {result['message']}")

    if args.num_workers <= 1:
        for split, scene_dir in scene_tasks:
            try:
                result = process_scene(args, split, scene_dir)
            except Exception as e:
                result = scene_result(
                    os.path.basename(scene_dir),
                    split,
                    "failed",
                    None,
                    None,
                    0.0,
                    str(e),
                )
                print(f"Error processing scene {os.path.basename(scene_dir)}: {e}")
            handle_result(result)
    else:
        print(f"Using {args.num_workers} worker processes for static generation.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_scene, args, split, scene_dir)
                for split, scene_dir in scene_tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = scene_result(
                        "unknown",
                        "unknown",
                        "failed",
                        None,
                        None,
                        0.0,
                        str(e),
                    )
                    print(f"Error processing scene worker: {e}")
                handle_result(result)

    total_elapsed_seconds = time.time() - global_start
    print("\nGlobal static generation summary")
    print(f"  total_scenes: {global_summary['total_scenes']}")
    print(f"  processed_scenes: {global_summary['processed_scenes']}")
    print(f"  skipped_existing: {global_summary['skipped_existing']}")
    print(f"  failed_scenes: {global_summary['failed_scenes']}")
    print(f"  total_elapsed_seconds: {total_elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
