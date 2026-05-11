import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np

o3d = None


def load_open3d():
    global o3d
    if o3d is None:
        import open3d as open3d_module
        o3d = open3d_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one static/background PLY per scene from per-frame point clouds."
    )
    parser.add_argument("--pcd-root", required=True,
                        help="Root point cloud folder.")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to process.")
    parser.add_argument("--scene-name", action="append", default=None,
                        help="Optional scene name filter. Can be passed multiple times.")
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
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit on sampled frames for quick tests.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected frames and output paths without reading or writing PLY files.")
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


def get_scene_dirs(pcd_root, split, scene_names):
    split_dir = os.path.join(pcd_root, split)
    if scene_names:
        return [
            os.path.join(split_dir, scene)
            for scene in scene_names
            if os.path.isdir(os.path.join(split_dir, scene))
        ]
    if not os.path.isdir(split_dir):
        return []
    return [
        os.path.join(split_dir, name)
        for name in sorted(os.listdir(split_dir))
        if name.startswith("Warehouse_") and os.path.isdir(os.path.join(split_dir, name))
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


def build_output_paths(scene_dir, scene, voxel_size):
    voxel_mm = int(voxel_size * 1000)
    base_name = f"{scene}_static_voxel{voxel_mm:03d}"
    static_dir = os.path.join(scene_dir, "static")
    return (
        os.path.join(static_dir, f"{base_name}.ply"),
        os.path.join(static_dir, f"{base_name}_summary.json"),
    )


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


def select_sampled_files(ply_files, frame_stride_static, max_frames):
    sampled = [
        (frame_id, path)
        for frame_id, path in ply_files
        if frame_id % frame_stride_static == 0
    ]
    if max_frames is not None:
        sampled = sampled[:max_frames]
    return sampled


def accumulate_frame(
    ply_path,
    voxel_size,
    voxel_presence_count,
    voxel_xyz_sum,
    voxel_rgb_sum,
    voxel_point_count,
):
    pcd = o3d.io.read_point_cloud(ply_path)
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


def process_scene(args, split, scene_dir):
    scene = os.path.basename(scene_dir)
    start_time = time.time()
    pcd_dir = os.path.join(scene_dir, "pcd")
    output_path, summary_path = build_output_paths(scene_dir, scene, args.voxel_size)

    ply_files = get_scene_ply_files(scene_dir, scene)
    sampled_files = select_sampled_files(
        ply_files,
        args.frame_stride_static,
        args.max_frames,
    )

    print(f"\nScene: {split}/{scene}")
    print(f"  pcd_dir: {pcd_dir}")
    print(f"  total PLY files: {len(ply_files)}")
    print(f"  sampled frames: {len(sampled_files)}")
    print(f"  output path: {output_path}")

    if args.dry_run:
        print("  dry-run selected frames:")
        for frame_id, path in sampled_files:
            print(f"    {frame_id:05d}: {path}")
        return

    if os.path.exists(output_path) and not args.overwrite:
        print("  Skipping existing output. Use --overwrite to regenerate.")
        return

    voxel_presence_count = defaultdict(int)
    voxel_xyz_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    voxel_rgb_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    voxel_point_count = defaultdict(int)

    valid_sampled_files = 0
    invalid_sampled_files = 0

    for sampled_idx, (frame_id, ply_path) in enumerate(sampled_files, start=1):
        if not is_valid_ply(ply_path, args.min_ply_size_mb):
            invalid_sampled_files += 1
            print(f"  Warning: skipping invalid PLY frame {frame_id:05d}: {ply_path}")
            continue

        try:
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
            print(f"  Warning: failed to read PLY frame {frame_id:05d}: {e}")
            continue

        valid_sampled_files += 1
        if valid_sampled_files % 10 == 0 or sampled_idx == len(sampled_files):
            print(
                f"  Progress: {sampled_idx}/{len(sampled_files)} sampled, "
                f"valid={valid_sampled_files}, invalid={invalid_sampled_files}, "
                f"last_points={point_count}, last_voxels={voxel_count}"
            )

    if valid_sampled_files == 0:
        print(f"  Error: no valid sampled frames for {split}/{scene}; skipping.")
        return

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

    print(f"  invalid files skipped: {invalid_sampled_files}")
    print(f"  valid sampled frames: {valid_sampled_files}")
    print(f"  min_presence_frames: {min_presence_frames}")
    print(f"  static point count: {static_points}")

    if static_points == 0:
        print("  Warning: no static voxels found. Try lowering --presence-ratio.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_ok = o3d.io.write_point_cloud(output_path, static_pcd, format="ply")
    if not write_ok:
        print(f"  Error: failed to write static PLY: {output_path}")
        return

    summary = {
        "scene": scene,
        "split": split,
        "pcd_dir": pcd_dir,
        "output_path": output_path,
        "total_ply_files": len(ply_files),
        "sampled_files": len(sampled_files),
        "valid_sampled_files": valid_sampled_files,
        "invalid_sampled_files": invalid_sampled_files,
        "voxel_size": args.voxel_size,
        "presence_ratio": args.presence_ratio,
        "min_presence_frames": min_presence_frames,
        "static_points": static_points,
        "elapsed_seconds": elapsed_seconds,
    }
    write_summary(summary_path, summary)

    print(f"  summary path: {summary_path}")
    print(f"  elapsed time: {elapsed_seconds:.2f} seconds")


def main():
    args = parse_args()

    if args.frame_stride_static <= 0:
        raise ValueError("--frame-stride-static must be positive.")
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive.")
    if not 0.0 <= args.presence_ratio <= 1.0:
        raise ValueError("--presence-ratio must be between 0 and 1.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")

    print("Static PCD generation")
    print(f"  pcd_root: {args.pcd_root}")
    print(f"  splits: {args.splits}")
    print(f"  scene filters: {args.scene_name if args.scene_name else 'Warehouse_*'}")
    print(f"  frame_stride_static: {args.frame_stride_static}")
    print(f"  voxel_size: {args.voxel_size}")
    print(f"  presence_ratio: {args.presence_ratio}")
    print(f"  min_ply_size_mb: {args.min_ply_size_mb}")
    print(f"  dry_run: {args.dry_run}")

    if not args.dry_run:
        load_open3d()

    for split in args.splits:
        scene_dirs = get_scene_dirs(args.pcd_root, split, args.scene_name)
        print(f"\nSelected scenes for split {split}: {[os.path.basename(path) for path in scene_dirs]}")
        if not scene_dirs:
            print(f"Warning: no scenes found for split {split}")
            continue
        for scene_dir in scene_dirs:
            process_scene(args, split, scene_dir)


if __name__ == "__main__":
    main()
