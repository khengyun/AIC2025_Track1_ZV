#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import laspy


def run_cmd(cmd, dry_run=False):
    print("[CMD]", " ".join(map(str, cmd)))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ply_to_las(ply_path: Path, las_path: Path, scale: float = 0.0001, overwrite: bool = False):
    if las_path.exists() and not overwrite:
        print(f"[SKIP LAS] {las_path}")
        return

    las_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[PLY→LAS] {ply_path.name} -> {las_path.name}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)

    if points.size == 0:
        raise RuntimeError(f"No points loaded from {ply_path}")

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([scale, scale, scale])
    header.offsets = points.min(axis=0)

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    colors = np.asarray(pcd.colors)
    if colors.size > 0:
        colors16 = np.clip(colors * 65535, 0, 65535).astype(np.uint16)
        las.red = colors16[:, 0]
        las.green = colors16[:, 1]
        las.blue = colors16[:, 2]

    las.write(str(las_path))

    print(f"  points={len(points):,}")
    print(f"  min={points.min(axis=0)}")
    print(f"  max={points.max(axis=0)}")
    print(f"  saved={las_path}")


def las_to_potree(
    las_path: Path,
    potree_out_dir: Path,
    converter: Path,
    overwrite: bool = False,
    method: str = "poisson",
    generate_page: bool = False,
):
    metadata = potree_out_dir / "metadata.json"

    if metadata.exists() and not overwrite:
        print(f"[SKIP POTREE] {potree_out_dir}")
        return

    if potree_out_dir.exists() and overwrite:
        shutil.rmtree(potree_out_dir)

    potree_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(converter),
        "-i", str(las_path),
        "-o", str(potree_out_dir),
        "-m", method,
    ]

    if generate_page:
        cmd += ["-p", potree_out_dir.name]

    print(f"[LAS→POTREE] {las_path.name} -> {potree_out_dir}")
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert AICity PLY sequence to Potree format."
    )

    parser.add_argument(
        "--input-dir",
        default="/home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/pcd",
        help="Folder containing .ply files.",
    )
    parser.add_argument(
        "--pattern",
        default="Lab_000_*.ply",
        help="PLY glob pattern.",
    )
    parser.add_argument(
        "--las-dir",
        default="/home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/las",
        help="Output folder for intermediate .las files.",
    )
    parser.add_argument(
        "--potree-out-root",
        default="/home/niran/Documents/GitHub/potree/pointclouds/aicity/Lab_000",
        help="Output root for Potree converted frame folders.",
    )
    parser.add_argument(
        "--converter",
        default="/home/niran/Documents/GitHub/potree/PotreeConverter/build/PotreeConverter",
        help="Path to PotreeConverter binary.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of frames. 0 means all.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index after sorting matched files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing LAS and Potree output.",
    )
    parser.add_argument(
        "--only-las",
        action="store_true",
        help="Only convert PLY to LAS.",
    )
    parser.add_argument(
        "--only-potree",
        action="store_true",
        help="Only convert existing LAS to Potree.",
    )
    parser.add_argument(
        "--method",
        default="poisson",
        choices=["poisson", "poisson_average", "random"],
        help="PotreeConverter sampling method.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0001,
        help="LAS coordinate scale.",
    )
    parser.add_argument(
        "--scene-config",
        default="/home/niran/Documents/GitHub/potree/aicity/Lab_000/scene_config.json",
        help="Write frame list JSON for Potree viewer.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    las_dir = Path(args.las_dir)
    potree_out_root = Path(args.potree_out_root)
    converter = Path(args.converter)
    scene_config_path = Path(args.scene_config)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if not converter.exists():
        raise FileNotFoundError(f"PotreeConverter not found: {converter}")

    ply_files = sorted(input_dir.glob(args.pattern))

    if args.start_index > 0:
        ply_files = ply_files[args.start_index:]

    if args.max_frames > 0:
        ply_files = ply_files[: args.max_frames]

    if not ply_files:
        raise RuntimeError(f"No PLY files found: {input_dir}/{args.pattern}")

    print("=" * 80)
    print("AICity PLY sequence to Potree converter")
    print("=" * 80)
    print("input_dir       :", input_dir)
    print("pattern         :", args.pattern)
    print("num_frames      :", len(ply_files))
    print("las_dir         :", las_dir)
    print("potree_out_root :", potree_out_root)
    print("converter       :", converter)
    print("overwrite       :", args.overwrite)
    print("=" * 80)

    t0 = time.time()
    frames = []

    for idx, ply_path in enumerate(ply_files):
        frame_name = ply_path.stem
        las_path = las_dir / f"{frame_name}.las"
        potree_out_dir = potree_out_root / frame_name

        print()
        print(f"[{idx + 1}/{len(ply_files)}] {frame_name}")

        if not args.only_potree:
            ply_to_las(
                ply_path=ply_path,
                las_path=las_path,
                scale=args.scale,
                overwrite=args.overwrite,
            )

        if not args.only_las:
            if not las_path.exists():
                raise FileNotFoundError(f"LAS not found: {las_path}")

            las_to_potree(
                las_path=las_path,
                potree_out_dir=potree_out_dir,
                converter=converter,
                overwrite=args.overwrite,
                method=args.method,
            )

        frames.append({
            "id": idx,
            "name": frame_name,
            "ply_path": str(ply_path),
            "las_path": str(las_path),
            "potree_dir": str(potree_out_dir),
            "cloud_url": f"../pointclouds/aicity/Lab_000/{frame_name}/metadata.json",
        })

    scene_config_path.parent.mkdir(parents=True, exist_ok=True)
    scene_config = {
        "scene_id": "Lab_000",
        "calibration_url": "../aicity/Lab_000/calibration.json",
        "frames": frames,
    }

    with open(scene_config_path, "w") as f:
        json.dump(scene_config, f, indent=2)

    print()
    print("=" * 80)
    print("DONE")
    print(f"frames converted: {len(frames)}")
    print(f"scene_config: {scene_config_path}")
    print(f"elapsed: {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
