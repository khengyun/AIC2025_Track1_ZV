#!/usr/bin/env python3
"""
Point Cloud Generation Wrapper for Single Scene

This script wraps the generate_pcd_data.py functionality to provide:
- Flexible input/output path handling
- Automatic output directory organization
- Validation of input structure
- Detailed progress reporting

Usage:
    python tools/generate_pcd_single_scene.py \
        --data-root /path/to/dataset \
        --scene-name Warehouse_017 \
        --split train \
        --output-dir /path/to/output \
        [--out-layout scene] \
        [--frame-stride 1] \
        [--max-seconds 300] \
        [--max-cameras 8] \
        [--fps 30]

Example with local test data:
    python tools/generate_pcd_single_scene.py \
        --data-root dataset/MTMC_Tracking_2025 \
        --scene-name Warehouse_019 \
        --split test \
        --output-dir outputs/pcd_warehouse_019 \
        --out-layout scene \
        --frame-stride 30 \
        --max-cameras 5
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional


def validate_scene_directory(scene_path: str) -> dict:
    """
    Validate that the scene directory has the required structure.
    
    Returns:
        dict: Information about the scene (calibration, cameras, etc.)
    """
    scene_path = Path(scene_path)
    
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_path}")
    
    # Check required files/directories
    required_items = {
        'calibration.json': 'file',
        'depth_maps': 'dir',
        'videos': 'dir',
    }
    
    missing = []
    for item_name, item_type in required_items.items():
        item_path = scene_path / item_name
        if not item_path.exists():
            missing.append(f"  - {item_name} ({item_type})")
    
    if missing:
        raise ValueError(
            f"Scene directory {scene_path} is missing required items:\n" +
            "\n".join(missing)
        )
    
    # Parse calibration to get camera count
    with open(scene_path / 'calibration.json', 'r') as f:
        calib_data = json.load(f)
    
    num_cameras = len(calib_data.get('sensors', []))
    
    # Count video files
    video_dir = scene_path / 'videos'
    video_files = list(video_dir.glob('*.mp4'))
    
    # Count depth map files
    depth_dir = scene_path / 'depth_maps'
    depth_files = list(depth_dir.glob('*.h5'))
    
    return {
        'cameras': num_cameras,
        'videos': len(video_files),
        'depth_maps': len(depth_files),
        'has_map': (scene_path / 'map.png').exists(),
    }


def create_output_directory(output_dir: str, split: str, scene_name: str, out_layout: str) -> str:
    """
    Create output directory structure and return the PLY output path.
    
    The PLY files will be written to:
      - scene layout: {output_dir}/{scene_name}/
      - split layout: {output_dir}/{split}/pcd/
    """
    if out_layout == 'scene':
        pcd_dir = Path(output_dir) / scene_name
    else:
        pcd_dir = Path(output_dir) / split / 'pcd'
    pcd_dir.mkdir(parents=True, exist_ok=True)
    
    return str(Path(output_dir))


def run_pcd_generation(
    data_root: str,
    scene_name: str,
    split: str,
    output_dir: str,
    out_layout: str = "scene",
    frame_stride: int = 1,
    max_seconds: Optional[int] = None,
    max_cameras: Optional[int] = None,
    fps: int = 30,
    include_gt: bool = False,
    verbose: bool = True,
) -> int:
    """
    Run point cloud generation using the main generate_pcd_data.py script.
    """
    
    # Validate input
    scene_path = Path(data_root) / split / scene_name
    
    if verbose:
        print("=" * 60)
        print("Point Cloud Generation for Single Scene")
        print("=" * 60)
        print()
        print("Validating input scene directory...")
    
    try:
        scene_info = validate_scene_directory(str(scene_path))
        
        if verbose:
            print(f"OK: Scene directory valid: {scene_path}")
            print(f"  - Cameras: {scene_info['cameras']}")
            print(f"  - Video files: {scene_info['videos']}")
            print(f"  - Depth maps: {scene_info['depth_maps']}")
            print(f"  - Has map.png: {scene_info['has_map']}")
            print()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Validation failed: {e}", file=sys.stderr)
        return 1
    
    # Create output directory
    if verbose:
        print("Setting up output directory...")
    
    create_output_directory(output_dir, split, scene_name, out_layout)
    if out_layout == "scene":
        pcd_output_dir = Path(output_dir) / scene_name
    else:
        pcd_output_dir = Path(output_dir) / split / 'pcd'
    
    if verbose:
        print(f"OK: Output directory: {pcd_output_dir}")
        print()
        print("Configuration:")
        print(f"  Data Root:    {data_root}")
        print(f"  Scene Name:   {scene_name}")
        print(f"  Split:        {split}")
        print(f"  Out Layout:   {out_layout}")
        print(f"  Frame Stride: {frame_stride}")
        print(f"  Max Seconds:  {max_seconds if max_seconds else 'None (all frames)'}")
        print(f"  Max Cameras:  {max_cameras if max_cameras else 'None (all cameras)'}")
        print(f"  FPS:          {fps}")
        print(f"  Export GT:    {include_gt}")
        print()
        print("Running point cloud generation...")
        print()
    
    # Build command for generate_pcd_data.py
    cmd = [
        sys.executable,
        'tools/generate_pcd_data.py',
        '--data-root', data_root,
        '--out-dir', output_dir,
        '--out-layout', out_layout,
        '--splits', split,
        '--scene-name', scene_name,
        '--frame-stride', str(frame_stride),
        '--fps', str(fps),
    ]
    
    if max_seconds is not None:
        cmd.extend(['--max-seconds', str(max_seconds)])
    
    if max_cameras is not None:
        cmd.extend(['--max-cameras', str(max_cameras)])

    if not include_gt:
        cmd.append('--no-gt')
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
        print()
    
    # Run the generation script
    try:
        result = subprocess.run(cmd, check=True)
        return_code = result.returncode
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Generation script failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"ERROR: Error running generation script: {e}", file=sys.stderr)
        return 1
    
    # Check and report results
    if verbose:
        print()
        print("=" * 60)
        print("Generation Complete - Validating Output")
        print("=" * 60)
        print()
    
    ply_files = sorted(pcd_output_dir.glob(f'{scene_name}_*.ply'))
    
    if not ply_files:
        print(f"ERROR: No .ply files were generated in {pcd_output_dir}", file=sys.stderr)
        return 1
    
    if verbose:
        print(f"OK: Generated {len(ply_files)} .ply files")
        print()
        print("Output directory: " + str(pcd_output_dir))
        print()
        print("First 5 generated files:")
        for ply_file in ply_files[:5]:
            size_mb = ply_file.stat().st_size / (1024 * 1024)
            print(f"  - {ply_file.name} ({size_mb:.2f} MB)")
        
        if len(ply_files) > 5:
            print(f"  ... and {len(ply_files) - 5} more files")
        
        print()
        print("=" * 60)
        print(f"SUCCESS: Point cloud generation completed for {scene_name}")
        print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate point cloud (.ply) files for a single scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of the dataset (containing train/val/test splits)',
    )
    
    parser.add_argument(
        '--scene-name',
        type=str,
        required=True,
        help='Scene name to process (e.g., Warehouse_017)',
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to process (default: train)',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for .ply files. Layout depends on --out-layout.',
    )

    parser.add_argument(
        '--out-layout',
        type=str,
        choices=['scene', 'split'],
        default='scene',
        help='Output layout. "scene" writes to {output-dir}/{scene_name}/, "split" writes to {output-dir}/{split}/pcd/',
    )
    
    parser.add_argument(
        '--frame-stride',
        type=int,
        default=1,
        help='Frame sampling stride (default: 1, process every frame)',
    )
    
    parser.add_argument(
        '--max-seconds',
        type=int,
        default=None,
        help='Maximum duration to process (in seconds)',
    )
    
    parser.add_argument(
        '--max-cameras',
        type=int,
        default=None,
        help='Maximum number of cameras to use',
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Dataset frames per second (default: 30)',
    )

    parser.add_argument(
        '--include-gt',
        action='store_true',
        help='Also export ground-truth txt files for train/val splits',
    )
    
    args = parser.parse_args()
    
    return run_pcd_generation(
        data_root=args.data_root,
        scene_name=args.scene_name,
        split=args.split,
        output_dir=args.output_dir,
        out_layout=args.out_layout,
        frame_stride=args.frame_stride,
        max_seconds=args.max_seconds,
        max_cameras=args.max_cameras,
        fps=args.fps,
        include_gt=args.include_gt,
        verbose=True,
    )


if __name__ == '__main__':
    sys.exit(main())
