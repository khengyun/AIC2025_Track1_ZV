import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reduced PCD data for a single scene.")
    parser.add_argument("--data-root", type=str, default="dataset/MTMC_Tracking_2025")
    parser.add_argument("--out-dir", type=str, default="dataset/pcd_dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scene-name", type=str, required=True)
    parser.add_argument("--max-seconds", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = Path(__file__).with_name("generate_pcd_data.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root", args.data_root,
        "--out-dir", args.out_dir,
        "--splits", args.split,
        "--scene-name", args.scene_name,
        "--max-seconds", str(args.max_seconds),
        "--fps", str(args.fps),
        "--frame-stride", str(args.frame_stride),
        "--max-cameras", str(args.max_cameras),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
