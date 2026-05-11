import os
import cv2
import numpy as np
import h5py
import json
import open3d as o3d
import time

import argparse
import concurrent.futures
from typing import Dict, List, Tuple, Optional



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate point clouds (PLY) from multi-camera RGB + depth sequences."
    )
    parser.add_argument("--data-root", type=str, default="dataset/MTMC_Tracking_2025",
                        help="Dataset root directory.")
    parser.add_argument("--out-dir", type=str, default="dataset/pcd_dataset",
                        help="Output root directory for generated PLY files.")
    parser.add_argument("--out-layout", choices=["split", "scene"], default="split",
                        help="Output layout. 'split' writes to <out_dir>/<split>/<scene_name>, 'scene' writes to <out_dir>/<scene_name>.")
    parser.add_argument("--mode", choices=["all", "pcd", "gt"], default="all",
                        help="Generation mode: 'all' writes PLY and GT, 'pcd' writes PLY only, 'gt' writes GT only.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Splits to process (space-separated). e.g., --splits train val")
    parser.add_argument("--scene-name", action="append", default=None,
                        help="Optional scene name filter. Can be passed multiple times.")
    parser.add_argument("--no-gt", action="store_true",
                        help="Skip exporting ground-truth text files for train/val splits.")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help="Optional maximum duration to process per scene.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Dataset FPS used to convert seconds into frame indices.")
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="Optional frame stride. Defaults to 100 for train/val and 1 for test.")
    parser.add_argument("--max-cameras", type=int, default=None,
                        help="Optional maximum number of cameras to use per scene.")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                        help="Voxel size for downsampling when point count is large.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of worker processes for per-frame generation. 1 disables multiprocessing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing PLY files instead of skipping.")
    parser.add_argument("--repair-bad", action="store_true",
                        help="Regenerate existing PLY files that are missing, too small, or have an invalid header.")
    parser.add_argument("--min-ply-size-mb", type=float, default=10.0,
                        help="Minimum valid PLY file size in MB when --repair-bad is enabled.")
    parser.add_argument("--gt-only", action="store_true",
                        help="Only export ground-truth text files and skip PLY generation.")
    
    return parser.parse_args()


def normalize_camera_name(camera_name):
    camera_names = camera_name.split('_')
    if len(camera_names) > 1:
        return camera_names[0] + '_' + f"{int(camera_names[1]):04d}"
    return camera_names[0] + '_0000'


def get_selected_frame_indices(split, max_seconds, fps, frame_stride):
    if frame_stride is None:
        frame_stride = 100 if split in {'train', 'val'} else 1
    max_frame = 9000 if max_seconds is None else min(9000, max_seconds * fps)
    return list(range(0, max_frame, frame_stride))

def get_calibration_per_camera(calibration_file_path):
    """
    Reads the camera calibration file and returns the intrinsic and extrinsic parameters for each camera.
    """
    with open(calibration_file_path, 'r') as f:
        data = json.load(f)

    camera_params ={}
    for calib in data['sensors']:
        camera_name = normalize_camera_name(calib['id'])
        camera_intrinsic = np.array(calib['intrinsicMatrix']).reshape(3, 3)
        camera_intrinsic = camera_intrinsic[0,0], camera_intrinsic[1,1], camera_intrinsic[0,2], camera_intrinsic[1,2]
        camera_extrinsic = np.eye(4)
        camera_extrinsic[:3,:4] = np.array(calib['extrinsicMatrix']).reshape(3, 4)
        camera_params[camera_name] = {
            'intrinsic': camera_intrinsic,
            'extrinsic': camera_extrinsic
        }
    
    return camera_params

def get_point_cloud(rgb_image_per_camera, depth_image_per_camera, camera_params, voxel_size=0.02):
    pcd_combined = o3d.geometry.PointCloud()

    for camera_name in rgb_image_per_camera.keys():
        if depth_image_per_camera[camera_name] is None:
            continue

        rgb_image = rgb_image_per_camera[camera_name]
        H, W = rgb_image.shape[:2]
        depth_image = depth_image_per_camera[camera_name]
        depth_image = np.array(depth_image, dtype=np.float32) / 1000.0  # mm → meters

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_o3d = o3d.geometry.Image(rgb_image)
        depth_o3d = o3d.geometry.Image(depth_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)

        fx, fy, cx, cy = camera_params[camera_name]['intrinsic']
        cam_intr = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)

        # RGBD → PointCloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intr)

        extrinsic = camera_params[camera_name]['extrinsic']
        pcd.transform(np.linalg.inv(extrinsic)) 

        pcd_combined += pcd

    if len(pcd_combined.points) > 1000000:
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

    return pcd_combined


def build_pcd_output_path(out_dir, out_layout, split, scene_name, frame_count):
    if out_layout == "scene":
        return os.path.join(out_dir, scene_name, "pcd", f"{scene_name}_{frame_count:05d}.ply")
    return os.path.join(out_dir, split, scene_name, "pcd", f"{scene_name}_{frame_count:05d}.ply")


def build_gt_output_path(out_dir, out_layout, split, scene_name, frame_count):
    if out_layout == "scene":
        return os.path.join(out_dir, scene_name, "gt", f"{scene_name}_{frame_count:05d}.txt")
    return os.path.join(out_dir, split, scene_name, "gt", f"{scene_name}_{frame_count:05d}.txt")


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


def get_existing_ply_action(output_path, overwrite, repair_bad, min_ply_size_mb):
    if not os.path.exists(output_path) or overwrite:
        return "generate_missing"
    if not repair_bad:
        return "skip"
    if is_valid_ply(output_path, min_ply_size_mb):
        return "skip"
    return "regenerate_invalid"


def write_point_cloud_atomic(output_path, pcd, min_ply_size_mb):
    tmp_output_path = output_path + ".tmp"
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


def new_scene_summary(total_frames):
    return {
        "total_frames": total_frames,
        "skipped_valid": 0,
        "regenerated_invalid": 0,
        "generated_missing": 0,
        "failed": 0,
    }


def print_scene_summary(scene_name, summary):
    print(
        f"Summary for {scene_name}: "
        f"total frames={summary['total_frames']}, "
        f"skipped valid={summary['skipped_valid']}, "
        f"regenerated invalid={summary['regenerated_invalid']}, "
        f"generated missing={summary['generated_missing']}, "
        f"failed={summary['failed']}"
    )


def process_frame_task(task):
    start_time = time.time()
    split = task["split"]
    domain_name = task["domain_name"]
    domain_path = task["domain_path"]
    frame_count = task["frame_count"]
    camera_name_list = task["camera_name_list"]
    video_path_list = task["video_path_list"]
    depth_map_path_list = task["depth_map_path_list"]
    camera_params = task["camera_params"]
    out_dir = task["out_dir"]
    out_layout = task["out_layout"]
    voxel_size = task["voxel_size"]
    overwrite = task["overwrite"]
    repair_bad = task["repair_bad"]
    min_ply_size_mb = task["min_ply_size_mb"]
    generated_type = task.get("generated_type")
    prefiltered = generated_type is not None

    output_path = build_pcd_output_path(out_dir, out_layout, split, domain_name, frame_count)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing_action = get_existing_ply_action(output_path, overwrite, repair_bad, min_ply_size_mb)
    if generated_type is None:
        generated_type = existing_action
    if existing_action == "skip":
        return {
            "status": "skipped",
            "path": output_path,
            "frame": frame_count,
            "seconds": 0.0,
        }
    if existing_action == "regenerate_invalid" and not prefiltered:
        print(f"Regenerating invalid PLY {output_path} ...")

    video_capture_per_camera = {}
    depth_map_per_camera = {}
    try:
        for idx, video_path in enumerate(video_path_list):
            if not os.path.exists(video_path):
                return {
                    "status": "missing_video",
                    "path": output_path,
                    "frame": frame_count,
                    "message": f"Video file not found {video_path}",
                }
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                return {
                    "status": "video_open_failed",
                    "path": output_path,
                    "frame": frame_count,
                    "message": f"Cannot open video file: {video_path}",
                }
            video_capture_per_camera[camera_name_list[idx]] = video_capture

        for idx, depth_map_path in enumerate(depth_map_path_list):
            if not os.path.exists(depth_map_path):
                depth_map_per_camera[camera_name_list[idx]] = None
            else:
                depth_map_per_camera[camera_name_list[idx]] = h5py.File(depth_map_path, 'r')

        rgb_image_per_camera = {}
        depth_image_per_camera = {}
        for camera_name in video_capture_per_camera:
            video_capture_per_camera[camera_name].set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = video_capture_per_camera[camera_name].read()
            if not ret:
                return {
                    "status": "frame_unavailable",
                    "path": output_path,
                    "frame": frame_count,
                    "message": f"Frame {frame_count} unavailable for {camera_name}",
                }
            rgb_image_per_camera[camera_name] = frame

            depth_map = None
            depth_map_file = depth_map_per_camera.get(camera_name)
            if depth_map_file is not None:
                try:
                    key = f"distance_to_image_plane_{frame_count:05d}.png"
                    depth_map = depth_map_file[key][:]
                except Exception:
                    depth_map = None
            depth_image_per_camera[camera_name] = depth_map

        if not rgb_image_per_camera:
            return {
                "status": "frame_unavailable",
                "path": output_path,
                "frame": frame_count,
                "message": "No RGB frames available",
            }

        pcd = get_point_cloud(
            rgb_image_per_camera,
            depth_image_per_camera,
            camera_params,
            voxel_size=voxel_size,
        )
        write_ok, write_error = write_point_cloud_atomic(output_path, pcd, min_ply_size_mb)
        if not write_ok:
            return {
                "status": "write_failed",
                "path": output_path,
                "frame": frame_count,
                "message": write_error,
            }

        return {
            "status": "ok",
            "path": output_path,
            "frame": frame_count,
            "seconds": time.time() - start_time,
            "generated_type": generated_type,
        }
    except Exception as e:
        return {
            "status": "error",
            "path": output_path,
            "frame": frame_count,
            "message": str(e),
        }
    finally:
        for video in video_capture_per_camera.values():
            video.release()
        for depth_map in depth_map_per_camera.values():
            if depth_map is not None:
                depth_map.close()


def main():
    args = parse_args()

    data_root = args.data_root
    out_dir = args.out_dir
    out_layout = args.out_layout
    if args.num_workers < 1:
        args.num_workers = 1

    split_set = args.splits
    scene_filter = set(args.scene_name) if args.scene_name else None

    if args.gt_only:
        args.mode = "gt"
    if args.gt_only and args.no_gt:
        print("Warning: --gt-only ignores --no-gt.")
        args.no_gt = False

    generate_ply = args.mode in {"all", "pcd"}
    generate_gt = args.mode in {"all", "gt"} and not (args.mode == "all" and args.no_gt)

    print(
        f"mode={args.mode}, "
        f"out_layout={args.out_layout}, "
        f"voxel_size={args.voxel_size}, "
        f"repair_bad={args.repair_bad}, "
        f"min_ply_size_mb={args.min_ply_size_mb}"
    )
    if not generate_ply:
        print(f"Skipping PLY generation (--mode {args.mode}).")
    if not generate_gt:
        print(f"Skipping GT generation (--mode {args.mode}{' --no-gt' if args.no_gt else ''}).")

    for split in (split_set if generate_ply else []):
        domain_name_list = sorted(os.listdir(os.path.join(data_root, split)))
        if scene_filter is not None:
            domain_name_list = [name for name in domain_name_list if name in scene_filter]
        frame_indices = get_selected_frame_indices(split, args.max_seconds, args.fps, args.frame_stride)
        for domain_idx, domain_name in enumerate(domain_name_list):
            domain_path = os.path.join(data_root, split, domain_name)
            print(f'Processing {domain_path}...')
            scene_summary = new_scene_summary(len(frame_indices))
            camera_params = get_calibration_per_camera(os.path.join(domain_path, 'calibration.json'))
            camera_name_list = sorted([name.split('.')[0] for name in os.listdir(os.path.join(domain_path,'videos'))])
            if args.max_cameras is not None:
                camera_name_list = camera_name_list[:args.max_cameras]
            video_path_list = [os.path.join(domain_path, 'videos', f'{name}.mp4') for name in camera_name_list]
            depth_map_path_list = [os.path.join(domain_path, 'depth_maps', f'{name}.h5') for name in camera_name_list]

            camera_name_list = [normalize_camera_name(camera_name) for camera_name in camera_name_list]

            if args.num_workers <= 1:
                depth_map_per_camera = {}
                for idx, depth_map_path in enumerate(depth_map_path_list):
                    try:
                        if not os.path.exists(depth_map_path):
                            print(f"Warning: Depth map file not found {depth_map_path}")
                            depth_map_per_camera[camera_name_list[idx]] = None
                        else:
                            depth_map = h5py.File(depth_map_path, 'r')
                            depth_map_per_camera[camera_name_list[idx]] = depth_map
                    except Exception as e:
                        print(f"Error opening depth map file {depth_map_path}: {e}")
                        depth_map_per_camera[camera_name_list[idx]] = None

                video_capture_per_camera = {}
                for idx, video_path in enumerate(video_path_list):
                    try:
                        if not os.path.exists(video_path):
                            print(f"Warning: Video file not found {video_path}")
                            continue
                        video_capture = cv2.VideoCapture(video_path)
                        if not video_capture.isOpened():
                            raise Exception(f"Cannot open video file: {video_path}")
                        video_capture_per_camera[camera_name_list[idx]] = video_capture
                    except Exception as e:
                        print(f"Error opening video file {video_path}: {e}")
                        continue
                
                for frame_count in frame_indices:
                    output_path = build_pcd_output_path(out_dir, out_layout, split, domain_name, frame_count)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    existing_action = get_existing_ply_action(
                        output_path,
                        args.overwrite,
                        args.repair_bad,
                        args.min_ply_size_mb,
                    )
                    if existing_action == "skip":
                        scene_summary["skipped_valid"] += 1
                        print(f"Skipping existing {output_path}")
                        continue
                    if existing_action == "regenerate_invalid":
                        print(f"Regenerating invalid PLY {output_path} ...")

                    rgb_image_per_camera = {}
                    depth_image_per_camera = {}
                    for camera_name in video_capture_per_camera:
                        video_capture_per_camera[camera_name].set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                        ret, frame = video_capture_per_camera[camera_name].read()
                        if not ret:
                            rgb_image_per_camera = {}
                            break
                        rgb_image_per_camera[camera_name] = frame
                        
                        depth_map = None
                        if depth_map_per_camera.get(camera_name) is not None:
                            try:
                                depth_map = depth_map_per_camera[camera_name][f'distance_to_image_plane_{frame_count:05d}.png'][:]
                            except Exception as e:
                                print(f"Error reading depth map for {camera_name}: {e}")
                        depth_image_per_camera[camera_name] = depth_map
                    if not rgb_image_per_camera:
                        scene_summary["failed"] += 1
                        print(f"Stopping {domain_name} at frame {frame_count} because video frames are unavailable.")
                        break

                    start_time = time.time()

                    pcd = get_point_cloud(rgb_image_per_camera, depth_image_per_camera, camera_params, voxel_size=args.voxel_size)

                    write_ok, write_error = write_point_cloud_atomic(output_path, pcd, args.min_ply_size_mb)
                    if not write_ok:
                        scene_summary["failed"] += 1
                        detail = write_error if write_error else "write failed"
                        print(f"Warning: {detail} (frame {frame_count})")
                        continue

                    if existing_action == "regenerate_invalid":
                        scene_summary["regenerated_invalid"] += 1
                    else:
                        scene_summary["generated_missing"] += 1

                    print(f"{output_path} Point cloud generated in {time.time() - start_time:.2f} seconds.")
                for video in video_capture_per_camera.values():
                    video.release()
                
                # Close h5py files to prevent resource leak
                for depth_map in depth_map_per_camera.values():
                    if depth_map is not None:
                        depth_map.close()
                print_scene_summary(domain_name, scene_summary)
            else:
                tasks = []
                for frame_count in frame_indices:
                    output_path = build_pcd_output_path(out_dir, out_layout, split, domain_name, frame_count)
                    existing_action = get_existing_ply_action(
                        output_path,
                        args.overwrite,
                        args.repair_bad,
                        args.min_ply_size_mb,
                    )
                    if existing_action == "skip":
                        scene_summary["skipped_valid"] += 1
                        print(f"Skipping existing {output_path}")
                        continue
                    if existing_action == "regenerate_invalid":
                        print(f"Regenerating invalid PLY {output_path} ...")
                    tasks.append({
                        "split": split,
                        "domain_name": domain_name,
                        "domain_path": domain_path,
                        "frame_count": frame_count,
                        "camera_name_list": camera_name_list,
                        "video_path_list": video_path_list,
                        "depth_map_path_list": depth_map_path_list,
                        "camera_params": camera_params,
                        "out_dir": out_dir,
                        "out_layout": out_layout,
                        "voxel_size": args.voxel_size,
                        "overwrite": args.overwrite,
                        "repair_bad": args.repair_bad,
                        "min_ply_size_mb": args.min_ply_size_mb,
                        "generated_type": existing_action,
                    })

                print(f"Using {args.num_workers} worker processes for {domain_name}...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(process_frame_task, task) for task in tasks]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                        except Exception as e:
                            scene_summary["failed"] += 1
                            print(f"Error processing frame task: {e}")
                            continue

                        status = result.get("status")
                        path = result.get("path")
                        seconds = result.get("seconds", 0.0)
                        frame = result.get("frame")
                        message = result.get("message")

                        if status == "ok":
                            if result.get("generated_type") == "regenerate_invalid":
                                scene_summary["regenerated_invalid"] += 1
                            else:
                                scene_summary["generated_missing"] += 1
                            print(f"{path} Point cloud generated in {seconds:.2f} seconds.")
                        elif status == "skipped":
                            scene_summary["skipped_valid"] += 1
                            print(f"Skipping existing {path}")
                        else:
                            scene_summary["failed"] += 1
                            detail = message if message else status
                            print(f"Warning: {detail} (frame {frame})")
                print_scene_summary(domain_name, scene_summary)

    if generate_gt:
        OBJECT_TYPES = ['Person', 'Forklift', 'NovaCarter', 'Transporter', 'FourierGR1T2', 'AgilityDigit', 'PalletTruck']
        for split in split_set:
            if split == 'test':
                continue
            domain_name_list = sorted(os.listdir(os.path.join(data_root, split)))
            if scene_filter is not None:
                domain_name_list = [name for name in domain_name_list if name in scene_filter]
            for domain_name in domain_name_list:
                domain_path = os.path.join(data_root, split, domain_name)
                gt_path = os.path.join(domain_path, 'ground_truth.json')
                with open(gt_path, 'r') as f:
                    data = json.load(f)
                
                frame_count = 0
                for frame_count in get_selected_frame_indices(split, args.max_seconds, args.fps, args.frame_stride):
                    # Skip if frame doesn't exist in ground truth
                    if str(frame_count) not in data:
                        print(f"Warning: Frame {frame_count} not in ground_truth.json for {domain_name}")
                        continue
                    
                    gt_str_line = []
                    gt = data[str(frame_count)]
                    for obj in gt:
                        try:
                            obj_type = obj['object type']
                            label = OBJECT_TYPES.index(obj_type)
                            obj_id = obj['object id']
                            location = obj['3d location']
                            scale = obj['3d bounding box scale']
                            rotation = obj['3d bounding box rotation']
                            gt_str_line.append(f'{label} {obj_id} {location[0]} {location[1]} {location[2]} {scale[0]} {scale[1]} {scale[2]} {rotation[0]} {rotation[1]} {rotation[2]}')
                        except (KeyError, ValueError, IndexError) as e:
                            print(f"Error processing object in frame {frame_count}: {e}")
                            continue

                    gt_txt_path = build_gt_output_path(out_dir, out_layout, split, domain_name, frame_count)
                    os.makedirs(os.path.dirname(gt_txt_path), exist_ok=True)
                    with open(gt_txt_path, 'w') as f:
                        for line in gt_str_line:
                            f.write(line + '\n')

if __name__ == "__main__":
    main()
