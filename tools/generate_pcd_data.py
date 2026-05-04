import os
import cv2
import numpy as np
import h5py
import json
import open3d as o3d
import time

import argparse
from typing import Dict, List, Tuple, Optional



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate point clouds (PLY) from multi-camera RGB + depth sequences."
    )
    parser.add_argument("--data-root", type=str, default="dataset/MTMC_Tracking_2025",
                        help="Dataset root directory.")
    parser.add_argument("--out-dir", type=str, default="dataset/pcd_dataset",
                        help="Output root directory for generated PLY files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Splits to process (space-separated). e.g., --splits train val")
    parser.add_argument("--scene-name", action="append", default=None,
                        help="Optional scene name filter. Can be passed multiple times.")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help="Optional maximum duration to process per scene.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Dataset FPS used to convert seconds into frame indices.")
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="Optional frame stride. Defaults to 100 for train/val and 1 for test.")
    parser.add_argument("--max-cameras", type=int, default=None,
                        help="Optional maximum number of cameras to use per scene.")
    
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

def get_point_cloud(rgb_image_per_camera, depth_image_per_camera, camera_params):
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
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)

    return pcd_combined


def main():
    args = parse_args()

    data_root = args.data_root
    out_dir = args.out_dir

    split_set = args.splits
    scene_filter = set(args.scene_name) if args.scene_name else None

    for split in split_set:
        domain_name_list = sorted(os.listdir(os.path.join(data_root, split)))
        if scene_filter is not None:
            domain_name_list = [name for name in domain_name_list if name in scene_filter]
        frame_indices = get_selected_frame_indices(split, args.max_seconds, args.fps, args.frame_stride)
        for domain_idx, domain_name in enumerate(domain_name_list):
            domain_path = os.path.join(data_root, split, domain_name)
            print(f'Processing {domain_path}...')
            camera_params = get_calibration_per_camera(os.path.join(domain_path, 'calibration.json'))
            camera_name_list = sorted([name.split('.')[0] for name in os.listdir(os.path.join(domain_path,'videos'))])
            if args.max_cameras is not None:
                camera_name_list = camera_name_list[:args.max_cameras]
            video_path_list = [os.path.join(domain_path, 'videos', f'{name}.mp4') for name in camera_name_list]
            depth_map_path_list = [os.path.join(domain_path, 'depth_maps', f'{name}.h5') for name in camera_name_list]

            camera_name_list = [normalize_camera_name(camera_name) for camera_name in camera_name_list]

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
                    print(f"Stopping {domain_name} at frame {frame_count} because video frames are unavailable.")
                    break

                output_path = f'{out_dir}/{split}/pcd/{domain_name}_{frame_count:05d}.ply'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                start_time = time.time()
                if not os.path.exists(output_path):
                    with open(output_path, 'w') as f:
                        f.write(f"# {domain_name} frame {frame_count}\n")

                    pcd = get_point_cloud(rgb_image_per_camera, depth_image_per_camera, camera_params)
                    
                    o3d.io.write_point_cloud(output_path, pcd)

                print(f"{output_path} Point cloud generated in {time.time() - start_time:.2f} seconds.")
            for video in video_capture_per_camera.values():
                video.release()
            
            # Close h5py files to prevent resource leak
            for depth_map in depth_map_per_camera.values():
                if depth_map is not None:
                    depth_map.close()

    OBJECT_TYPES = ['Person', 'Forklift', 'NovaCarter', 'Transporter', 'FourierGR1T2', 'AgilityDigit']
    for split in split_set:
        if split == 'test':
            continue
        domain_name_list = os.listdir(os.path.join(data_root, split))
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

                gt_txt_path = f'{out_dir}/{split}/gt/{domain_name}_{frame_count:05d}.txt'
                os.makedirs(os.path.dirname(gt_txt_path), exist_ok=True)
                with open(gt_txt_path, 'w') as f:
                    for line in gt_str_line:
                        f.write(line + '\n')

if __name__ == "__main__":
    main()
