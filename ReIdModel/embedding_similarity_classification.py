# match_with_avg_dist.py

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import glob
import argparse
from collections import defaultdict

from model import VoxelFeatureExtractor


def pcd_to_tensor(file_path_or_pcd, num_points=5000):
    if isinstance(file_path_or_pcd, str):
        pcd = o3d.io.read_point_cloud(file_path_or_pcd)
    else:
        pcd = file_path_or_pcd
    
    points_xyz = np.asarray(pcd.points, dtype=np.float32)
    if points_xyz.shape[0] == 0:
        return None
    xyz_min = points_xyz.min(axis=0)
    points_xyz -= xyz_min
    
    points_rgb = np.asarray(pcd.colors, dtype=np.float32)
    
    combined_features = np.concatenate((points_xyz, points_rgb), axis=1)
    if len(combined_features) < num_points:
        choice = np.random.choice(len(combined_features), num_points, replace=True)
    else:
        choice = np.random.choice(len(combined_features), num_points, replace=False)
    final_features = combined_features[choice, :]
    return torch.from_numpy(final_features)

def get_gallery_embeddings(gallery_dir,model,device):
    gallery_files = glob.glob(os.path.join(gallery_dir, '*.ply'))
    if not gallery_files:
        print(f"No .ply files found in {gallery_dir}")
        return None
    
    print(f"\nGenerating average gallery embedding from {len(gallery_files)} files...")
    gallery_embeddings_list = []
    for gallery_idx, gallery_path in enumerate(gallery_files):
        gallery_tensor = pcd_to_tensor(gallery_path)
        if gallery_tensor is not None:
            embedding = model([gallery_tensor.to(device)])
            gallery_embeddings_list.append(embedding)
    
    if not gallery_embeddings_list:
        print("Could not process any gallery files.")
        return None
    
    return torch.cat(gallery_embeddings_list, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Match gallery objects against an average distance to a gallery set.")
    parser.add_argument('--model_path', type=str, default="best_reid_model.pth", help="Path to the trained model weights.")
    parser.add_argument('--split_name', type=str, default='test', choices=['val', 'test'], help="Dataset split to process.")
    parser.add_argument('--det_root', type=str, default="/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025_outputs/detection", help="Root directory of the detection results.")
    parser.add_argument('--reid_feat_root', type=str, default="/perception/dataset/PhysicalAI-SmartSpaces/MTMC_Tracking_2025_outputs/reid_feat", help="Root directory of the ReID features.")

    parser.add_argument('--FourierGR1T2_dir', type=str, default="/perception/dataset/PhysicalAI-SmartSpaces/obj_crop_pcd_dataset/train/0_9998", help="Path to the FourierGR1T2 pcd directory.")
    parser.add_argument('--AgilityDigit_dir', type=str, default="/perception/dataset/PhysicalAI-SmartSpaces/obj_crop_pcd_dataset/train/0_9999", help="Path to the AgilityDigit pcd directory.")
    parser.add_argument('--scene_name', action='append', default=None, help="Optional scene filter. Can be passed multiple times.")
    
    args = parser.parse_args()

    gallery1_dir = args.FourierGR1T2_dir
    gallery2_dir = args.AgilityDigit_dir


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # phase 1
    model = VoxelFeatureExtractor().to(device)
    model.eval()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return

    with torch.no_grad():
        gallery1_embeddings = get_gallery_embeddings(gallery1_dir,model,device)
        gallery2_embeddings = get_gallery_embeddings(gallery2_dir,model,device)

    split_name = args.split_name
    det_input_dir = f'{args.det_root}/{split_name}_det_out'
    det_files = sorted(glob.glob(os.path.join(det_input_dir, '*.txt')))
    if args.scene_name:
        allowed_scenes = set(args.scene_name)
        det_files = [path for path in det_files if os.path.basename(path).rsplit('_', 1)[0] in allowed_scenes]
    scene_to_files = defaultdict(list)
    for det_file in det_files:
        scene_to_files[os.path.basename(det_file).rsplit('_', 1)[0]].append(det_file)
    scene_names = sorted(scene_to_files.keys())
    if not scene_names:
        raise FileNotFoundError(f"No detection files found in {det_input_dir} for split {split_name}")

    gallery1_neighbor_embeddings = []
    gallery2_neighbor_embeddings = []

    for scene_name in scene_names:
        sample_files = scene_to_files[scene_name][::max(1, len(scene_to_files[scene_name]) // 18 or 1)]
        for detection_txt_path in sample_files:
            frame_token = os.path.splitext(os.path.basename(detection_txt_path))[0].rsplit('_', 1)[1]
            gallery_feats = torch.load(f'{args.reid_feat_root}/{split_name}_reid_feat_out/{scene_name}_{frame_token}.pt')
            obj_keys = list(gallery_feats.keys())
            
            for obj_key in obj_keys:
                obj_feat = gallery_feats[obj_key]
                obj_feat = obj_feat.to(device)
                obj_row_idx = int(obj_key.split('_')[1])
                obj_feat = obj_feat.unsqueeze(0)

                g1_distance = torch.cdist(gallery1_embeddings, obj_feat, p=2).squeeze(1).mean().item()
                g2_distance = torch.cdist(gallery2_embeddings, obj_feat, p=2).squeeze(1).mean().item()

                if g1_distance < 0.3:
                    gallery1_neighbor_embeddings.append(obj_feat)
                if g2_distance < 0.3:
                    gallery2_neighbor_embeddings.append(obj_feat)
    
    gallery1_neighbor_embeddings = torch.stack(gallery1_neighbor_embeddings)
    
    gallery2_neighbor_embeddings = torch.stack(gallery2_neighbor_embeddings)
    #torch.save(gallery1_neighbor_embeddings, 'gallery1_neighbor_embeddings.pt')
    #torch.save(gallery2_neighbor_embeddings, 'gallery2_neighbor_embeddings.pt')

    output_dir = f'{args.det_root}/{split_name}_det_out_refine_cls'
    for scene_name in scene_names:
        for detection_txt_path in scene_to_files[scene_name]:
            frame_token = os.path.splitext(os.path.basename(detection_txt_path))[0].rsplit('_', 1)[1]
            gallery_feats = torch.load(f'{args.reid_feat_root}/{split_name}_reid_feat_out/{scene_name}_{frame_token}.pt')
            os.makedirs(output_dir, exist_ok=True)
            original_detections = np.loadtxt(detection_txt_path)
            obj_keys = list(gallery_feats.keys())
            
            g1_matched_row_indices = []
            g2_matched_row_indices = []
            for obj_key in obj_keys:
                obj_feat = gallery_feats[obj_key]
                obj_feat = obj_feat.to(device)
                obj_row_idx = int(obj_key.split('_')[1])
                obj_feat = obj_feat.unsqueeze(0)
                

                g1_distance = torch.cdist(gallery1_neighbor_embeddings, obj_feat, p=2).squeeze(1).min().item()
                g2_distance = torch.cdist(gallery2_neighbor_embeddings, obj_feat, p=2).squeeze(1).min().item()

                if g1_distance < 0.25:
                    g1_matched_row_indices.append(obj_row_idx)
                if g2_distance < 0.2:
                    g2_matched_row_indices.append(obj_row_idx)
                
            print(f"scene_name: {scene_name}, frame {frame_token}, Found {len(g1_matched_row_indices)} matches for query1, {len(g2_matched_row_indices)} matches for query2")

            modified_detections = original_detections.copy()
            for idx in g1_matched_row_indices:
                if  modified_detections[idx, 1] == 0:
                    modified_detections[idx, 1] = 4

            for idx in g2_matched_row_indices:
                if  modified_detections[idx, 1] == 0:
                    modified_detections[idx, 1] = 5
            
            output_filename = os.path.basename(detection_txt_path)
            output_path = os.path.join(output_dir, output_filename)

            fmt = ['%d', '%d', '%d', '%d'] + ['%.4f'] * 7 + ['%.4f']
            np.savetxt(output_path, modified_detections, fmt=fmt, delimiter=' ')
    

if __name__ == "__main__":
    main()
