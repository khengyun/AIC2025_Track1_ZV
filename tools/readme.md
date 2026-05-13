
show video cam and label
```bash
python tools/view_ply_motion.py   --mode sequence   --folder /home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/pcd   --scene Warehouse_017   --voxel 0.05   --sleep 0.1   --max-files 100   --axis   --gt-json /home/niran/Documents/data/aic2025/test/Warehouse_017/ground_truth.json   --show-boxes   --show-label-text   --calibration-json /home/niran/Documents/data/aic2025/test/Lab_000/calibration.json   --show-cameras   --camera-scale 2.0   --camera-axis --skip-frames 29 --show-camera-ids 
```

save video 
```bash
python tools/view_ply_motion.py   --mode sequence   --folder /home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/pcd   --scene Lab_000   --voxel 0.05   --sleep 0.1   --max-files 100   --axis   --gt-json /home/niran/Documents/data/aic2025/test/Lab_000/ground_truth.json   --show-boxes   --show-label-text   --calibration-json /home/niran/Documents/data/aic2025/test/Lab_000/calibration.json   --show-cameras   --camera-scale 2.0   --camera-axis --skip-frames 29 --show-camera-ids --save-video lab_camera_ids.mp4
```

```
--camera-label-radius 0.01
--camera-label-size 0.15 
```
```bash
python tools/view_ply_motion.py   --mode sequence   --folder /home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/pcd   --scene Warehouse_017   --voxel 0.05   --sleep 0.1   --max-files 100   --axis    --show-boxes   --show-label-text   --calibration-json /home/niran/Documents/data/aic2025/test/Warehouse_017/calibration.json   --show-cameras   --camera-scale 2.0   --camera-axis --show-camera-ids 
```

```bash
python tools/view_dynamic_pcd.py \
  --processing-root "$BASE" \
  --data-root "$DATA" \
  --split train \
  --scene-name Warehouse_002 \
  --frame-id 0 \
  --static-voxel-size 0.05 \
  --compare-voxel-size 0.10 \
  --distance-threshold 0.05 \
  --show-static \
  --clip-z-max 3.0 \
  --clip-apply-to all \
  --show-gt \
  --gt-color-by-class \
  --gt-yaw-only \
  --viewer gui \
  --show-gt-id-labels \
  --gt-id-label-scale 1.2 \
  --gt-id-label-color 1,1,0 \
  --show-static-in-gt-boxes \
  --static-in-box-color 1,0,0 \
  --static-in-box-point-size 5.0 \
  --paint \
  --save
```

```bash
cd /home/niran/Documents/GitHub/AIC2025_Track1_ZV
conda activate aic2025-zv-py38

BASE=/home/niran/mnt/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026_processing
DATA=/home/niran/mnt/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026

python tools/generate_static_pcd.py \
  --processing-root "$BASE" \
  --data-root "$DATA" \
  --splits train \
  --scene-name Warehouse_000 \
  --source auto \
  --ensure-sampled-pcd \
  --frame-stride-static 100 \
  --voxel-size 0.05 \
  --merge-voxel-size 0.05 \
  --presence-ratio 0.7 \
  --min-ply-size-mb 0.1 \
  --exclude-labeled-objects \
  --gt-box-scale 1.1 \
  --gt-yaw-only \
  --overwrite
  --num-workers 2
```