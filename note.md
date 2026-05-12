```bash
python generate_pcd_data.py   --data-root /media/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026  --out-dir /media/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026/point_cloud --splits train   --scene-name Warehouse_017  --frame-stride 1 

```
run with docker 

```bash
docker run --rm --gpus all \
  -v /media/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026:/data/MTMC_Tracking_2026 \
  -w /workspace/AIC2025_Track1_ZV \
  aic2025-full:latest \
  bash -lc "python tools/generate_pcd_data.py \
    --data-root /data/MTMC_Tracking_2026 \
    --out-dir /data/MTMC_Tracking_2026/point_cloud \
    --frame-stride 1 \
    --voxel-size 0.05"


docker run --rm -it  \
  -v /media/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026:/data/MTMC_Tracking_2026 \
  -w /workspace/AIC2025_Track1_ZV \
  aic2025-full:latest \
  bash -lc "python tools/generate_pcd_data.py \
    --data-root /data/MTMC_Tracking_2026 \
    --out-dir /data/MTMC_Tracking_2026/point_cloud \
    --frame-stride 1 \
    --voxel-size 0.05 \
    --overwrite \
    --num-workers 2"


```