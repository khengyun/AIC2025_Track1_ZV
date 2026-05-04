
show video cam and label
```bash
python tools/view_ply_motion.py   --mode sequence   --folder /home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/pcd   --scene Lab_000   --voxel 0.05   --sleep 0.1   --max-files 100   --axis   --gt-json /home/niran/Documents/data/aic2025/test/Lab_000/ground_truth.json   --show-boxes   --show-label-text   --calibration-json /home/niran/Documents/data/aic2025/test/Lab_000/calibration.json   --show-cameras   --camera-scale 2.0   --camera-axis --skip-frames 29 --show-camera-ids 
```

save video 
```bash
python tools/view_ply_motion.py   --mode sequence   --folder /home/niran/Documents/GitHub/AIC2025_Track1_ZV/dataset/pcd_dataset/test/pcd   --scene Lab_000   --voxel 0.05   --sleep 0.1   --max-files 100   --axis   --gt-json /home/niran/Documents/data/aic2025/test/Lab_000/ground_truth.json   --show-boxes   --show-label-text   --calibration-json /home/niran/Documents/data/aic2025/test/Lab_000/calibration.json   --show-cameras   --camera-scale 2.0   --camera-axis --skip-frames 29 --show-camera-ids --save-video lab_camera_ids.mp4
```

```
--camera-label-radius 0.01
--camera-label-size 0.15 
```