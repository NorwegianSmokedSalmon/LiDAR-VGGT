# Reconstruction from prepared data

Run commands from the repository root with the environment activated. Use `--model vggt`, `--model pi3`, or `--model vggt_omega` with either pipeline.

## Input layout

Set `Data` in [`config/config.yaml`](../config/config.yaml) to your dataset:

```text
Data.base_path/Data.dataset/
  data/images/          # JPEG or PNG images, naturally sorted by filename
  data/pcd/             # One body-frame point cloud per frame: <frame_id>.pcd
  data/timestamps.txt   # image_filename timestamp
  data/pose.txt         # frame_id timestamp x y z qw qx qy qz
```

`Data.base_path` is resolved relative to the repository root. Other paths are relative to the dataset directory. Coordinates are in meters and timestamps in seconds. Poses describe **world-from-body** transforms; `calibration.R` and `calibration.t` describe **camera-from-body**. The calibration frame must match the frame of the exported scans.

The indexed pose file uses **qw qx qy qz**, whereas standard TUM files use **qx qy qz qw**. Convert external TUM trajectories to the indexed format before running these entry points.

## Original pipeline

```bash
python tools/run_pipeline.py --config config/config.yaml \
  --model vggt --all-images --merge --output-dir outputs/baseline
```

Trajectory alignment initializes each chunk, followed by regularized point-cloud Sim(3) and ICP. Settings are grouped under `Chunk`, `pose_sim3`, `PCD_sim3`, `Regularization`, and `Downsample` in the config. `--refine-line` enables attitude refinement for nearly straight trajectories.

## Correspondence pipeline

```bash
python tools/run_projection_pipeline.py --config config/projection_amtown.yaml \
  --model vggt --all-images --merge --output-dir outputs/correspondence
```

Copy the projection config for your sensor. Set `base_config` to the dataset config, then provide `camera_matrix`, `camera_image_size`, `distortion`, and `images_undistorted` under `Projection`. Use `intrinsics_source: calibration` for calibrated projection. The supplied AMtown preset is specific to that camera.

LiDAR points are projected onto the original image and mapped to network pixels using the model's saved preprocessing transform. RANSAC rejects incorrect pairs before the final Sim(3) fit.

## Common options

| Option | Purpose |
| --- | --- |
| `--all-images` | Reconstruct the full sequence, including the final partial chunk |
| `--chunk-size 40 --chunk-step 25` | Set chunk length and spacing |
| `--max-chunks 2` | Limit reconstruction for a smoke run |
| `--sample-ratio 0.05` | Export a smaller fraction of valid points |
| `--prediction-dir PATH` | Reuse saved model predictions |
| `--checkpoint PATH` | Override weights for new inference |
| `--device auto\|cuda\|cpu` | Select the inference device |
| `--merge --merge-voxel-size 0.1` | Export merged and voxel-downsampled maps |

Fresh inference processes one chunk unless `--all-images` or `--max-chunks` is specified. Use a separate output directory for each model, method, or parameter setting.

## Shared predictions

Run inference once and use the same cache for both alignment methods:

```bash
python tools/run_inference.py --model vggt \
  --image-dir data/AMtown01/data/images --output-dir outputs/predictions

python tools/run_pipeline.py --config config/config.yaml --model vggt \
  --prediction-dir outputs/predictions --all-images --merge --output-dir outputs/baseline

python tools/run_projection_pipeline.py --config config/projection_amtown.yaml --model vggt \
  --prediction-dir outputs/predictions --all-images --merge --output-dir outputs/correspondence
```

The prediction manifest records the model, image order, and chunk boundaries. Both methods check this information before reuse.

## Outputs and inspection

Each reconstruction directory contains per-chunk maps in `vggt_chunk_reg/final/`, transforms in `sim3_total.txt`, and merged maps in `merged_map.ply` and `merged_map_0.1m.ply/.pcd`. The historical `vggt_chunk_reg` directory name is shared by all three models.

```bash
python tools/preview_map.py outputs/correspondence
python tools/preview_result.py outputs/baseline --chunk 0
python tools/merge_chunks.py --help
```

`summary.json` and `merge_summary.json` record the run configuration and chunk coverage. For overlap refinement, run the [Python PGO backend](../PGO/README.md) on a completed result. The bag pipeline enables it with `--pgo`; prepared-data reconstruction uses `tools/run_pgo.py`. For cross-modal bundle adjustment, see [LiDAR-VGGT-Bundle-Adjustment](https://github.com/NorwegianSmokedSalmon/LiDAR-VGGT-Bundle-Adjustment).
