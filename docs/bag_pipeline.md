# Reconstruction from ROS 1 bags

The bag entry point runs FAST-LIO, selects image keyframes, reconstructs chunks with a selected visual model, and merges the registered maps.

## Prerequisites

Install the Python environment using `bash tools/setup_env.sh`. Raw-bag processing also needs Docker and a local FAST-LIO image with per-scan export support. The default image tag is `lemon-mapping:fastlio`; override it with `--docker-image IMAGE`.

This image is an external prerequisite, not downloaded by the setup script. It must provide ROS 1 Noetic, the `fast_lio` package, and the `lemon_export` interface used by [`run_fastlio_ros.py`](../tools/bag/run_fastlio_ros.py): `pose.json`, `timestamps.txt`, `frames.json`, `progress.json`, and per-frame PCDs. The [`FAST-LIO config`](../config/bag/fastlio_avia.yaml) defines its export settings. The `lemon-mapping:noetic` image alone does not include FAST-LIO.

Host ROS is not required. The runner starts an isolated ROS master in the container and mounts the input bag read-only.

## Run

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python tools/run_bag_pipeline.py \
  /path/to/AMtown02.bag /path/to/HKisland03.bag \
  --model vggt --method both
```

| Option | Values / purpose |
| --- | --- |
| `--model` | `vggt`, `pi3`, `vggt_omega` (default: `pi3`) |
| `--method` | `baseline`, `correspondence`, `both` (default: `correspondence`) |
| `--pgo --pgo-tune` | Optimize overlapping chunks with the Python pose graph; tune ICP weight |
| `--pgo-config PATH` | PGO settings (default: `config/pgo.yaml`) |
| `--prepare-only` | Export scans, trajectory, and selected images without reconstruction |
| `--preset PATH` | Override automatic UAVScenes/Lighthouse calibration selection |
| `--parallax-px 12` | Minimum rotation-compensated image parallax at tracking resolution |
| `--start-offset S --end-offset S` | Set the image window relative to the preset clock (`bag_record` or `image_header`) |
| `--chunk-size 40 --chunk-step 25` | Reconstruction chunk length and spacing |
| `--max-chunks 2` | Limit reconstruction for a smoke run |
| `--dataset-root PATH --output-root PATH` | Choose separate inputs/results for another configuration |

`both` shares frontend data, selected images, and model inference between the two alignment methods. All selected images are reconstructed by default. Repeating an unchanged command reuses completed stages after checking their inputs and coverage. Use new dataset/output roots when changing settings.

## Calibration and keyframes

The supplied [`amtown.yaml`](../config/bag/amtown.yaml), [`amvalley.yaml`](../config/bag/amvalley.yaml), [`hkairport.yaml`](../config/bag/hkairport.yaml), and [`hkisland.yaml`](../config/bag/hkisland.yaml) presets define topics, camera intrinsics, distortion, extrinsics, timing, takeoff detection, and keyframe thresholds. For another sensor, copy and adapt a preset, then pass `--preset`.

The UAVScenes presets use intrinsics from [UAVScenes](https://github.com/sijieaaa/UAVScenes/blob/main/calibration_results.py) and extrinsics/timing from [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2/blob/main/config/MARS_LVIG.yaml). Their image timestamp correction is `header_stamp - 0.1 s`, applied once. HKairport03 follows the official scene mapping to `HK_GNSS_calibration`. The AMvalley frontend preset uses a 0.8 m blind range and waits for at least 200 initial map points to handle its initially empty/sparse scans.

FAST-LIO processes the bag from the beginning so IMU initialization remains available. Takeoff filtering applies only to image selection. When the height topic is unavailable or uninformative, takeoff detection uses the gravity-aligned LIO trajectory. Adjust the flight-height thresholds for your data or provide explicit time offsets.

The selector compensates camera rotation before measuring parallax. It uses LK tracking with SIFT/RANSAC recovery, a minimum camera translation, and a trajectory-coverage check. AMvalley inspects every 10 Hz image as a candidate to retain overlap over nearby terrain; the same parallax and translation thresholds still select keyframes. UAVScenes images retain their original compressed bytes; model preprocessing happens later. Fisheye inputs use the rectification path described below.

## MUN-FRL Lighthouse

Use the [MUN-FRL dataset](https://mun-frl-vil-dataset.readthedocs.io/en/latest/) **DJI M600 Lighthouse** calibration. The local workshop sample `lighthouse_francis_sample.bag` is approximately 181 seconds; it is a subset of the full Lighthouse sequence.

```bash
python tools/run_bag_pipeline.py /path/to/lighthouse_francis_sample.bag \
  --model vggt --method both
```

Names beginning with `lighthouse` select [`lighthouse.yaml`](../config/bag/lighthouse.yaml) automatically. For another filename, add `--preset config/bag/lighthouse.yaml`. All three model backends and both alignment methods use the same prepared dataset.

| Input | Setting |
| --- | --- |
| LiDAR | `/velodyne_points`, VLP-16, per-point time in seconds |
| IMU | `/imu/data`; LiDAR–IMU time offset `0.0034 s` |
| Camera | `/camera/image_color/compressed`, downward camera, 1440 × 1080 |
| Calibration | Published DJI M600 Kannala–Brandt intrinsics and camera-from-LiDAR transform |
| Image time | Original camera header, no extra offset |
| Selection offsets | Seconds from the first camera header (`selection_time_basis: image_header`) |
| Takeoff | Gravity-aligned FAST-LIO height; minimum 5 m, 3 s stable interval |

Camera parameters come from the [official calibration page](https://mun-frl-vil-dataset.readthedocs.io/en/latest/Calibration.html). LiDAR–IMU extrinsics and the FAST-LIO timing setting come from the [dataset FAST-LIO configuration](https://github.com/sendtooscar/FAST_LIO/blob/main/config/velodyne.yaml), reproduced in [`fastlio_lighthouse.yaml`](../config/bag/fastlio_lighthouse.yaml). The Bell 412 calibration belongs to other MUN-FRL sequences.

The workshop bag was re-recorded with a different recording clock from its sensor headers. FAST-LIO and image matching retain the original sensor timestamps; manual selection offsets follow the explicit preset clock. For example, `--start-offset 30 --end-offset 160` selects that interval relative to the first camera header. FAST-LIO still processes the raw bag from the beginning. Bag-provided estimator trajectories and registered clouds are not reconstruction inputs.

The published Kannala–Brandt model is rectified to a pinhole camera at the original image resolution (`rectification_balance: 0.0`). Selected source JPEGs remain byte-identical in `images_raw/`; lossless rectified PNGs in `images/` are used for tracking and reconstruction. The generated `projection.yaml` contains the **rectified intrinsics and zero distortion**, matching these PNGs. `selection.json` records the output calibration, original-image association, and timestamp convention. Validation checks both the original compressed bytes and every rectified image against the calibrated remapping.

The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please credit Thalagala et al., *MUN-FRL: A Visual-Inertial-LiDAR Dataset for Aerial Autonomous Navigation and Mapping*, IJRR, 2024 ([DOI](https://doi.org/10.1177/02783649241238358)), when using their data.

## Data and outputs

```text
data/<sequence>_raw/
  frontend/
    pcd/<frame_id>.pcd
    trajectory.tum
    pose.txt
  images/
  timestamps.txt
  dataset_config.yaml
  projection.yaml
  selection.json

outputs/from_bag/<sequence>/<model>/
  predictions/                 # Shared model inference
  baseline/
    merged_map.ply
    merged_map_0.1m.ply
    merged_map_0.1m.pcd
    map_preview.png
  correspondence/              # Same map exports
  methods_summary.json
```

Existing `npy/` prediction caches are also supported. FAST-LIO exports deskewed scans in the **IMU/body frame** and world-from-body poses at scan end. The runner composes camera–LiDAR and LiDAR–IMU calibration for projection. Indexed poses use `qw qx qy qz`; TUM uses `qx qy qz qw`.

## Validate a run

```bash
python tools/validate_bag_pipeline.py --bag /path/to/AMtown02.bag \
  --result-dir outputs/from_bag/AMtown02/vggt/correspondence
```

The validator checks scan/pose association, original image bytes, timestamps, complete chunk coverage, and exported XYZ/RGB data. To run every model with both methods:

```bash
set -e
for model in vggt pi3 vggt_omega; do
  python tools/run_bag_pipeline.py /path/to/AMtown02.bag /path/to/HKisland03.bag \
    --model "$model" --method both
  for sequence in AMtown02 HKisland03; do
    for method in baseline correspondence; do
      python tools/validate_bag_pipeline.py --bag "/path/to/$sequence.bag" \
        --result-dir "outputs/from_bag/$sequence/$model/$method"
    done
  done
done
python tools/report_model_validation.py --verify-weights
```

Run-specific diagnostics and validation reports stay in the output directories. They describe execution and registration consistency, not independent ground-truth map accuracy.

## Retaining final results only

After full validation, prediction arrays and per-chunk registration clouds can be removed while keeping merged maps, transforms, configuration, previews, and reports. A `retention.json` records this state and checksums of the retained maps. Such a result remains viewable; use a new `--output-root` when reconstructing again because its intermediate cache is no longer available.
