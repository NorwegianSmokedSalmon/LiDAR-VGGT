# Color-map evaluation

The standalone toolkit is maintained in [Color-Map-Evaluation](https://github.com/NorwegianSmokedSalmon/Color-Map-Evaluation). This directory retains the local evaluation commands and their regression tests.

The metrics follow [LiDAR-VGGT, Section III-D, Equations (16)–(18)](https://arxiv.org/abs/2511.01186): color distance/fidelity (CD/CF), local color recall (LCR), and color consistency score (CCS).

## Run

Inputs must be aligned colored PLY/PCD maps in meters. Open3D RGB values are interpreted in [0,1].

```bash
python evaluation/cal_colordist.py map.ply reference.ply --json outputs/evaluation/cd_cf.json
python evaluation/cal_lccr.py --map map.ply --truth reference.ply \
  --tau 0.1 --radius 0.5 --json outputs/evaluation/lcr.json
python evaluation/cal_cis.py --pcd map.ply --voxel-size 0.1 \
  --singleton-policy zero --json outputs/evaluation/ccs.json
```

## Conventions

- CD averages the two directed nearest-neighbor RGB distances with equal weights; CF is `-20 * log10(CD)`.
- LCR counts reference points with at least one reconstructed neighbor inside the geometric radius and RGB distance `<= 3 * tau`.
- CCS averages the trace of sample RGB covariance over occupied voxels, with equal voxel weights.

CCS defaults to a 0.1 m grid at the origin. `--singleton-policy zero|exclude|error` controls one-point voxels, whose sample covariance is undefined. These are explicit implementation settings; use the same settings when comparing maps. No hidden scaling is applied to reproduce reported table values.

JSON output records parameters, units, and point counts. The historical `cal_cis.py`, `compute_cis`, and `compute_LCCR` names remain supported for existing callers.

## Full evaluation and visualization

Use the standalone toolkit for JSON configurations and CD, LCR and CCS point-cloud
visualizations. For maps exported from a raw bag, estimate the evaluation frame
from the shared FAST-LIO geometry, then apply the same transform to each model:

```bash
python tools/export_lidar_map.py --frontend-dir data/AMtown02_raw/frontend \
  --output outputs/evaluation/AMtown02/lidar_world.ply
# In the installed Color-Map-Evaluation environment:
color-map-align --source /path/to/lidar_world.ply --reference /path/to/AMtown_gt.ply \
  --output-dir results/alignment
color-map-evaluate --map /path/to/merged_map.ply --reference /path/to/AMtown_gt.ply \
  --map-transform results/alignment/map_to_reference.txt \
  --visualization-dir results/run/visualizations --output results/run/metrics.json
```

Use the original `merged_map.ply`, before voxel color averaging. Inspect the saved
alignment overlay. LCR uses the full supplied reference, including missed regions.
Diagnostic display sampling does not alter the metric populations.
