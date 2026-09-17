# Python pose graph optimization

The Python backend needs only the repository's NumPy, SciPy and Open3D dependencies. It consumes a completed `baseline` or `correspondence` result from any supported model.

For cross-modal bundle adjustment using point-cloud geometry, see the companion [LiDAR-VGGT-Bundle-Adjustment](https://github.com/NorwegianSmokedSalmon/LiDAR-VGGT-Bundle-Adjustment) repository.

## Run

```bash
python tools/run_pgo.py \
  --result-dir outputs/from_bag/HKairport03/pi3/correspondence \
  --config config/pgo.yaml --tune
```

When dense predictions have been cleaned, add `--regenerate-predictions`. This streams the original images through the same pinned model and retains compact PGO observations instead of full prediction arrays. Image and checkpoint checksums, preprocessing, chunk layout and model revision are checked before reuse.

Add `--prior-mode chunk_first` to fix the first camera of **every chunk** instead of only the sequence start. This preserves each chunk start in the initial LiDAR world and is the validated setting for AMvalley03. The default `sequence_first` reproduces the C++ anchoring policy.

Use `--cache-dir PATH` to share compact observations between parameter experiments or the two alignment methods. Use a new `--output-dir PATH` for each experiment. `--reuse` verifies and reuses an identical completed run; the bag entry point enables this automatically. `--prepare-only` builds inputs without optimizing. Compact input generation uses the GPU; registration and sparse graph optimization run on the CPU.

For a new reconstruction from a bag:

```bash
python tools/run_bag_pipeline.py /path/to/HKairport03.bag \
  --model pi3 --method correspondence --pgo --pgo-tune \
  --pgo-config config/pgo.yaml
```

Change `--model` to `vggt` or `vggt_omega`, or use `--method baseline` / `both`. PGO outputs are separate from the original merged map.

## Graph and coordinate conventions

- Each `(chunk, frame)` occurrence is a separate camera-pose node. An overlapping image therefore has nodes in multiple chunks.
- Existing LiDAR-to-model Sim(3) alignment supplies metric scale and the initial world pose. Scale is fixed throughout PGO; optimization variables are SE(3) poses.
- Consecutive frames **within each chunk** receive odometry factors from that model's decoded camera poses. There is no artificial odometry edge across a chunk boundary.
- Identical original-image pixels provide 3D correspondences between overlapping pointmaps. Preprocessing transforms map the same original pixels into each model input, including crop and padding.
- Correspondence RANSAC estimates a rigid alignment. Point-to-point ICP refines it using fitting points only. An ICP refinement that worsens the known-pixel median is discarded in favor of its RANSAC initialization.
- Accepted overlapping-image measurements become low-weight pose factors. Low-confidence pixels, depth discontinuities and unreliable overlap edges are rejected. A graph component without a fixed prior fails explicitly. In `chunk_first` mode, every chunk already has its own fixed anchor; rejected overlap edges remain visible in the report.
- The first camera of the entire sequence is fixed **exactly** by removing its six variables. It is the strongest prior, equivalent to infinite information. Other nodes remain free in `sequence_first` mode. `chunk_first` removes the six variables of every chunk-start camera.

`X_i` denotes `world_from_camera`. A factor `(i, j)` measures `Z_ij = inverse(X_i) @ X_j`, which maps camera `j` points into camera `i`. Rotation residuals use the SO(3) logarithm in radians; translation residuals are in meters in camera `i`. Camera-local pointmaps are transformed with the optimized pose, so the exported map actually reflects the optimized graph.

With only the first pose anchored, improved overlap does not guarantee that later chunks remain close to their original LiDAR world positions. No additional absolute pose, gravity or loop-closure constraint is silently introduced. The report includes world-position changes and odometry deviations alongside overlap residuals.

## Configuration and tuning

All settings are in [`config/pgo.yaml`](../config/pgo.yaml).

| Setting | Default | Meaning |
| --- | ---: | --- |
| `prior.mode` | `sequence_first` | Fix the sequence start; `chunk_first` fixes every chunk start |
| Fixed-pose prior | Fixed | Infinite weight; never optimized |
| `weights.odometry` | `10.0` | Model odometry information multiplier |
| `weights.overlap_icp` | `1.0` | Overlap information multiplier; must be below odometry |
| `noise.translation_sigma_m` | `0.2` | Common translation residual normalization |
| `noise.rotation_sigma_deg` | `1.0` | Common rotation residual normalization |
| `icp.ransac_threshold_m` | `3.0` | Known-pixel RANSAC inlier distance for these outdoor UAV scenes |
| `icp.max_correspondence_distance_m` | `1.0` | NN ICP correspondence distance after initialization |
| `icp.max_median_error_m` | `3.0` | Maximum fitting-pixel median for accepting an edge |
| `icp.min_pairs` / `min_inlier_ratio` | `80` / `0.25` | Minimum fitting consensus |
| `solver.icp_huber_delta` | `3.0` | Huber threshold on the whitened overlap-factor norm |
| `solver.max_nfev` | `150` | Maximum nonlinear iterations/evaluations per candidate |
| `solver.warm_start` | `true` | Rigidly connect chunks before per-frame optimization |
| `tuning.overlap_icp_weights` | `[0.1, 0.3, 1.0]` | ICP weights compared by `--tune`; odometry remains `10` |
| `observations.grid_side` | `64` | Original-image grid for overlap matching |
| `observations.export_sample_ratio` | `0.05` | Uniform per-frame map sampling |
| `export.voxel_size_m` | `0.1` | Final voxel map resolution |

The `model_overrides.vggt.icp` section sets VGGT RANSAC and median-error gates to `5 m`, reflecting the larger cross-chunk variation in its pointmap head. Pi3 and Omega retain the `3 m` gates. Every result saves the resolved configuration.

Weights multiply the squared, normalized residual: the residual multiplier is `sqrt(weight) / sigma`. The same normalizations apply to both factor types, so the information ordering is unambiguous: **fixed prior > odometry > ICP**. Huber robustification only further reduces the influence of inconsistent overlap factors.

Increase odometry weight or reduce ICP weight when local model motion deforms too much. Increase ICP weight, while keeping it below odometry, when reliable overlap measurements have too little influence. Adjust matching distances to scene depth and prediction noise; inspect rejected edges before changing quality gates. Changing observation settings requires a new cache directory; changing graph weights reuses the compact inputs.

## Validation and outputs

Original-pixel grid locations are deterministically split into **60% fitting, 20% tuning and 20% test**. RANSAC/ICP use fitting pixels only. `--tune` selects a converged candidate by tuning RMSE; test pixels are evaluated only after selection. If no converged candidate improves tuning RMSE, the original poses and geometry are retained (`applied: false`). Test data never decide this guard. Before/after evaluation uses the same pairs and includes observations from rejected edges. These are overlap consistency diagnostics, not ground-truth map accuracy. SE(3) cannot correct within-chunk scale inconsistencies, and improvement is not guaranteed at every boundary; per-boundary results are included.

The output directory contains:

- `merged_map.ply`, `merged_map_0.1m.ply` / `.pcd`: optimized colored maps, fully read back and checked.
- `map_preview.png`: before/after using the same sampled points and plot axes.
- `trajectory.tum`: one camera pose per original image, preferring the most central chunk occurrence.
- `trajectories/chunk_*.tum`: every optimized chunk camera trajectory.
- `graph.npz`, `nodes.json`: original/optimized poses, factor measurements and node identities.
- `overlap_edges.json`: accepted/rejected factors and fitting diagnostics.
- `tuning.json`, `report.json`, `README.md`, `config.yaml`: parameter selection, held-out results and reproducible settings.

Run regression tests with `python -m unittest discover -s tests`. PGO tests cover transform direction, metric scale, outliers, gauge fixing, information ordering, disconnected graphs and test-pixel separation.

To audit saved runs and build a comparison index, run `python tools/report_pgo_validation.py --root outputs/pgo`.
