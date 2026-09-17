"""LiDAR projection -> multi-model pixel correspondences -> RANSAC Sim3 -> colored map.

Runs model inference or reuses saved canonical chunks. The previous registration is applied only AFTER estimation,
solely for a held-out correspondence comparison; it never initializes the solver.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import open3d as o3d
from PIL import Image
import yaml

from Registration.projection_sim3 import (LidarTrajectory, preprocess_affine,
    error_metrics, project_visible, ransac_sim3, residuals, sample_pointmap)
from Utils.utils import save_ply, save_sim3_transforms
from tools.merge_chunks import merge_result
from models.predictions import load_prediction_manifest, prediction_directory
from models.inference import natural_key, run_inference, saved_model
from models.registry import MODEL_NAMES, model_spec, configured_checkpoint


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def intrinsics_for_image(options, predicted_k, image_path, image_hw, pixel_transform=None):
    source = options['intrinsics_source']
    if source in ('vggt', 'model'):
        # VGGT intrinsics are already expressed on the pointmap pixel grid.
        return predicted_k, None
    if source != 'calibration':
        raise ValueError('intrinsics_source must be model (legacy: vggt) or calibration')
    if options.get('camera_matrix') is None or options.get('camera_image_size') is None:
        raise ValueError('Calibrated projection requires camera_matrix and camera_image_size')
    k = np.asarray(options['camera_matrix'], dtype=float).reshape(3, 3)
    with Image.open(image_path) as image:
        original_wh = image.size
    calibrated_wh = tuple(options['camera_image_size'])
    if calibrated_wh != original_wh:
        if not options.get('allow_intrinsics_rescale', False):
            raise ValueError(f'Calibration resolution {calibrated_wh} differs from image {original_wh}')
        sx, sy = np.asarray(original_wh) / np.asarray(calibrated_wh)
        if not np.isclose(sx, sy, rtol=1e-5):
            raise ValueError('Calibration rescaling requires the same image aspect ratio')
        k = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]]) @ k
    affine = (preprocess_affine(original_wh, image_hw, options['preprocess_mode']) if pixel_transform is None
              else np.asarray(pixel_transform, dtype=float))
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise ValueError('Invalid saved original-image to pointmap pixel transform')
    distortion = None if options['images_undistorted'] else np.asarray(options['distortion'], dtype=float)
    return affine @ k, distortion


def build_pairs(predictions, entry, dataset, data, options, timestamps, trajectory, rcl, tcl):
    sources, targets, pixels, frame_ids, validation, all_world = [], [], [], [], [], []
    stats = []
    rng = np.random.default_rng(int(options['seed']) + entry['index'])
    n, height, width, _ = predictions['world_points'].shape
    if n != len(entry['images']) or predictions['intrinsic'].shape != (n, 3, 3):
        raise ValueError('Prediction shapes do not match the inference manifest')
    for frame, name in enumerate(entry['images']):
        stamp = timestamps[Path(name).name]
        scan_idx = trajectory.closest(stamp, float(options['time_tolerance_s']))
        scan_id = int(trajectory.rows[scan_idx, 0])
        path = dataset / data['body_pcd_dir'] / f'{scan_id}.pcd'
        if not path.is_file():
            raise FileNotFoundError(path)
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        points = points[np.isfinite(points).all(axis=1)]
        world, camera = trajectory.world_and_camera(points, scan_idx, stamp, rcl, tcl,
                                                    bool(options['motion_compensation']))
        all_world.append(world)
        pixel_transform = predictions.get('image_pixel_transform')
        k, distortion = intrinsics_for_image(options, predictions['intrinsic'][frame], name, (height, width),
                                             None if pixel_transform is None else pixel_transform[frame])
        indices, uv = project_visible(camera, k, (height, width), distortion,
                                     float(options['min_depth_m']), float(options['max_depth_m']))
        if 'image_valid_bounds' in predictions:
            left, top, right, bottom = predictions['image_valid_bounds'][frame]
            observed = (uv[:, 0] >= left) & (uv[:, 0] < right) & (uv[:, 1] >= top) & (uv[:, 1] < bottom)
            indices, uv = indices[observed], uv[observed]
        conf = predictions['world_points_conf'][frame]
        finite_conf = conf[np.isfinite(conf)]
        if not len(finite_conf):
            raise ValueError(f'No finite confidence in {name}')
        conf_threshold = max(float(options['min_confidence']),
                             float(np.percentile(finite_conf, float(options['confidence_percentile']))))
        source, valid = sample_pointmap(predictions['world_points'][frame], conf,
                                        predictions['depth'][frame], uv, conf_threshold,
                                        float(options['max_relative_depth_jump']))
        target, uv = world[indices[valid]], uv[valid]
        candidates = len(source)
        take = rng.choice(candidates, min(candidates, int(options['max_pairs_per_frame'])), replace=False)
        source, target, uv = source[take], target[take], uv[take]
        test = np.zeros(len(source), dtype=bool)
        if len(source) >= 5:
            test[rng.choice(len(source), max(1, int(len(source) * options['validation_fraction'])), replace=False)] = True
        sources.append(source)
        targets.append(target)
        pixels.append(uv)
        frame_ids.append(np.full(len(source), frame, dtype=np.int32))
        validation.append(test)
        stats.append({'frame': frame, 'image': name, 'scan_id': scan_id,
                      'scan_time_difference_s': float(trajectory.times[scan_idx] - stamp),
                      'lidar_points': len(points), 'visible_pixels': len(indices),
                      'filtered_pairs': candidates, 'selected_pairs': len(source),
                      'confidence_threshold': conf_threshold, 'intrinsic': np.asarray(k).tolist()})
    return (np.concatenate(sources), np.concatenate(targets), np.concatenate(pixels),
            np.concatenate(frame_ids), np.concatenate(validation), stats, np.concatenate(all_world))


def export_registered(predictions, matrix, path, sample_ratio, confidence_threshold, seed):
    """Uniform per-frame sampling; pointmap points are in the chunk world frame."""
    rng = np.random.default_rng(seed)
    points, colors = [], []
    for xyz, conf, image in zip(predictions['world_points'], predictions['world_points_conf'], predictions['images']):
        xyz = xyz.reshape(-1, 3)
        rgb = np.rint(np.clip(image.transpose(1, 2, 0).reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
        valid = np.flatnonzero(np.isfinite(xyz).all(axis=1) & np.isfinite(conf.reshape(-1)) & (conf.reshape(-1) >= confidence_threshold) & (conf.reshape(-1) > 0))
        count = int(len(valid) * sample_ratio)
        chosen = rng.choice(valid, count, replace=False)
        points.append(xyz[chosen] @ matrix[:3, :3].T + matrix[:3, 3])
        colors.append(rgb[chosen])
    points, colors = np.concatenate(points), np.concatenate(colors)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Registered output is empty or non-finite')
    save_ply(points, colors, str(path))
    return len(points)


def projection_preview(predictions, uv, frames, errors, threshold, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = sorted(set([0, len(predictions['images']) // 2, len(predictions['images']) - 1]))
    fig, axes = plt.subplots(1, len(chosen), figsize=(6 * len(chosen), 5), squeeze=False, constrained_layout=True)
    for ax, frame in zip(axes[0], chosen):
        mask = frames == frame
        ax.imshow(predictions['images'][frame].transpose(1, 2, 0))
        ax.scatter(uv[mask, 0], uv[mask, 1], c=np.where(errors[mask] <= threshold, '#24d65b', '#ff3636'), s=2)
        ax.set_title(f'Frame {frame}: green = paired residual <= {threshold:g} m')
        ax.axis('off')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/projection_amtown.yaml',
                        help='Projection config; default uses the public AMtown camera preset')
    parser.add_argument('--model', choices=MODEL_NAMES, help='Default: vggt for new inference, or model recorded in cached predictions')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--chunk-step', type=int)
    parser.add_argument('--all-images', action='store_true', help='Infer every dataset image; default fresh run is one chunk')
    parser.add_argument('--prediction-dir', type=Path, help='Reuse existing canonical predictions instead of running inference')
    parser.add_argument('--baseline-dir', type=Path, help='Optional original registration for held-out comparison only')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-chunks', type=int, help='Default: one chunk for fresh inference, all chunks for cached predictions')
    parser.add_argument('--intrinsics-source', choices=['model', 'vggt', 'calibration'])
    parser.add_argument('--sample-ratio', type=float)
    parser.add_argument('--min-confidence', type=float)
    parser.add_argument('--export-confidence', type=float)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge-voxel-size', type=float, default=0.1)
    args = parser.parse_args()
    if args.all_images and args.max_chunks is not None:
        parser.error('Choose all-images or max-chunks, not both')
    started = time.perf_counter()
    document = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load(project_path(document.get('base_config', 'config/config.yaml')).read_text())
    cfg['Projection'] = options = document['Projection']
    if args.intrinsics_source:
        options['intrinsics_source'] = args.intrinsics_source
    sample_ratio = options['export_sample_ratio'] if args.sample_ratio is None else args.sample_ratio
    if not 0 < sample_ratio <= 1 or not 0 < options['validation_fraction'] < 1 or not 0 <= options['confidence_percentile'] <= 100:
        parser.error('Invalid point sampling, validation split, or confidence percentile')
    if options['max_pairs_per_frame'] < 5 or options['min_inlier_frames'] < 1:
        parser.error('Require at least five pairs per frame and one inlier frame')
    if not 0 <= options['min_depth_m'] < options['max_depth_m'] or options['time_tolerance_s'] < 0:
        parser.error('Invalid depth range or scan time tolerance')
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error('max-chunks must be positive')
    if not np.isfinite(args.merge_voxel_size) or args.merge_voxel_size < 0:
        parser.error('merge-voxel-size must be finite and nonnegative')
    data = cfg['Data']
    dataset = (project_path(data['base_path']) / data['dataset']).resolve()
    output = (args.output_dir or PROJECT_ROOT / 'outputs' / datetime.now().strftime('AMtown01_projection_%Y%m%d_%H%M%S')).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {output}')
    if args.prediction_dir:
        if args.checkpoint or args.chunk_size or args.chunk_step:
            parser.error('checkpoint/chunk-size/chunk-step apply only to fresh inference')
        prediction_dir = args.prediction_dir.resolve()
        expected_images = sorted((p for p in (dataset / data['images_dir']).iterdir()
                                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), key=natural_key)
        expected_images = expected_images[int(cfg['Chunk'].get('start_idx', 0)):]
        manifest = load_prediction_manifest(prediction_dir, expected_images, args.model,
            max_chunks=args.max_chunks, require_all=args.all_images)
        model_name = saved_model(manifest, args.model)
    else:
        model_name = args.model or 'vggt'
        size = args.chunk_size if args.chunk_size is not None else int(cfg['Chunk']['chunk_size'])
        step = args.chunk_step if args.chunk_step is not None else min(size, int(cfg['Chunk']['chunk_step']))
        if not 0 < step <= size:
            parser.error('Require 0 < chunk-step <= chunk-size')
        images = sorted((p for p in (dataset / data['images_dir']).iterdir()
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), key=natural_key)
        images = images[int(cfg['Chunk'].get('start_idx', 0)):]
        prediction_dir = output / 'npy'
        manifest = run_inference(images, prediction_dir, configured_checkpoint(model_name, cfg, args.checkpoint), size, size-step,
                                 None if args.all_images else (args.max_chunks or 1), args.device, model_name)
    spec = model_spec(model_name)
    if args.min_confidence is not None:
        options['min_confidence'] = args.min_confidence
    if options.get('min_confidence') is None:
        options['min_confidence'] = spec['default_min_confidence']
    export_confidence = args.export_confidence if args.export_confidence is not None else options.get('export_confidence')
    if export_confidence is None:
        export_confidence = spec['default_min_confidence']
    if not np.isfinite([options['min_confidence'], export_confidence]).all() or min(options['min_confidence'], export_confidence) < 0:
        parser.error('Confidence thresholds must be finite and nonnegative')
    options['export_confidence'] = export_confidence
    cfg['Reconstruction'] = {'model': model_name, 'confidence_type': spec['confidence']}
    if model_name != 'vggt' and manifest.get('schema_version', 0) < 2:
        raise ValueError('Non-VGGT predictions require schema v2 with explicit pixel transforms')
    entries = manifest['chunks'][:args.max_chunks]
    if not entries or [e['index'] for e in entries] != list(range(len(entries))):
        raise ValueError('Expected a contiguous prefix of inference chunks starting at zero')
    manifest = dict(manifest, chunks=entries)
    baseline = None
    if args.baseline_dir:
        baseline_dir = args.baseline_dir.resolve()
        if prediction_directory(baseline_dir) != prediction_dir:
            raise ValueError('Baseline comparison requires the exact same VGGT prediction directory')
        baseline = np.loadtxt(baseline_dir / 'sim3_total.txt').reshape(-1, 4, 4)
        if len(baseline) < len(entries) or not np.isfinite(baseline).all():
            raise ValueError('Invalid or incomplete baseline transforms')
    trajectory = LidarTrajectory(np.loadtxt(dataset / data['lio_pose_file']))
    timestamps = {}
    for line in (dataset / data['img_time_file']).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, stamp = line.split()[:2]
            timestamps[name] = float(stamp)
    rcl = np.asarray(cfg['calibration']['R']).reshape(3, 3)
    tcl = np.asarray(cfg['calibration']['t'], dtype=float)
    if tcl.shape != (3,) or not np.isfinite(tcl).all():
        raise ValueError('calibration.t must contain three finite values')
    if not np.allclose(rcl.T @ rcl, np.eye(3), atol=1e-3) or np.linalg.det(rcl) < 0:
        raise ValueError('calibration.R must be a proper LiDAR-to-camera rotation')
    if options.get('calibration_source'):
        print(f"Camera calibration source: {options['calibration_source']}", flush=True)
        print(options.get('calibration_note', ''), flush=True)
    if options['intrinsics_source'] in ('vggt', 'model'):
        print(f'Projection uses {model_name} intrinsics (approximate); calibrated intrinsics are preferred.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ['vggt_chunk_reg/final', 'correspondences', 'lidar_chunk', 'diagnostics']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    cfg['Chunk'].update(chunk_num=len(entries), chunk_size=manifest['chunk_size'],
                        chunk_step=manifest['chunk_size'] - manifest['overlap'], sample_ratio=sample_ratio)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    threshold = float(options['ransac_threshold_m'])
    summary = {'method': 'projection_ransac_sim3', 'metric_type': 'paired_correspondence',
               'dataset': str(dataset), 'output': str(output), 'prediction_dir': str(prediction_dir),
               'model': model_name, 'inference_reused': bool(args.prediction_dir), 'inference': manifest, 'intrinsics_source': options['intrinsics_source'],
               'residual_threshold_m': threshold, 'registration': []}
    transforms = []
    for entry in entries:
        tick = time.perf_counter()
        idx = entry['index']
        predictions = np.load(prediction_dir / entry['file'], allow_pickle=True).item()
        if model_name != 'vggt' and 'image_pixel_transform' not in predictions:
            raise ValueError('Missing original-image to pointmap pixel transform for this model')
        source, target, uv, frames, validation, frame_stats, lidar_world = build_pairs(
            predictions, entry, dataset, data, options, timestamps, trajectory, rcl, tcl)
        print(f'Chunk {idx + 1}/{len(entries)}: {len(source):,} pixel pairs, {validation.sum():,} held out', flush=True)
        result = ransac_sim3(source[~validation], target[~validation], threshold,
            int(options['ransac_max_iterations']), float(options['ransac_confidence']),
            int(options['ransac_min_inliers']), float(options['ransac_min_inlier_ratio']),
            int(options['ransac_scoring_points']), int(options['seed']) + idx)
        errors = residuals(source, target, result.scale, result.rotation, result.translation)
        consensus_frames = np.unique(frames[(errors <= threshold) & ~validation])
        if len(consensus_frames) < int(options['min_inlier_frames']):
            raise ValueError(f'Chunk {idx} consensus covers only {len(consensus_frames)} frames')
        train_metrics = error_metrics(errors[~validation], threshold)
        heldout_metrics = error_metrics(errors[validation], threshold)
        matrix = result.matrix
        output_path = output / 'vggt_chunk_reg/final' / f'{idx}_pc.ply'
        points = export_registered(predictions, matrix, output_path, sample_ratio,
                                   export_confidence, int(options['seed']) + idx)
        np.savez_compressed(output / 'correspondences' / f'chunk_{idx}.npz',
            source_vggt=source.astype(np.float32), target_lidar_world=target, uv=uv.astype(np.float32),
            frame=frames, validation=validation, inlier=errors <= threshold, residual_m=errors.astype(np.float32))
        lidar_path = output / 'lidar_chunk' / f'chunk_{idx}.pcd'
        # Reuse the verified reference cloud only when it accompanies these same predictions.
        existing_lidar = prediction_dir.parent / 'lidar_chunk' / lidar_path.name
        if existing_lidar.is_file():
            lidar_path.symlink_to(existing_lidar.resolve())
        else:
            lidar = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_world))
            lidar = lidar.voxel_down_sample(float(cfg['Downsample']['lidar_voxel_size']))
            if not o3d.io.write_point_cloud(str(lidar_path), lidar):
                raise OSError(f'Could not write {lidar_path}')
        metric = {'index': idx, 'vggt_file': output_path.name, 'output': str(output_path),
                  'method': 'projection_ransac_sim3', 'points': points, 'correspondences': len(source),
                  'inlier_frames': len(consensus_frames), 'ransac_iterations': result.iterations,
                  'scale': result.scale, 'transform': matrix.tolist(), 'train': train_metrics,
                  'validation': heldout_metrics, 'frames': frame_stats,
                  'fitness': train_metrics['inlier_ratio'], 'inlier_rmse': train_metrics['inlier_rmse_m']}
        if baseline is not None:
            old_points = source @ baseline[idx, :3, :3].T + baseline[idx, :3, 3]
            old_errors = np.linalg.norm(old_points - target, axis=1)
            metric['baseline_validation'] = error_metrics(old_errors[validation], threshold)
        metric['seconds'] = time.perf_counter() - tick
        summary['registration'].append(metric)
        transforms.append(matrix)
        save_sim3_transforms(transforms, str(output / 'sim3_total.txt'))
        summary['seconds'] = time.perf_counter() - started
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(f'  scale={result.scale:.4f}, train inliers={train_metrics["inlier_ratio"]:.1%}, '
              f'held-out median={heldout_metrics["median_m"]:.3f}m, {metric["seconds"]:.1f}s', flush=True)
        if idx in (0, len(entries) - 1):
            projection_preview(predictions, uv, frames, errors, threshold, output / 'diagnostics' / f'projection_{idx}.png')
        del predictions, source, target, uv, frames, validation, lidar_world
    if args.merge:
        merge_result(output, args.merge_voxel_size)
    print(f'Completed correspondence-based registration: {output}', flush=True)


if __name__ == '__main__':
    main()
