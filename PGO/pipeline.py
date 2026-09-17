"""Build, tune, validate and export a frame-level cross-chunk pose graph."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import yaml

from PGO.geometry import correspondence_icp, invert, metric_camera_poses, transform
from PGO.graph import check_connected, edge_errors, optimize, warm_start
from PGO.inputs import digest, ensure_inputs
from tools.merge_chunks import VERTEX_DTYPE, ply_header


def resolve_config(config, model):
    resolved = deepcopy(config)
    resolved.setdefault('prior', {'mode': 'sequence_first'})
    overrides = resolved.pop('model_overrides', {})
    for section, values in overrides.get(model, {}).items():
        if section not in resolved or not isinstance(values, dict) or not set(values) <= set(resolved[section]):
            raise ValueError(f'Unknown model override section/option: {section}')
        resolved[section].update(values)
    return resolved


def validate_config(config):
    if 'model_overrides' in config:
        for model in [None, *config['model_overrides']]:
            validate_config(resolve_config(config, model))
        return
    expected = {'prior', 'weights', 'noise', 'observations', 'icp', 'solver', 'tuning', 'export'}
    if set(config) != expected:
        raise ValueError(f'PGO configuration requires exactly {sorted(expected)}')
    if config['prior']['mode'] not in ('sequence_first', 'chunk_first'):
        raise ValueError('prior.mode must be sequence_first or chunk_first')
    weights = config['weights']
    candidates = [weights['overlap_icp'], *config['tuning']['overlap_icp_weights']]
    if not np.isfinite([weights['odometry'], *candidates]).all() or not all(0 < w < weights['odometry'] for w in candidates):
        raise ValueError('Require fixed first pose > odometry weight > every overlap ICP weight > 0')
    for section, keys in [('noise', ['translation_sigma_m', 'rotation_sigma_deg']),
                          ('icp', ['ransac_threshold_m', 'max_correspondence_distance_m', 'max_median_error_m']),
                          ('solver', ['ftol', 'icp_huber_delta']), ('export', ['voxel_size_m'])]:
        if any(not np.isfinite(config[section][k]) or config[section][k] <= 0 for k in keys):
            raise ValueError(f'Expected positive finite values in {section}')
    for section, key, minimum in [('observations', 'grid_side', 8), ('icp', 'min_pairs', 3),
                                  ('icp', 'ransac_iterations', 1), ('icp', 'iterations', 1),
                                  ('solver', 'max_nfev', 1)]:
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{section}.{key} must be an integer >= {minimum}')
    obs = config['observations']
    if not (0 <= obs['confidence_percentile'] < 100 and 0 < obs['export_sample_ratio'] <= 1
            and 0 <= obs['min_depth_m'] < obs['max_depth_m']
            and 0 < config['icp']['min_inlier_ratio'] <= 1):
        raise ValueError('Invalid sampling, confidence, depth, or consensus options')


def build_graph(cache, summary, transforms, config, diagnostics_path=None):
    entries = summary['inference']['chunks']
    nodes, poses, scales, overlap_points, overlap_valid = [], [], [], [], []
    occurrences = defaultdict(list)
    sources, targets, measurements, kinds, diagnostics, pairs = [], [], [], [], [], []
    for entry, sim3 in zip(entries, transforms):
        idx = entry['index']
        with np.load(cache/f'chunk_{idx}.npz', allow_pickle=False) as compact:
            cameras, scale = metric_camera_poses(compact['extrinsic'], sim3)
            points = compact['overlap_points'].astype(float) * scale
            valid = compact['overlap_valid'].copy()
        valid &= np.isfinite(points).all(2) & (points[:, :, 2] >= config['observations']['min_depth_m'])
        valid &= points[:, :, 2] <= config['observations']['max_depth_m']
        start = len(nodes)
        for frame, (image, pose) in enumerate(zip(entry['images'], cameras)):
            node = len(nodes)
            nodes.append({'id': node, 'chunk': idx, 'frame': frame, 'image': image,
                          'image_index': entry['start']+frame, 'scale': scale})
            poses.append(pose)
            scales.append(scale)
            overlap_points.append(points[frame])
            overlap_valid.append(valid[frame])
            occurrences[image].append(node)
            if frame:
                sources.append(node-1); targets.append(node)
                measurements.append(invert(cameras[frame-1]) @ cameras[frame])
                kinds.append('odom')
        assert len(nodes) - start == len(entry['images'])
    initial = np.array(poses)
    seed = int(config['observations']['seed'])
    for image, ids in occurrences.items():
        if len(ids) < 2:
            continue
        size = len(overlap_valid[ids[0]])
        rng = np.random.default_rng(seed+nodes[ids[0]]['image_index'])
        labels = np.empty(size, np.uint8)
        labels[rng.permutation(size)] = np.arange(size) % 5  # 0: test, 1: tune, 2-4: fit
        for position, i in enumerate(ids):
            for j in ids[position+1:]:
                common = overlap_valid[i] & overlap_valid[j]
                x, y, split = overlap_points[i][common], overlap_points[j][common], labels[common]
                record = {'i': i, 'j': j, 'chunk_i': nodes[i]['chunk'], 'chunk_j': nodes[j]['chunk'],
                          'image': image, 'valid_pairs': int(common.sum()), 'accepted': False}
                # Include rejected-edge observations in evaluation: no favourable
                # post-optimization correspondence filtering or rematching.
                if len(x):
                    pairs.append({'i': i, 'j': j, 'x': x, 'y': y, 'split': split,
                                  'boundary': (nodes[i]['chunk'], nodes[j]['chunk'])})
                try:
                    fitting = split >= 2
                    measurement, quality = correspondence_icp(y[fitting], x[fitting], config['icp'], seed+i*10007+j)
                except ValueError as error:
                    record['reason'] = str(error)
                else:
                    sources.append(i); targets.append(j); measurements.append(measurement); kinds.append('icp')
                    record.update(accepted=True, **quality)
                diagnostics.append(record)
    sources, targets, kinds = np.array(sources), np.array(targets), np.array(kinds)
    if diagnostics_path is not None:
        Path(diagnostics_path).write_text(json.dumps(diagnostics, indent=2)+'\n')
    fixed = [n['id'] for n in nodes if n['frame'] == 0] if config['prior']['mode'] == 'chunk_first' else [0]
    components = check_connected(len(nodes), sources, targets, fixed)
    print(f'Graph: {len(nodes)} poses, {(kinds == "odom").sum()} odometry, '
          f'{(kinds == "icp").sum()}/{len(diagnostics)} accepted overlap factors', flush=True)
    return {'initial': initial, 'sources': sources, 'targets': targets, 'measurements': np.array(measurements),
            'components': components, 'kinds': kinds, 'nodes': nodes, 'pairs': pairs, 'diagnostics': diagnostics,
            'chunk_ids': np.array([n['chunk'] for n in nodes])}


def statistics(values):
    values = np.asarray(values)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty/non-finite validation observations')
    return {'count': len(values), 'median': float(np.median(values)),
            'rmse': float(np.sqrt(np.mean(values**2))), 'p95': float(np.percentile(values, 95)),
            'max': float(values.max())}


def overlap_metrics(poses, pairs, split):
    per_boundary = defaultdict(list)
    for pair in pairs:
        selected = pair['split'] == split
        x = transform(pair['x'][selected], poses[pair['i']])
        y = transform(pair['y'][selected], poses[pair['j']])
        per_boundary[pair['boundary']].append(np.linalg.norm(x-y, axis=1))
    combined = {key: np.concatenate(value) for key, value in per_boundary.items()}
    return {'overall_m': statistics(np.concatenate(list(combined.values()))),
            'boundaries': [{'chunks': list(key), **statistics(value)} for key, value in sorted(combined.items())]}


def odometry_metrics(graph, poses):
    mask = graph['kinds'] == 'odom'
    residual = edge_errors(poses, graph['sources'][mask], graph['targets'][mask], graph['measurements'][mask])
    return {'translation_m': statistics(np.linalg.norm(residual[:, 3:], axis=1)),
            'rotation_deg': statistics(np.rad2deg(np.linalg.norm(residual[:, :3], axis=1)))}


def select_candidate(trials, tuning_before):
    eligible = [i for i, trial in enumerate(trials) if trial['eligible']]
    if not eligible:
        raise RuntimeError('No candidate solver converged; see tuning.json and inspect overlap factors')
    choice = min(eligible, key=lambda i: trials[i]['tuning_overlap']['overall_m']['rmse'])
    applied = trials[choice]['tuning_overlap']['overall_m']['rmse'] < tuning_before['overall_m']['rmse']
    return choice, bool(applied)


def export_maps(cache, summary, graph, optimized, output, voxel_size):
    entries = summary['inference']['chunks']
    total = 0
    for entry in entries:
        with np.load(cache/f'chunk_{entry["index"]}.npz') as compact:
            total += int(compact['sample_offsets'][-1])
    path = output/'merged_map.ply'
    before_preview, after_preview, rgb_preview = [], [], []
    node = 0
    with path.open('wb') as stream:
        stream.write(ply_header(total))
        for entry in entries:
            with np.load(cache/f'chunk_{entry["index"]}.npz') as compact:
                points, colors, offsets = compact['sample_points'], compact['sample_colors'], compact['sample_offsets']
            for frame in range(len(entry['images'])):
                a, b = offsets[frame:frame+2]
                local = points[a:b].astype(float) * graph['nodes'][node]['scale']
                world = transform(local, optimized[node])
                vertices = np.empty(len(world), dtype=VERTEX_DTYPE)
                for axis, key in enumerate(('x', 'y', 'z')):
                    vertices[key] = world[:, axis]
                for channel, key in enumerate(('red', 'green', 'blue')):
                    vertices[key] = colors[a:b, channel]
                stream.write(vertices.tobytes())
                take = np.linspace(0, len(local)-1, min(200, len(local))).astype(int)
                before_preview.append(transform(local[take], graph['initial'][node]))
                after_preview.append(world[take]); rgb_preview.append(colors[a:b][take]/255.)
                node += 1
    if path.stat().st_size != len(ply_header(total)) + total*VERTEX_DTYPE.itemsize:
        raise ValueError('Merged PLY byte count differs from declared point count')
    maps = []
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) != total:
        raise ValueError('Merged map did not read back completely')
    down = cloud.voxel_down_sample(voxel_size)
    for name in (f'merged_map_{voxel_size:g}m.ply', f'merged_map_{voxel_size:g}m.pcd'):
        if not o3d.io.write_point_cloud(str(output/name), down, compressed=True):
            raise OSError(f'Failed to write {name}')
    del cloud, down
    for map_path in [path, output/f'merged_map_{voxel_size:g}m.ply', output/f'merged_map_{voxel_size:g}m.pcd']:
        checked = o3d.io.read_point_cloud(str(map_path))
        xyz, rgb = np.asarray(checked.points), np.asarray(checked.colors)
        if not len(xyz) or xyz.shape != rgb.shape or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
            raise ValueError(f'Invalid XYZ/RGB in {map_path}')
        maps.append({'file': map_path.name, 'points': len(xyz), 'finite_xyz_rgb': True, 'sha256': digest(map_path)})
    draw_preview(np.concatenate(before_preview), np.concatenate(after_preview), np.concatenate(rgb_preview), graph, optimized, output)
    return maps


def draw_preview(before, after, colors, graph, poses, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dims = np.argsort(np.ptp(graph['initial'][:, :3, 3], axis=0))[-2:][::-1]
    bounds = np.percentile(np.concatenate([before, after])[:, dims], [.5, 99.5], axis=0)
    margin = (bounds[1]-bounds[0]) * .04
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, points, title in zip(axes, (before, after), ('Before PGO', 'After PGO')):
        ax.scatter(points[:, dims[0]], points[:, dims[1]], c=colors, s=.2, rasterized=True)
        ax.set(xlim=(bounds[0, 0]-margin[0], bounds[1, 0]+margin[0]),
               ylim=(bounds[0, 1]-margin[1], bounds[1, 1]+margin[1]), title=title,
               xlabel='XYZ'[dims[0]]+' (m)', ylabel='XYZ'[dims[1]]+' (m)')
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.suptitle('Same metric point samples; SE(3) corrections, fixed first camera and chunk scales')
    fig.savefig(output/'map_preview.png', dpi=150); plt.close(fig)


def write_trajectories(graph, optimized, summary, output):
    source = Path(summary['dataset'])
    result_config = yaml.safe_load((Path(summary['output'])/'config.yaml').read_text())
    timestamps = {}
    for line in (source/result_config['Data']['img_time_file']).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, stamp = line.split()[:2]; timestamps[name] = float(stamp)
    folder = output/'trajectories'
    folder.mkdir()
    unique = {}
    for chunk in sorted(set(graph['chunk_ids'])):
        selected = np.flatnonzero(graph['chunk_ids'] == chunk)
        rows = []
        for node in selected:
            metadata = graph['nodes'][node]
            stamp = timestamps[Path(metadata['image']).name]
            pose = optimized[node]
            row = np.r_[stamp, pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
            rows.append(row)
            centrality = min(metadata['frame'], len(selected)-1-metadata['frame'])
            index = metadata['image_index']
            if index not in unique or centrality > unique[index][0]:
                unique[index] = (centrality, row)
        np.savetxt(folder/f'chunk_{chunk}.tum', rows, fmt='%.10f', header='timestamp tx ty tz qx qy qz qw; world_from_camera')
    np.savetxt(output/'trajectory.tum', [unique[k][1] for k in sorted(unique)], fmt='%.10f',
               header='world_from_camera; one pose per original image, prefer central chunk occurrence')


def run_pgo(result_dir, output_dir, cache_dir, config, *, regenerate=False, tune=False, device='auto', prepare_only=False, reuse=False):
    if 'prior' not in config:
        config = dict(config, prior={'mode': 'sequence_first'})
    validate_config(config)
    model = json.loads((Path(result_dir)/'summary.json').read_text())['model']
    config = resolve_config(config, model)
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    existing = not prepare_only and output.exists() and any(output.iterdir())
    if existing and not (reuse and (output/'report.json').is_file()):
        raise FileExistsError(f'Choose an empty PGO output directory: {output}')
    cache, summary, transforms = ensure_inputs(result_dir, cache_dir, config, regenerate, device)
    if prepare_only:
        return {'cache': str(cache), 'complete': True}
    if existing:
        previous = json.loads((output/'report.json').read_text())
        compatible = (previous.get('complete')
                      and previous.get('source_result') == str(Path(result_dir).resolve())
                      and yaml.safe_load((output/'config.yaml').read_text()) == config
                      and previous.get('tuning_enabled') == bool(tune)
                      and previous.get('source_sim3_sha256') == digest(Path(result_dir)/'sim3_total.txt')
                      and previous.get('input_manifest_sha256') == digest(cache/'inputs.json'))
        if not compatible or any(digest(output/m['file']) != m['sha256'] for m in previous['maps']):
            raise ValueError('Completed PGO settings/inputs/maps changed; choose a new output directory')
        print(f'Reusing validated PGO result: {output}', flush=True)
        return previous
    summary = dict(summary, output=str(Path(result_dir).resolve()))

    output.mkdir(parents=True, exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    graph = build_graph(cache, summary, transforms, config, output/'overlap_edges.json')
    initial = graph['initial']
    start = warm_start(initial, graph['chunk_ids'], graph['sources'], graph['targets'], graph['measurements'], graph['kinds']) if config['solver']['warm_start'] and config['prior']['mode'] == 'sequence_first' else initial
    fixed = ([n['id'] for n in graph['nodes'] if n['frame'] == 0]
             if config['prior']['mode'] == 'chunk_first' else [0])
    candidates = config['tuning']['overlap_icp_weights'] if tune else [config['weights']['overlap_icp']]
    trials, solutions = [], []
    for weight in candidates:
        print(f'Optimizing with odometry={config["weights"]["odometry"]:g}, ICP={weight:g}, first pose fixed', flush=True)
        poses, info = optimize(initial, graph['sources'], graph['targets'], graph['measurements'], graph['kinds'], config, weight, start, fixed_nodes=fixed)
        info['tuning_overlap'] = overlap_metrics(poses, graph['pairs'], 1)
        info['odometry'] = odometry_metrics(graph, poses)
        info['eligible'] = (info['converged'] and info['final_cost'] <= info['initial_cost']+1e-8)
        trials.append(info); solutions.append(poses)
        print(f'  converged={info["converged"]}, tuning RMSE={info["tuning_overlap"]["overall_m"]["rmse"]:.4f} m, nfev={info["nfev"]}', flush=True)
    (output/'tuning.json').write_text(json.dumps(trials, indent=2, allow_nan=False)+'\n')
    tuning_before = overlap_metrics(initial, graph['pairs'], 1)
    choice, applied = select_candidate(trials, tuning_before)
    optimized = solutions[choice] if applied else initial.copy()
    if not applied:
        print('PGO candidate did not improve tuning RMSE; retaining initial poses and map geometry', flush=True)
    before, after = overlap_metrics(initial, graph['pairs'], 0), overlap_metrics(optimized, graph['pairs'], 0)
    # Test pixels are evaluated only after parameter selection is complete.
    improved = after['overall_m']['rmse'] < before['overall_m']['rmse']
    np.savez_compressed(output/'graph.npz', initial=initial, optimized=optimized, sources=graph['sources'],
                        targets=graph['targets'], measurements=graph['measurements'], kinds=graph['kinds'], chunk_ids=graph['chunk_ids'])
    (output/'nodes.json').write_text(json.dumps(graph['nodes'], indent=2)+'\n')
    (output/'overlap_edges.json').write_text(json.dumps(graph['diagnostics'], indent=2)+'\n')
    write_trajectories(graph, optimized, summary, output)
    maps = export_maps(cache, summary, graph, optimized, output, config['export']['voxel_size_m'])
    report = {'complete': True, 'source_result': str(Path(result_dir).resolve()), 'source_method': summary['method'],
              'tuning_enabled': bool(tune), 'model': summary['model'], 'dataset': summary['dataset'], 'compact_inputs': str(cache),
              'source_sim3_sha256': digest(Path(result_dir)/'sim3_total.txt'), 'input_manifest_sha256': digest(cache/'inputs.json'),
              'chunks': len(summary['inference']['chunks']), 'images': len({n['image'] for n in graph['nodes']}),
              'nodes': len(initial), 'odometry_edges': int(np.sum(graph['kinds']=='odom')),
              'overlap_edges': int(np.sum(graph['kinds']=='icp')), 'rejected_overlap_edges': sum(not e['accepted'] for e in graph['diagnostics']),
              'selected': trials[choice], 'odometry_after': odometry_metrics(graph, optimized), 'applied': applied, 'tuning_before': tuning_before,
              'graph_components': graph['components'], 'first_pose_fixed': bool(np.array_equal(initial[0], optimized[0])),
              'prior_mode': config['prior']['mode'], 'fixed_nodes': fixed,
              'anchors_fixed': bool(np.array_equal(initial[fixed], optimized[fixed])),
              'chunk_scales_fixed': True, 'test_before': before, 'test_after': after, 'test_overlap_improved': bool(improved),
              'node_translation_change_m': statistics(np.linalg.norm(optimized[:, :3, 3]-initial[:, :3, 3], axis=1)),
              'maps': maps, 'seconds': time.perf_counter()-started,
              'metric_note': 'Same held-out original-image pixels before/after, including rejected-edge observations. '
                             '60% fit, 20% tune, 20% test. No ground-truth map accuracy claim; see prior_mode for LiDAR-world anchoring.'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_report(report, output)
    print(f'Complete: test overlap RMSE {before["overall_m"]["rmse"]:.4f} -> {after["overall_m"]["rmse"]:.4f} m; {output}', flush=True)
    return report


def write_report(report, output):
    before, after = report['test_before']['overall_m'], report['test_after']['overall_m']
    lines = ['# Python PGO result', '', f'Model: {report["model"]}; source: `{report["source_result"]}`.', '',
             f'PGO applied: {report.get("applied", True)}. ', '',
             f'{report["chunks"]} chunks, {report["images"]} unique images, {report["nodes"]} frame-occurrence nodes.', '',
             f'Prior mode: {report.get("prior_mode", "sequence_first")}; anchors and chunk scales fixed. Odometry weight {report["selected"]["odometry_weight"]:g}, '
             f'overlap ICP weight {report["selected"]["icp_weight"]:g}.', '',
             '| Held-out overlap | Before (m) | After (m) |', '| --- | ---: | ---: |']
    for key in ('median', 'rmse', 'p95'):
        lines.append(f'| {key} | {before[key]:.4f} | {after[key]:.4f} |')
    lines += ['', report['metric_note'], '', '![Before and after](map_preview.png)', '',
              'Final maps: `merged_map.ply` and voxelized PLY/PCD. All three fully read back with finite XYZ/RGB.',
              'Camera trajectories: `trajectory.tum`, `trajectories/chunk_*.tum`; all graph poses in `graph.npz`.',
              'Diagnostics: `report.json`, `overlap_edges.json`, `tuning.json`, `nodes.json`, `config.yaml`.', '']
    (output/'README.md').write_text('\n'.join(lines))
