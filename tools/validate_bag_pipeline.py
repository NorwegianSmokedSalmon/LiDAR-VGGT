"""Read back a raw-bag result, including every PCD, saved image and merged map."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import open3d as o3d
from tools.bag.reader import bag_identity, iter_images
from tools.bag.images import BagImageProcessor
from tools.extract_bag_keyframes import validate_coverage


def require(condition, message):
    if not condition:
        raise ValueError(message)



def registration_metrics(summary):
    """Keep nearest-neighbor baseline diagnostics distinct from held-out pixel pairs."""
    if summary.get('metric_type') == 'paired_correspondence':
        stats = [item['validation'] for item in summary['registration']]
        pairs, inliers = sum(v['pairs'] for v in stats), sum(v['inliers'] for v in stats)
        return {'metric_type': 'paired_correspondence', 'held_out_pairs': pairs,
                'held_out_inlier_ratio': inliers/pairs,
                'held_out_inlier_rmse_m': float(np.sqrt(sum(v['inlier_rmse_m']**2*v['inliers'] for v in stats)/inliers)),
                'worst_chunk_held_out_median_m': max(v['median_m'] for v in stats),
                'metric_note': 'Pixel correspondence residuals in the FAST-LIO world; not independent map accuracy.'}
    stats = summary['registration']
    return {'metric_type': 'nearest_neighbor_icp',
            'mean_icp_fitness': float(np.mean([item['fitness'] for item in stats])),
            'mean_icp_inlier_rmse_m': float(np.mean([item['inlier_rmse'] for item in stats])),
            'metric_note': 'Nearest-neighbor ICP diagnostics; do not compare directly with pixel correspondence residuals.'}


def validate(bag, result):
    started = time.perf_counter()
    run = json.loads((result/'bag_run.json').read_text())
    require(run['complete'] and run['full_sequence'], 'Expected a completed full-sequence run')
    require(run['request']['bag'] == bag_identity(bag), 'Bag identity differs from reconstruction')
    dataset = Path(run['dataset'])
    selection = json.loads((dataset/'selection.json').read_text())
    require(hashlib.sha256((dataset/'selection.json').read_bytes()).hexdigest() == run['request']['selection_sha256'],
            'Selection changed after reconstruction')
    rows = np.loadtxt(dataset/'frontend/pose.txt', ndmin=2)
    tum = np.loadtxt(dataset/'frontend/trajectory.tum', ndmin=2)
    require(rows.shape == (run['frontend_scans'], 9), 'Wrong indexed pose count')
    require(np.isfinite(rows).all() and np.all(np.diff(rows[:, 1]) > 0), 'Invalid pose values/timestamps')
    require(np.allclose(np.linalg.norm(rows[:, 5:9], axis=1), 1, atol=1e-5), 'Invalid pose quaternions')
    expected_tum = rows[:, [1, 2, 3, 4, 6, 7, 8, 5]]
    require(tum.shape == expected_tum.shape and np.allclose(tum, expected_tum, rtol=0, atol=1e-8),
            'TUM timestamps, positions or xyzw quaternions disagree with indexed poses')
    files = list((dataset/'frontend/pcd').glob('*.pcd'))
    require(len(files) == len(rows), 'Point cloud / pose counts differ')
    scan_points = 0
    for frame in rows[:, 0].astype(int):
        cloud = o3d.io.read_point_cloud(str(dataset/'frontend/pcd'/f'{frame}.pcd'))
        points = np.asarray(cloud.points)
        require(len(points) > 0 and np.isfinite(points).all(), f'Empty or invalid PCD {frame}')
        scan_points += len(points)
    print(f'Validated {len(rows)} body scans and TUM poses', flush=True)
    entries = {entry['bag_index']: entry for entry in selection['selected']}
    require(len(entries) == selection['selected_images'], 'Duplicate selected bag indices')
    checked = 0
    config = selection['request']['config']
    processor = BagImageProcessor(config)
    for index, header, _, _, payload in iter_images(bag, config['image_topic']):
        if index not in entries:
            continue
        entry = entries[index]
        saved = (dataset/'images'/entry['image']).read_bytes()
        original = (dataset/entry['original_image']).read_bytes() if processor.rectified else saved
        require(original == payload.tobytes(), f'Saved original differs from bag payload: {index}')
        if processor.rectified:
            decoded = cv2.imdecode(np.frombuffer(saved, np.uint8), cv2.IMREAD_COLOR)
            require(np.array_equal(decoded, processor.decode(payload)),
                    f'Rectified image differs from calibrated remapping: {index}')
        require(abs(header+config['image_time_offset_s']-entry['timestamp']) < 1e-6,
                f'Incorrect image timestamp offset: {index}')
        require(abs(rows[entry['scan_id'], 1]-entry['timestamp']) <= .06, f'Incorrect scan match: {index}')
        checked += 1
    require(checked == len(entries), 'Selected image not present in bag')
    stamps = np.loadtxt(dataset/'timestamps.txt', dtype=str, ndmin=2)
    require(stamps.shape == (checked, 2), 'Wrong image timestamp count')
    require(stamps[:, 0].tolist() == [e['image'] for e in selection['selected']], 'Wrong timestamp image names')
    require(np.allclose(stamps[:, 1].astype(float), [e['timestamp'] for e in selection['selected']], rtol=0, atol=1e-6),
            'Timestamp file differs from selection')
    print(f'Validated {checked} original image payloads and timestamps', flush=True)
    coverage = validate_coverage(dataset, selection)
    merge = json.loads((result/'merge_summary.json').read_text())
    summary = json.loads((result/'summary.json').read_text())
    require(merge['unique_images'] == checked and merge['chunks'] == len(summary['registration']) == run['chunks'],
            'Reconstruction is missing images or chunks')
    maps = []
    for key, count_key in [('merged_file', 'merged_points'), ('voxel_ply', 'voxel_points'), ('voxel_pcd', 'voxel_points')]:
        if not merge.get(key):
            continue
        path = Path(merge[key])
        cloud = o3d.io.read_point_cloud(str(path))
        xyz, rgb = np.asarray(cloud.points), np.asarray(cloud.colors)
        require(len(xyz) == merge[count_key] and len(xyz) > 0, f'Map point count mismatch: {path}')
        require(rgb.shape == xyz.shape and np.isfinite(xyz).all() and np.isfinite(rgb).all(), f'Invalid map: {path}')
        require(np.min(rgb) >= 0 and np.max(rgb) <= 1, f'Invalid colors: {path}')
        maps.append({'file': path.name, 'points': len(xyz), 'finite_xyz_rgb': True})
        del xyz, rgb, cloud
        print(f'Validated {path.name}', flush=True)
    report = {'complete': True, 'bag': bag_identity(bag), 'frontend_scans': len(rows), 'frontend_points': scan_points,
              'original_image_payloads_checked': checked,
              'rectified_images_checked': checked if processor.rectified else 0, 'coverage': coverage, 'chunks': merge['chunks'], 'maps': maps,
              **registration_metrics(summary), 'seconds': time.perf_counter()-started}
    (result/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    args = parser.parse_args()
    validate(args.bag.resolve(strict=True), args.result_dir.resolve(strict=True))


if __name__ == '__main__':
    main()
