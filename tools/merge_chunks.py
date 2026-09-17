"""Merge all registered chunks, already in the LiDAR world frame, into one map."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import open3d as o3d

VERTEX_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


def ply_header(count):
    return ("ply\nformat binary_little_endian 1.0\n"
            "comment coordinate_frame lidar_world\n"
            f"element vertex {count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()


def merge_result(result_dir, voxel_size=0.1):
    if not np.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError('voxel_size must be finite and nonnegative')
    started = time.perf_counter()
    root = Path(result_dir).resolve()
    summary = json.loads((root / 'summary.json').read_text())
    entries = sorted(summary['inference']['chunks'], key=lambda entry: entry['index'])
    indices = [entry['index'] for entry in entries]
    if not indices or indices != list(range(len(indices))):
        raise ValueError('Inference chunk indices must be contiguous and start at zero')
    expected_names = {f'{idx}_pc.ply' for idx in indices}
    reg_names = [item['vggt_file'] for item in summary['registration']]
    if len(reg_names) != len(indices) or set(reg_names) != expected_names:
        raise ValueError('Registration is incomplete or contains duplicate chunks')
    final_dir = root / 'vggt_chunk_reg/final'
    actual_names = {path.name for path in final_dir.glob('*.ply')}
    if actual_names != expected_names:
        raise ValueError(f'Final chunk mismatch: missing={expected_names - actual_names}, extra={actual_names - expected_names}')
    raw_path = root / 'merged_map.ply'
    voxel_label = f'{voxel_size:g}m'
    voxel_path = root / f'merged_map_{voxel_label}.ply'
    pcd_path = root / f'merged_map_{voxel_label}.pcd'
    report_path = root / 'merge_summary.json'
    paths = [raw_path, report_path] + ([voxel_path, pcd_path] if voxel_size else [])
    if any(path.exists() for path in paths):
        raise FileExistsError('Merged outputs already exist; use a new result directory or move them before rerunning')
    count, per_chunk = 0, []
    bounds = np.array([[np.inf] * 3, [-np.inf] * 3])
    # Stream one chunk at a time. Apply no additional transform: final/*.ply
    # already contains the complete registration into the shared world frame.
    with tempfile.TemporaryDirectory(prefix='.merge-', dir=root) as work:
        body_path = Path(work) / 'vertices.bin'
        with body_path.open('wb') as body:
            for idx in indices:
                path = final_dir / f'{idx}_pc.ply'
                cloud = o3d.io.read_point_cloud(str(path))
                points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
                if not len(points) or colors.shape != points.shape:
                    raise ValueError(f'Empty or uncolored chunk: {path}')
                if not np.isfinite(points).all() or not np.isfinite(colors).all():
                    raise ValueError(f'Non-finite points or colors: {path}')
                bounds[0] = np.minimum(bounds[0], points.min(axis=0))
                bounds[1] = np.maximum(bounds[1], points.max(axis=0))
                for start in range(0, len(points), 1000000):
                    pts, rgb = points[start:start + 1000000], colors[start:start + 1000000]
                    vertices = np.empty(len(pts), dtype=VERTEX_DTYPE)
                    for axis, name in enumerate(('x', 'y', 'z')):
                        vertices[name] = pts[:, axis]
                    rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    for channel, name in enumerate(('red', 'green', 'blue')):
                        vertices[name] = rgb[:, channel]
                    body.write(vertices.tobytes())
                count += len(points)
                per_chunk.append({'index': idx, 'points': len(points), 'file': str(path)})
                print(f'Merged chunk {idx + 1}/{len(indices)}: {len(points):,} points', flush=True)
        staged_raw = Path(work) / 'merged_map.ply'
        with staged_raw.open('wb') as output, body_path.open('rb') as body:
            output.write(ply_header(count))
            shutil.copyfileobj(body, output, length=16 * 1024 * 1024)
        expected_size = len(ply_header(count)) + count * VERTEX_DTYPE.itemsize
        if staged_raw.stat().st_size != expected_size:
            raise OSError('Merged PLY payload size does not match the declared point count')
        staged_raw.replace(raw_path)
    images = {image for entry in entries for image in entry['images']}
    report = {'coordinate_frame': 'lidar_world', 'chunks': len(indices),
              'unique_images': len(images), 'total_frame_occurrences': sum(len(e['images']) for e in entries),
              'input_points': count, 'merged_points': count, 'merged_file': str(raw_path),
              'bounds_min_max': bounds.tolist(), 'per_chunk': per_chunk}
    if voxel_size:
        print(f'Voxelizing the complete map at {voxel_size:g} m...', flush=True)
        cloud = o3d.io.read_point_cloud(str(raw_path))
        if len(cloud.points) != count:
            raise OSError('Merged PLY could not be read back completely')
        downsampled = cloud.voxel_down_sample(voxel_size)
        del cloud
        if not len(downsampled.points):
            raise ValueError('Voxelized map is empty')
        if not o3d.io.write_point_cloud(str(voxel_path), downsampled):
            raise OSError(f'Failed to write {voxel_path}')
        if not o3d.io.write_point_cloud(str(pcd_path), downsampled, compressed=True):
            raise OSError(f'Failed to write {pcd_path}')
        report.update(voxel_size=voxel_size, voxel_points=len(downsampled.points),
                      voxel_ply=str(voxel_path), voxel_pcd=str(pcd_path))
    report['seconds'] = time.perf_counter() - started
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Complete: {len(indices)} chunks, {len(images)} images, {count:,} points -> {raw_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path, help='Completed output from tools/run_pipeline.py')
    parser.add_argument('--voxel-size', type=float, default=0.1,
                        help='Also export a voxelized PLY/PCD; zero keeps only the original merged PLY')
    args = parser.parse_args()
    merge_result(args.result_dir, args.voxel_size)


if __name__ == '__main__':
    main()
