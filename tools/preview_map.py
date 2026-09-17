"""Render a complete merged map, with a chunk coverage view, without a GUI."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    root = args.result_dir
    report = json.loads((root / 'merge_summary.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    rng = np.random.default_rng(0)
    xyz, rgb, ids, lidar = [], [], [], []
    for entry in report['per_chunk']:
        idx = entry['index']
        cloud = o3d.io.read_point_cloud(str(root / 'vggt_chunk_reg/final' / f'{idx}_pc.ply'))
        chosen = rng.choice(len(cloud.points), min(5000, len(cloud.points)), replace=False)
        xyz.append(np.asarray(cloud.points)[chosen])
        rgb.append(np.asarray(cloud.colors)[chosen])
        ids.append(np.full(len(chosen), idx))
        cloud = o3d.io.read_point_cloud(str(root / 'lidar_chunk' / f'chunk_{idx}.pcd'))
        chosen = rng.choice(len(cloud.points), min(3000, len(cloud.points)), replace=False)
        lidar.append(np.asarray(cloud.points)[chosen])
    xyz, rgb, ids, lidar = np.concatenate(xyz), np.concatenate(rgb), np.concatenate(ids), np.concatenate(lidar)
    # Use the two widest LiDAR world axes as the map plane; no point transform.
    reference_bounds = np.percentile(lidar, [0.5, 99.5], axis=0)
    dims = np.argsort(reference_bounds[1] - reference_bounds[0])[-2:][::-1]
    bounds = np.percentile(np.concatenate([xyz, lidar])[:, dims], [0.5, 99.5], axis=0)
    margin = (bounds[1] - bounds[0]) * 0.04
    bounds += [-margin, margin]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    axes[0].scatter(lidar[:, dims[0]], lidar[:, dims[1]], s=0.12, c='#63778a', rasterized=True)
    axes[0].set_title('LiDAR reference')
    axes[1].scatter(xyz[:, dims[0]], xyz[:, dims[1]], s=0.16, c=rgb, rasterized=True)
    axes[1].set_title('Merged colored map')
    plot = axes[2].scatter(xyz[:, dims[0]], xyz[:, dims[1]], s=0.16, c=ids, cmap='turbo', rasterized=True)
    axes[2].set_title('Chunk coverage')
    fig.colorbar(plot, ax=axes[2], label='Chunk index', shrink=0.65)
    for ax in axes:
        ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], xlabel='XYZ'[dims[0]] + ' (m)', ylabel='XYZ'[dims[1]] + ' (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.15)
    paired = summary.get('metric_type') == 'paired_correspondence'
    metric_label = 'held-out paired inliers' if paired else 'ICP fitness'
    fitness = np.array([m['validation']['inlier_ratio'] if paired else m['fitness'] for m in summary['registration']])
    rmse = np.array([m['validation']['inlier_rmse_m'] if paired else m['inlier_rmse'] for m in summary['registration']])
    method_label = "correspondence" if paired else "baseline"
    fig.suptitle(f"{summary.get('model', 'vggt')} / {method_label} | {Path(summary.get('dataset', root.name)).name} | {report['chunks']} chunks | {report['unique_images']:,} images | {report['merged_points']:,} colored points\n"
                 f"Sampled preview; 0.5–99.5% display bounds | mean {metric_label} {fitness.mean():.1%}, inlier RMSE {rmse.mean():.3f} m")
    path = root / 'map_preview.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
    axes[0].plot(np.arange(len(fitness)), fitness * 100, marker='.', lw=1)
    axes[0].set_ylabel(metric_label.capitalize() + ' (%)')
    axes[0].set_ylim(0, 100)
    axes[1].plot(np.arange(len(rmse)), rmse, marker='.', lw=1, color='#d86732')
    axes[1].set(xlabel='Chunk index', ylabel='Inlier RMSE (m)')
    for ax in axes:
        ax.grid(alpha=0.2)
    threshold = summary.get('residual_threshold_m', 2)
    fig.suptitle(f'Registration diagnostics ({threshold:g} m inlier threshold)')
    fig.savefig(root / 'registration_metrics.png', dpi=160)
    print(path)


if __name__ == '__main__':
    main()
