"""Save a sampled 3D comparison of a pipeline result without a GUI."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--chunk', type=int, default=0)
    args = parser.parse_args()
    root, idx = args.result_dir, args.chunk
    clouds = [o3d.io.read_point_cloud(str(path)) for path in (
        root / 'lidar_chunk' / f'chunk_{idx}.pcd',
        root / 'vggt_chunk' / f'{idx}_pc.ply',
        root / 'vggt_chunk_reg' / 'final' / f'{idx}_pc.ply')]
    if any(not len(cloud.points) for cloud in clouds):
        raise ValueError('Missing or empty result point cloud')
    rng = np.random.default_rng(0)
    percentiles = [np.percentile(np.asarray(c.points), [1, 99], axis=0) for c in clouds]
    bounds = np.array([np.min([p[0] for p in percentiles], axis=0),
                       np.max([p[1] for p in percentiles], axis=0)])
    margin = np.maximum((bounds[1] - bounds[0]) * 0.05, 0.1)
    bounds += np.stack([-margin, margin])
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    for col, (cloud, title) in enumerate(zip(clouds, ['LiDAR reference', 'After pose Sim3', 'After point Sim3 + ICP'])):
        points = np.asarray(cloud.points)
        chosen = rng.choice(len(points), min(40000, len(points)), replace=False)
        points = points[chosen]
        colors = np.asarray(cloud.colors)[chosen] if cloud.has_colors() else '#52637a'
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        ax.scatter(*points.T, c=colors, s=0.35, depthshade=False, rasterized=True)
        ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2],
               xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
        ax.set_box_aspect(bounds[1] - bounds[0])
        ax.view_init(elev=18, azim=15)
        ax.tick_params(labelsize=8)
    fig.suptitle(f'Chunk {idx}: 40k sampled points per panel, shared 1–99% bounds')
    path = root / f'preview_{idx}.png'
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == '__main__':
    main()
