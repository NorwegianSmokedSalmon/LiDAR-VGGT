"""Export a sparse FAST-LIO world map for independent evaluation-frame alignment."""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-step', type=int, default=30)
    parser.add_argument('--voxel-size', type=float, default=.5)
    args = parser.parse_args()
    if args.frame_step < 1 or not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error('frame-step and voxel-size must be positive')
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    poses = np.loadtxt(args.frontend_dir/'pose.txt', ndmin=2)
    if poses.shape[1] != 9 or not np.isfinite(poses).all():
        parser.error('Expected pose.txt columns: frame_id timestamp x y z qw qx qy qz')
    chunks = []
    for row in poses[::args.frame_step]:
        path = args.frontend_dir/'pcd'/f'{int(row[0])}.pcd'
        if not path.is_file():
            raise FileNotFoundError(path)
        cloud = o3d.io.read_point_cloud(str(path)).voxel_down_sample(args.voxel_size)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(row[[6, 7, 8, 5]]).as_matrix()
        transform[:3, 3] = row[2:5]
        cloud.transform(transform)
        chunks.append(np.asarray(cloud.points))
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(chunks)))
    result = result.voxel_down_sample(args.voxel_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output), result):
        raise OSError(f'Failed to write {args.output}')
    print(f'Exported {len(result.points):,} points from {len(chunks)} body scans to {args.output}')


if __name__ == '__main__':
    main()
