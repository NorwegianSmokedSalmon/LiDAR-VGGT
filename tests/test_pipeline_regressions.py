"""Regression tests for geometry/data bugs found while bringing up the pipeline."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from Registration import sim3
from Utils import utils
from Registration.pointcloud import register_clouds
from models.inference import chunk_ranges


class PipelineRegressions(unittest.TestCase):
    def test_one_nearest_lidar_pose_per_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'images').mkdir()
            (root / 'pcd').mkdir()
            for name in ('100.jpg', '101.jpg'):
                (root / 'images' / name).touch()
            for idx in range(3):
                (root / 'pcd' / f'{idx}.pcd').touch()
            (root / 'timestamps.txt').write_text('100.jpg 1.0\n101.jpg 2.0\n')
            (root / 'pose.txt').write_text('0 0.97 10 0 0 1 0 0 0\n1 1.01 20 0 0 1 0 0 0\n2 2.0 30 0 0 1 0 0 0\n')
            images, _, poses = sim3.match_poses(folder, 'images', 'pcd', 'timestamps.txt', 'pose.txt', 0.06)
            self.assertEqual([Path(p).name for p in images], ['100.jpg', '101.jpg'])
            np.testing.assert_array_equal(np.asarray(poses)[:, 0], [20, 30])

    def test_short_chunk_has_only_real_camera_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            extrinsic = np.tile(np.eye(4)[:3], (2, 1, 1))
            extrinsic[1, 0, 3] = -2
            np.save(Path(folder) / 'chunk_0.npy', {'extrinsic': extrinsic})
            result = sim3.load_cameraposes_from_npy(folder, '.', 40)['chunk_0']
            self.assertEqual(result['positions'].shape, (2, 3))
            np.testing.assert_allclose(result['positions'], [[0, 0, 0], [2, 0, 0]])
            self.assertEqual(list(chunk_ranges(41, 40, 15)), [(0, 40), (25, 41)])
            self.assertEqual(list(chunk_ranges(65, 40, 15)), [(0, 40), (25, 65)])
            ranges = list(chunk_ranges(1224, 40, 15))
            self.assertEqual(len(ranges), 49)
            self.assertEqual(ranges[-1], (1200, 1224))
            self.assertEqual({i for start, end in ranges for i in range(start, end)}, set(range(1224)))

    def test_full_export_applies_sim3(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'out.ply'
            points = np.array([[1., 2, 3], [4, 5, 6]])
            rotation = Rotation.from_euler('z', 30, degrees=True).as_matrix()
            translation = np.array([4., -5, 6])
            utils.save_confident_pointcloud_batch(points, np.ones((2, 3), dtype=np.uint8) * 80,
                np.ones(2), str(path), 0.5, 1.0, 1, 2., rotation, translation, np.eye(3), np.zeros(3))
            actual = np.asarray(o3d.io.read_point_cloud(str(path)).points)
            np.testing.assert_allclose(actual, 2 * points @ rotation.T + translation, atol=2e-6)

    def test_saved_transforms_reproduce_each_registered_cloud(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'lidar').mkdir()
            (root / 'vggt').mkdir()
            rng = np.random.default_rng(4)
            source = rng.normal(size=(100, 3))
            for idx in range(2):
                target = source * (1.02 + 0.01 * idx) + np.array([0.03, 0.02, -0.02]) * (idx + 1)
                for sub, pts, name in [('lidar', target, f'chunk_{idx}.pcd'), ('vggt', source, f'{idx}_pc.ply')]:
                    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
                    o3d.io.write_point_cloud(str(root / sub / name), cloud)
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = register_clouds(str(root / 'lidar'), str(root / 'vggt'), str(root / 'transforms.txt'),
                    str(root / 'registered'), False, False, False, False, 0.1, 0.3,
                    max_sim3_iterations=4, min_sim3_iterations=4, icp_max_iterations=3)
            transforms = np.loadtxt(root / 'transforms.txt').reshape(-1, 4, 4)
            self.assertEqual(len(transforms), 2)
            for idx, transform in enumerate(transforms):
                saved = np.asarray(o3d.io.read_point_cloud(str(root / 'registered' / 'final' / f'{idx}_pc.ply')).points)
                np.testing.assert_allclose(source @ transform[:3, :3].T + transform[:3, 3], saved, atol=1e-8)
                self.assertLess(metrics[idx]['inlier_rmse'], 1e-5)


if __name__ == '__main__':
    unittest.main()
