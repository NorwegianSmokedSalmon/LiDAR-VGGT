"""Geometric and adversarial checks for the correspondence-based pipeline."""
import unittest
import tempfile
from pathlib import Path
from PIL import Image

import numpy as np
from scipy.spatial.transform import Rotation

from Registration.projection_sim3 import (LidarTrajectory, fit_sim3, preprocess_affine,
    project_visible, ransac_sim3, sample_pointmap)


class ProjectionSim3Test(unittest.TestCase):
    def test_ransac_recovers_large_transform_with_70_percent_outliers(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(1200, 3))
        rotation = Rotation.from_euler('zyx', [150, -50, 70], degrees=True).as_matrix()
        translation = np.array([400., -200, 70])
        target = 17.3 * source @ rotation.T + translation + rng.normal(scale=0.002, size=source.shape)
        bad = rng.choice(len(source), 840, replace=False)
        target[bad] = rng.uniform(-1000, 1000, size=(len(bad), 3))
        result = ransac_sim3(source, target, threshold=0.015, max_iterations=3000, seed=42)
        self.assertAlmostEqual(result.scale, 17.3, places=3)
        np.testing.assert_allclose(result.rotation, rotation, atol=1e-4)
        np.testing.assert_allclose(result.translation, translation, atol=1e-3)
        self.assertEqual(result.inliers.sum(), 360)
        self.assertAlmostEqual(np.linalg.det(result.rotation), 1.0, places=10)

    def test_planar_pairs_are_valid_but_collinear_pairs_are_not(self):
        source = np.array([[0., 0, 0], [2, 0, 0], [0, 3, 0], [4, 4, 0]])
        rotation = Rotation.from_euler('xy', [80, -40], degrees=True).as_matrix()
        target = 3. * source @ rotation.T + [100, 40, -2]
        scale, r, t = fit_sim3(source, target)
        np.testing.assert_allclose(scale * source @ r.T + t, target, atol=1e-10)
        line = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            fit_sim3(line, line * 3)
        with self.assertRaises(ValueError):
            ransac_sim3(source, target, threshold=-1)

    def test_projection_removes_behind_outside_and_occluded_points(self):
        points = np.array([[0., 0, 4], [0, 0, 2], [0, 0, -1], [100, 0, 1], [1, 0, 2], [np.nan, 0, 2]])
        k = np.array([[2., 0, 2], [0, 2, 2], [0, 0, 1]])
        idx, uv = project_visible(points, k, (5, 5))
        np.testing.assert_array_equal(idx, [1, 4])
        np.testing.assert_allclose(uv, [[2, 2], [3, 2]])

    def test_original_intrinsics_follow_anisotropic_resize_and_crop(self):
        affine = preprocess_affine((1224, 1024), (434, 518))
        np.testing.assert_allclose(affine @ [611.5, 511.5, 1], [258.5, 216.5, 1])
        self.assertAlmostEqual(affine[0, 0], 518 / 1224)
        self.assertAlmostEqual(affine[1, 1], 434 / 1024)
        portrait = preprocess_affine((600, 1000), (518, 518))
        np.testing.assert_allclose(portrait @ [299.5, 499.5, 1], [258.5, 258.5, 1])
        pad = preprocess_affine((1000, 600), (518, 518), mode='pad')
        np.testing.assert_allclose(pad @ [499.5, 299.5, 1], [258.5, 258.5, 1])
        with self.assertRaises(ValueError):
            preprocess_affine((1224, 1024), (224, 224))

    def test_bilinear_lookup_filters_depth_edges_and_confidence(self):
        y, x = np.mgrid[:4, :4]
        pointmap = np.stack([x, y, x + y], axis=-1).astype(float)
        confidence, depth = np.ones((4, 4)), np.ones((4, 4))
        samples, good = sample_pointmap(pointmap, confidence, depth, np.array([[0.25, 0.75], [3, 3], [-1, 1]]), 0.5)
        np.testing.assert_allclose(samples, [[0.25, 0.75, 1]])
        np.testing.assert_array_equal(good, [True, False, False])
        depth[1, 1] = 2
        _, good = sample_pointmap(pointmap, confidence, depth, np.array([[0.5, 0.5]]), 0.5)
        self.assertFalse(good[0])
        depth[:] = 1
        confidence[1, 1] = 0
        _, good = sample_pointmap(pointmap, confidence, depth, np.array([[0.5, 0.5]]), 0.5)
        self.assertFalse(good[0])

    def test_scan_world_point_and_image_time_motion_compensation(self):
        rows = np.array([[0, 10, 0, 0, 0, 1, 0, 0, 0], [1, 12, 2, 0, 0, 1, 0, 0, 0]], dtype=float)
        trajectory = LidarTrajectory(rows)
        rc = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        world, camera = trajectory.world_and_camera(np.array([[3., 2, 4]]), 0, 11, rc, np.array([0., 1, 0]))
        np.testing.assert_allclose(world, [[3, 2, 4]])
        np.testing.assert_allclose(camera, [[-2, 3, 4]])
        self.assertEqual(trajectory.closest(11.99, 0.02), 1)
        with self.assertRaises(ValueError):
            trajectory.closest(11, 0.02)
        with self.assertRaises(ValueError):
            trajectory.at(9)

    def test_brown_distortion_is_applied_before_pixel_lookup(self):
        x, y = 0.2, -0.3
        k1, k2, p1, p2, k3 = 0.1, 0.02, 0.003, 0.004, 0.005
        radius = x*x + y*y
        radial = 1 + k1*radius + k2*radius**2 + k3*radius**3
        xd = x*radial + 2*p1*x*y + p2*(radius + 2*x*x)
        yd = y*radial + p1*(radius + 2*y*y) + 2*p2*x*y
        k = np.array([[80., 0, 50], [0, 70, 50], [0, 0, 1]])
        _, uv = project_visible(np.array([[x*5, y*5, 5]]), k, (100, 100), [k1, k2, p1, p2, k3])
        np.testing.assert_allclose(uv, [[80*xd + 50, 70*yd + 50]])

    def test_calibration_rescale_and_undistortion_switch(self):
        from tools.run_projection_pipeline import intrinsics_for_image
        options = {'intrinsics_source': 'calibration', 'camera_matrix': [1000., 0, 599.5, 0, 1100., 399.5, 0, 0, 1],
                   'camera_image_size': [1200, 800], 'allow_intrinsics_rescale': True,
                   'preprocess_mode': 'crop', 'images_undistorted': False, 'distortion': [-0.1, 0, 0, 0, 0]}
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'image.png'
            Image.new('RGB', (600, 400)).save(path)
            k, d = intrinsics_for_image(options, np.eye(3), path, (350, 518))
            np.testing.assert_allclose(k, [[1000*518/1200, 0, 258.5], [0, 1100*350/800, 174.5], [0, 0, 1]])
            self.assertEqual(d[0], -0.1)
            options['images_undistorted'] = True
            self.assertIsNone(intrinsics_for_image(options, np.eye(3), path, (350, 518))[1])
            options['allow_intrinsics_rescale'] = False
            with self.assertRaisesRegex(ValueError, 'resolution'):
                intrinsics_for_image(options, np.eye(3), path, (350, 518))

    def test_motion_compensation_interpolates_rotations(self):
        q = Rotation.from_euler('z', 90, degrees=True).as_quat()
        rows = [[0, 10, 0, 0, 0, 1, 0, 0, 0], [1, 12, 0, 0, 0, q[3], q[0], q[1], q[2]]]
        trajectory = LidarTrajectory(rows)
        world, camera = trajectory.world_and_camera(np.array([[2., 0, 3]]), 0, 11, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(world, [[2, 0, 3]])
        np.testing.assert_allclose(camera, [[np.sqrt(2), -np.sqrt(2), 3]])

    def test_ransac_rejects_inconsistent_pairs(self):
        rng = np.random.default_rng(1)
        source, target = rng.normal(size=(2, 100, 3))
        with self.assertRaisesRegex(ValueError, 'consensus too small'):
            ransac_sim3(source, target, threshold=1e-4, max_iterations=100, min_inliers=20)

    def test_pixel_pairs_recover_sim3_in_world_frame(self):
        y, x = np.mgrid[:20, :20]
        k = np.array([[10., 0, 9], [0, 10, 9], [0, 0, 1]])
        z = 8 + x * 0.08 + y * 0.05
        cam_grid = np.stack([(x - 9) * z / 10, (y - 9) * z / 10, z], axis=-1)
        rwl = Rotation.from_euler('zy', [35, 15], degrees=True).as_matrix()
        twl = np.array([20., -4, 5])
        rcl = Rotation.from_euler('x', 20, degrees=True).as_matrix()
        tcl = np.array([0.1, -0.2, 0.3])
        world_grid = (cam_grid - tcl) @ rcl @ rwl.T + twl
        rotation = Rotation.from_euler('yz', [50, 70], degrees=True).as_matrix()
        scale, translation = 4.3, np.array([-3., 8, 9])
        pointmap = (world_grid - translation) @ rotation / scale
        coords = np.array([[2, 3], [4, 15], [14, 4], [10, 12], [7, 8], [16, 16]])
        camera = cam_grid[coords[:, 1], coords[:, 0]]
        lidar = (camera - tcl) @ rcl
        target = lidar @ rwl.T + twl
        indices, uv = project_visible(camera, k, (20, 20))
        source, good = sample_pointmap(pointmap, np.ones((20, 20)), z, uv, max_depth_jump=0.1)
        s, r, t = fit_sim3(source, target[indices][good])
        self.assertAlmostEqual(s, scale, places=10)
        np.testing.assert_allclose(r, rotation, atol=1e-10)
        np.testing.assert_allclose(t, translation, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
