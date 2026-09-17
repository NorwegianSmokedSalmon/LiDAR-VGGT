"""Known-transform, gauge, outlier, leakage and sparse-solver PGO regressions."""
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from PGO.geometry import correspondence_icp, fit_rigid, invert, metric_camera_poses, transform
from PGO.graph import edge_errors, optimize, warm_start
from PGO.pipeline import overlap_metrics, validate_config, select_candidate

ROOT = Path(__file__).resolve().parents[1]


def pose(rotation=(0, 0, 0), translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    matrix[:3, 3] = translation
    return matrix


class PythonPGOTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT/'config/pgo.yaml').read_text())

    def test_metric_scale_changes_camera_centers_and_points_not_rotations(self):
        camera = pose((.2, -.4, .3), (2, 4, 1))
        alignment = pose((-.3, .1, .5), (5, -2, 7))
        alignment[:3, :3] *= 3.2
        actual, scale = metric_camera_poses(invert(camera)[None, :3], alignment)
        points = np.array([[1., 2., 9.], [-2, 3, 8]])
        world = transform(points, camera)
        np.testing.assert_allclose(transform(points*scale, actual[0]), transform(world, alignment), atol=1e-12)
        np.testing.assert_allclose(actual[0, :3, :3].T @ actual[0, :3, :3], np.eye(3), atol=1e-12)
        bad = alignment.copy(); bad[0, 0] *= 2
        with self.assertRaises(ValueError):
            metric_camera_poses(invert(camera)[None, :3], bad)

    def test_between_factor_direction_with_noncommuting_rotations(self):
        a = pose((.3, .6, -.2), (4, 2, 1))
        b = pose((-.5, .2, .7), (-2, 4, 8))
        measured = invert(a) @ b
        correct = edge_errors(np.array([a, b]), [0], [1], measured[None])
        np.testing.assert_allclose(correct, 0, atol=1e-12)
        wrong = edge_errors(np.array([a, b]), [0], [1], invert(measured)[None])
        self.assertGreater(np.linalg.norm(wrong), 1)

    def test_ransac_correspondence_initialization_survives_outliers(self):
        rng = np.random.default_rng(23)
        source = rng.uniform(-4, 4, (600, 3))
        expected = pose((.1, -.2, .3), (5, -3, 2))
        target = transform(source, expected) + rng.normal(0, .005, source.shape)
        target[:180] = rng.uniform(-15, 15, (180, 3))
        config = dict(self.config['icp'], ransac_threshold_m=.08, max_correspondence_distance_m=.08)
        actual, info = correspondence_icp(source, target, config, 5)
        np.testing.assert_allclose(actual, expected, atol=.003)
        self.assertGreater(info['fit_inlier_ratio'], .65)
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            fit_rigid(np.ones((5, 3)), np.ones((5, 3)))

    def test_overlap_edges_recover_misaligned_chunks_and_keep_first_pose_exact(self):
        truth = np.array([pose((0, 0, .02*i), (i, .1*i, 0)) for i in [0, 1, 2, 1, 2, 3]])
        initial = truth.copy()
        correction = pose((.02, -.05, .3), (3, -2, .5))
        initial[3:] = correction @ initial[3:]
        source = np.array([0, 1, 3, 4, 1, 2]); target = np.array([1, 2, 4, 5, 3, 4])
        kinds = np.array(['odom']*4 + ['icp']*2)
        measured = invert(truth[source]) @ truth[target]
        result, info = optimize(initial, source, target, measured, kinds, self.config, 1.)
        self.assertTrue(info['converged'])
        np.testing.assert_allclose(result, truth, atol=2e-4)
        np.testing.assert_array_equal(result[0], initial[0])
        warm = warm_start(initial, np.array([0, 0, 0, 1, 1, 1]), source, target, measured, kinds)
        np.testing.assert_allclose(warm, truth, atol=1e-12)

    def test_weak_icp_cannot_override_strong_odometry_or_fixed_prior(self):
        initial = np.array([pose(), pose(translation=(1, 0, 0))])
        measured = np.array([pose(translation=(1, 0, 0)), pose(translation=(2, 0, 0))])
        result, info = optimize(initial, [0, 0], [1, 1], measured, ['odom', 'icp'], self.config, .1)
        self.assertTrue(info['converged'])
        self.assertLess(abs(result[1, 0, 3]-1), .02)
        np.testing.assert_array_equal(result[0], initial[0])
        with self.assertRaises(ValueError):
            optimize(initial, [0], [1], measured[:1], ['icp'], self.config, 10.)

    def test_disconnected_graph_is_rejected_instead_of_moving_unanchored_chunks(self):
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            optimize(np.array([pose()]*4), [0, 2], [1, 3], np.array([pose()]*2), ['odom']*2, self.config, 1.)

    def test_test_pixels_are_not_changed_by_fitting_or_tuning_pixels(self):
        poses = np.array([pose(), pose()])
        x = np.zeros((10, 3)); y = np.zeros((10, 3))
        labels = np.arange(10) % 5
        y[labels == 0, 0] = 2
        pair = {'i': 0, 'j': 1, 'x': x, 'y': y, 'split': labels, 'boundary': (0, 1)}
        before = overlap_metrics(poses, [pair], 0)
        y[labels != 0] = 1e6
        self.assertEqual(before, overlap_metrics(poses, [pair], 0))
        self.assertEqual(before['overall_m']['rmse'], 2.)

    def test_each_disconnected_component_can_have_a_fixed_chunk_prior(self):
        initial = np.array([pose(translation=(x, 0, 0)) for x in (0, 1, 20, 21)])
        result, info = optimize(initial, [0, 2], [1, 3],
                                np.array([pose(translation=(1, 0, 0))]*2),
                                ['odom']*2, self.config, 1., fixed_nodes=[0, 2])
        self.assertTrue(info['converged'])
        np.testing.assert_array_equal(result[[0, 2]], initial[[0, 2]])
        np.testing.assert_allclose(result, initial, atol=1e-12)

    def test_tuning_guard_does_not_apply_a_converged_but_worse_candidate(self):
        trials = [{'eligible': True, 'tuning_overlap': {'overall_m': {'rmse': x}}} for x in (4, 3, 5)]
        choice, applied = select_candidate(trials, {'overall_m': {'rmse': 2}})
        self.assertEqual(choice, 1)
        self.assertFalse(applied)
        self.assertTrue(select_candidate(trials, {'overall_m': {'rmse': 4}})[1])

    def test_config_enforces_weight_order_and_finite_noise(self):
        validate_config(self.config)
        self.config['weights']['overlap_icp'] = 11
        with self.assertRaises(ValueError): validate_config(self.config)
        self.config['weights']['overlap_icp'] = 1
        self.config['noise']['translation_sigma_m'] = float('nan')
        with self.assertRaises(ValueError): validate_config(self.config)


if __name__ == '__main__':
    unittest.main()
