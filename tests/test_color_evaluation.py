import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d

from evaluation.cal_cis import compute_ccs, compute_cis
from evaluation.cal_colordist import compute_color_distance, to_db
from evaluation.cal_lccr import compute_lcr, compute_LCCR

ROOT = Path(__file__).resolve().parents[1]


def cloud(points, colors):
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float)))
    result.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float).reshape(-1, 3))
    return result


class ColorEvaluationTest(unittest.TestCase):
    def test_cd_uses_spatial_neighbors_and_rgb_l2(self):
        a = cloud([[0, 0, 0], [10, 0, 0]], [[0, 0, 0], [1, 1, 1]])
        b = cloud([[0, 0, 0], [10, 0, 0]], [[1, 1, 1], [0, 0, 0]])
        cd, forward, backward, err01, err10 = compute_color_distance(a, b, 1)
        self.assertAlmostEqual(cd, np.sqrt(3))
        self.assertAlmostEqual(forward, np.sqrt(3))
        self.assertAlmostEqual(backward, np.sqrt(3))
        np.testing.assert_allclose(err01, np.sqrt(3))
        np.testing.assert_allclose(err10, np.sqrt(3))
        self.assertAlmostEqual(to_db(cd), -20 * np.log10(np.sqrt(3)))

    def test_cd_equal_direction_weights_for_unequal_cloud_sizes(self):
        a = cloud([[0, 0, 0]], [[0, 0, 0]])
        b = cloud([[0, 0, 0], [10, 0, 0]], [[0, 0, 0], [1, 1, 1]])
        cd, forward, backward, _, _ = compute_color_distance(a, b, 1)
        self.assertEqual(forward, 0)
        self.assertAlmostEqual(backward, np.sqrt(3) / 2)
        self.assertAlmostEqual(cd, np.sqrt(3) / 4)
        self.assertAlmostEqual(compute_color_distance(b, a, 1)[0], cd)

    def test_cf_zero_small_errors_and_no_upper_clipping(self):
        self.assertEqual(to_db(0), np.inf)
        self.assertAlmostEqual(to_db(1e-10), 200)
        np.testing.assert_allclose(to_db(np.array([0, 1e-10, 2])), [np.inf, 200, -20*np.log10(2)])
        for invalid in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                to_db(invalid)

    def test_lcr_reference_denominator_and_any_neighbor(self):
        # The nearest point has the wrong color, but another local point matches.
        reconstructed = cloud([[0, 0, 0], [.2, 0, 0]], [[1, 1, 1], [0, 0, 0]])
        reference = cloud([[0, 0, 0], [.1, 0, 0], [10, 0, 0]], [[0, 0, 0]] * 3)
        for batch_size in (1, 2, 10):
            report = compute_lcr(reconstructed, reference, thread_num=1,
                                 batch_size=batch_size, return_details=True)
            self.assertAlmostEqual(report['LCR'], 2 / 3)
            self.assertEqual(report['matched_reference_points'], 2)
            self.assertEqual(report['reference_points'], 3)
        self.assertAlmostEqual(compute_LCCR(reconstructed, reference, save_pass=False), 2 / 3)

    def test_lcr_factor_three_l2_and_inclusive_boundaries(self):
        reconstructed = cloud([[0, 0, 0]], [[0, 0, 0]])
        reference = cloud([[.5, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0]],
                          [[.375, 0, 0], [.3750001, 0, 0], [.3, .3, 0], [0, 0, 0]])
        self.assertEqual(compute_lcr(reconstructed, reference, tau=.125, r_g=.5), .25)

    def test_lcr_nonoverlap_zero_and_export_unique_points(self):
        reconstructed = cloud([[0, 0, 0]], [[0, 0, 0]])
        reference = cloud([[.1, 0, 0], [.2, 0, 0]], [[0, 0, 0]] * 2)
        with tempfile.TemporaryDirectory() as tmp:
            score = compute_lcr(reconstructed, reference, save_pass=True, output_dir=tmp)
            self.assertEqual(score, 1)
            self.assertEqual(len(o3d.io.read_point_cloud(str(Path(tmp)/'map_pass.pcd')).points), 1)
            self.assertEqual(len(o3d.io.read_point_cloud(str(Path(tmp)/'truth_pass.pcd')).points), 2)
            far = cloud([[20, 0, 0]], [[0, 0, 0]])
            self.assertEqual(compute_lcr(reconstructed, far, save_pass=True, output_dir=tmp), 0)
            self.assertFalse((Path(tmp)/'truth_pass.pcd').exists())

    def test_ccs_sample_covariance_and_equal_voxel_weight(self):
        a = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        b = np.repeat(np.array([0, .25, .75, 1.])[:, None], 3, axis=1)
        points = [[.1, 0, 0], [.2, 0, 0], [2.1, 0, 0], [2.2, 0, 0], [2.3, 0, 0], [2.4, 0, 0]]
        pcd = cloud(points, np.concatenate([a, b]))
        expected = (np.trace(np.cov(a.T, ddof=1)) + np.trace(np.cov(b.T, ddof=1))) / 2
        self.assertAlmostEqual(compute_ccs(pcd, voxel_size=1, singleton_policy='error'), expected)
        self.assertAlmostEqual(expected, 1.0625)

    def test_ccs_singleton_conventions_and_two_point_voxel(self):
        pcd = cloud([[.1, 0, 0], [.2, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 1, 1], [1, 0, 0]])
        report = compute_ccs(pcd, 1, return_details=True)
        self.assertEqual(report['occupied_voxels'], 2)
        self.assertEqual(report['singleton_voxels'], 1)
        self.assertAlmostEqual(report['CCS'], .75)
        self.assertAlmostEqual(compute_ccs(pcd, 1, singleton_policy='exclude'), 1.5)
        with self.assertRaisesRegex(ValueError, 'singleton'):
            compute_ccs(pcd, 1, singleton_policy='error')
        singleton = cloud([[0, 0, 0]], [[.5, .5, .5]])
        self.assertEqual(compute_ccs(singleton), 0)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            compute_ccs(singleton, singleton_policy='exclude')

    def test_ccs_fixed_grid_negative_coordinates_and_uniform_color(self):
        pcd = cloud([[-.1, 0, 0], [.1, 0, 0]], [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(compute_ccs(pcd, 1), 0)
        self.assertEqual(compute_ccs(pcd, 1, voxel_origin=(-.5, 0, 0)), 1.5)
        pcd.colors = o3d.utility.Vector3dVector([[.3, .6, .9]] * 2)
        self.assertEqual(compute_ccs(pcd, 1, voxel_origin=(-.5, 0, 0)), 0)

    def test_invalid_clouds_and_parameters_fail(self):
        good = cloud([[0, 0, 0]], [[0, 0, 0]])
        invalid = [o3d.geometry.PointCloud(), cloud([[0, 0, 0]], []),
                   cloud([[np.nan, 0, 0]], [[0, 0, 0]]),
                   cloud([[0, 0, 0]], [[255, 0, 0]])]
        for bad in invalid:
            for compute in (lambda: compute_ccs(bad),
                            lambda: compute_lcr(good, bad),
                            lambda: compute_color_distance(good, bad)):
                with self.assertRaises(ValueError):
                    compute()
        for size in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                compute_ccs(good, size)
        with self.assertRaises(ValueError):
            compute_lcr(good, good, tau=-1)
        with self.assertRaises(ValueError):
            compute_lcr(good, good, r_g=0)
        with self.assertRaises(ValueError):
            compute_lcr(good, good, batch_size=0)
        with self.assertRaises(ValueError):
            compute_color_distance(good, good, thread_num=0)

    def test_cli_entrypoints_json_and_zero_error_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pcd = cloud([[.01, 0, 0], [.02, 0, 0]], [[0, 0, 0], [1, 1, 1]])
            path = tmp/'map.ply'
            o3d.io.write_point_cloud(str(path), pcd)
            self.assertAlmostEqual(compute_cis(path), 1.5)
            for script, args, expected in (
                ('cal_cis.py', ['--pcd', str(path)], {'CCS': 1.5, 'singleton_voxels': 0}),
                ('cal_lccr.py', ['--map', str(path), '--truth', str(path)], {'LCR': 1.0}),
                ('cal_colordist.py', [str(path), str(path), '--plot', str(tmp/'plot.png')],
                 {'CD': 0.0, 'CF_dB': '+inf'}),
            ):
                report = tmp/(script + '.json')
                subprocess.run([sys.executable, str(ROOT/'evaluation'/script), *args,
                                '--json', str(report)], cwd=tmp, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                data = json.loads(report.read_text())
                for key, value in expected.items():
                    self.assertEqual(data[key], value)
            self.assertTrue((tmp/'plot.png').is_file())


if __name__ == '__main__':
    unittest.main()
