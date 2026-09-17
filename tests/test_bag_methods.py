"""Shared input integrity and method isolation for raw-bag reconstruction."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from models.predictions import load_prediction_manifest, prediction_directory
from tools.bag.frontend import ensure_frontend
from tools.bag.reader import bag_identity
from tools.bag.reconstruction import method_command, completed_run
from tools.validate_bag_pipeline import registration_metrics

ROOT = Path(__file__).resolve().parents[1]


class BagMethodsTest(unittest.TestCase):
    def predictions(self, root):
        images = [root/f'{i}.jpg' for i in range(9)]
        cache = root/'predictions'
        cache.mkdir()
        manifest = {'model': 'pi3', 'chunk_size': 4, 'overlap': 1, 'chunks': []}
        for i, (start, end) in enumerate([(0, 4), (3, 7), (6, 9)]):
            name = f'chunk_{i}.npy'
            np.save(cache/name, {'fixture': i})
            manifest['chunks'].append({'index': i, 'start': start, 'end': end, 'file': name,
                                       'images': [str(p) for p in images[start:end]]})
        (cache/'manifest.json').write_text(json.dumps(manifest))
        return images, cache, manifest

    def test_shared_cache_rejects_wrong_model_layout_order_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            images, cache, _ = self.predictions(Path(folder))
            result = load_prediction_manifest(cache, images, 'pi3', chunk_size=4, chunk_step=3, require_all=True)
            self.assertEqual(len(result['chunks']), 3)
            for options in ({'requested_model': 'vggt'}, {'chunk_size': 8}, {'chunk_step': 2}):
                with self.assertRaises(ValueError):
                    load_prediction_manifest(cache, images, **options)
            with self.assertRaisesRegex(ValueError, 'dataset/order'):
                load_prediction_manifest(cache, images[::-1], 'pi3')
            (cache/'chunk_2.npy').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                load_prediction_manifest(cache, images, 'pi3', require_all=True)

    def test_partial_cache_is_valid_only_for_completed_requested_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            images, cache, manifest = self.predictions(Path(folder))
            manifest['chunks'] = manifest['chunks'][:2]
            (cache/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(len(load_prediction_manifest(cache, images, max_chunks=2)['chunks']), 2)
            with self.assertRaisesRegex(ValueError, 'cover all'):
                load_prediction_manifest(cache, images, require_all=True)
            with self.assertRaisesRegex(ValueError, 'missing requested'):
                load_prediction_manifest(cache, images, max_chunks=3)
            manifest['chunks'][1]['start'] = 4
            (cache/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'indices or bounds'):
                load_prediction_manifest(cache, images)

    def test_methods_share_predictions_but_use_distinct_registration_configs(self):
        commands = {name: method_command(name, Path('/data/scene'), Path('/out')/name,
                    Path('/shared/predictions'), 'pi3', .05, None, refine_line=True)
                    for name in ('baseline', 'correspondence')}
        for command in commands.values():
            self.assertEqual(command[command.index('--prediction-dir')+1], '/shared/predictions')
            self.assertIn('--all-images', command)
            self.assertIn('--merge', command)
        baseline, updated = commands['baseline'], commands['correspondence']
        self.assertTrue(baseline[1].endswith('/run_pipeline.py'))
        self.assertTrue(updated[1].endswith('/run_projection_pipeline.py'))
        self.assertEqual(baseline[baseline.index('--config')+1], '/data/scene/dataset_config.yaml')
        self.assertEqual(updated[updated.index('--config')+1], '/data/scene/projection.yaml')
        self.assertIn('--refine-line', baseline)
        self.assertNotIn('--refine-line', updated)
        self.assertNotIn('--refine-line', method_command('baseline', Path('/d'), Path('/o'), Path('/p'), 'pi3', .05, 2))

    def test_prediction_location_supports_old_local_and_new_shared_results(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder)
            self.assertEqual(prediction_directory(result), result/'npy')
            (result/'summary.json').write_text(json.dumps({'prediction_dir': str(result/'shared')}))
            self.assertEqual(prediction_directory(result), result/'shared')

    def test_completed_method_rejects_missing_chunks_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            request = {'method': 'baseline', 'sample_ratio': .05}
            (output/'bag_run.json').write_text(json.dumps({'request': request, 'complete': True}))
            (output/'merged_map.ply').write_bytes(b'fixture')
            (output/'merge_summary.json').write_text(json.dumps({'unique_images': 4, 'chunks': 1,
                                                              'merged_file': str(output/'merged_map.ply')}))
            (output/'summary.json').write_text(json.dumps({'registration': [{}]}))
            with self.assertRaisesRegex(ValueError, 'missing maps/chunks'):
                completed_run(output, request, 4, 1)
            (output/'vggt_chunk_reg/final').mkdir(parents=True)
            (output/'vggt_chunk_reg/final/0_pc.ply').write_bytes(b'fixture')
            self.assertTrue(completed_run(output, request, 4, 1)['complete'])
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                completed_run(output, dict(request, sample_ratio=.1), 4, 1)

    def test_completed_frontend_needs_no_docker_and_preserves_image_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bag = root/'scene.bag'
            bag.write_bytes(b'fixture')
            front = root/'frontend'
            (front/'pcd').mkdir(parents=True)
            preset = {'fastlio_config': 'config/bag/fastlio_avia.yaml'}
            config = yaml.safe_load((ROOT/preset['fastlio_config']).read_text())
            plan = {'bag': bag_identity(bag), 'image': 'lemon-mapping:fastlio', 'image_id': 'sha256:original-image',
                    'config': config, 'playback_rate': 4.}
            plan_path = front/'host_manifest.json'
            plan_path.write_text(json.dumps(plan))
            original = plan_path.read_bytes()
            (front/'sequence.json').write_text(json.dumps({'complete': True, 'keyframes': 1, 'record_start': 1000}))
            exported = copy.deepcopy(config)
            exported['lemon_export']['directory'] = '/result'
            exported['use_sim_time'] = True
            (front/'fastlio-config.yaml').write_text(yaml.safe_dump(exported))
            np.savetxt(front/'pose.txt', [[0, 1000, 0, 0, 0, 1, 0, 0, 0]])
            np.savetxt(front/'trajectory.tum', [[1000, 0, 0, 0, 0, 0, 0, 1]])
            (front/'pcd/0.pcd').touch()
            with patch('tools.bag.frontend.docker_command', side_effect=AssertionError('Docker must not be called')):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ensure_frontend(bag, root, preset, 'lemon-mapping:fastlio', 4.)
            self.assertTrue(result['complete'])
            self.assertEqual(plan_path.read_bytes(), original)
            bag.write_bytes(b'changed bag')
            with self.assertRaisesRegex(ValueError, 'input/config/image changed'):
                ensure_frontend(bag, root, preset, 'lemon-mapping:fastlio', 4.)

    def test_validator_keeps_icp_metrics_separate_from_paired_residuals(self):
        old = registration_metrics({'registration': [{'fitness': .2, 'inlier_rmse': .5}]})
        new = registration_metrics({'metric_type': 'paired_correspondence', 'registration': [
            {'validation': {'pairs': 100, 'inliers': 60, 'inlier_rmse_m': 1., 'median_m': 1.2}}]})
        self.assertEqual(old['mean_icp_fitness'], .2)
        self.assertNotIn('held_out_inlier_ratio', old)
        self.assertEqual(new['held_out_inlier_ratio'], .6)
        self.assertNotIn('mean_icp_fitness', new)


if __name__ == '__main__':
    unittest.main()
