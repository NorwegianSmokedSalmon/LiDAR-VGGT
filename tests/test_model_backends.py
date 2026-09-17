"""Geometry and pixel-coordinate contracts shared by the three real model backends."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from scipy.spatial.transform import Rotation

from models.backends import canonicalize_pi3, canonicalize_omega
from models.geometry import camera_to_extrinsic, depth_to_world, validate_predictions
from models.inference import chunk_ranges, saved_model
from models.preprocessing import prepare_images, omega_layout, resize_affine
from models.registry import MODEL_ROOT, use_repository, configured_checkpoint
from tools.run_projection_pipeline import intrinsics_for_image
from tools.download_models import verify_checkpoint


class ModelBackendsTest(unittest.TestCase):
    def test_pi3_camera_to_world_logits_and_pinhole_fit(self):
        h, w = 28, 42
        y, x = np.mgrid[:h, :w]
        z = 3 + 0.01*x + 0.02*y
        k = np.array([[35., 0, 20], [0, 32, 13], [0, 0, 1]], dtype=np.float32)
        local = np.stack([(x-20)/35*z, (y-13)/32*z, z], axis=-1).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = Rotation.from_euler('xyz', [20, -50, 70], degrees=True).as_matrix()
        pose[:3, 3] = [3, -2, 8]
        world = local @ pose[:3, :3].T + pose[:3, 3]
        native = {'local_points': torch.from_numpy(local)[None, None],
                  'points': torch.from_numpy(world)[None, None],
                  'conf': torch.full((1, 1, h, w, 1), 0.),
                  'camera_poses': torch.from_numpy(pose)[None, None]}
        result = canonicalize_pi3(native, torch.zeros(1, 3, h, w))
        validate_predictions(result, 1)
        np.testing.assert_allclose(result['world_points_conf'], .5)
        np.testing.assert_allclose(result['intrinsic'][0], k, atol=1e-4)
        np.testing.assert_allclose(result['world_points'][0] @ result['extrinsic'][0, :, :3].T
                                   + result['extrinsic'][0, :, 3], local, atol=2e-6)
        np.testing.assert_allclose(result['depth'][0], z, atol=1e-6)

    @unittest.skipUnless((MODEL_ROOT/'repos/vggt_omega').is_dir(), 'Omega source not downloaded')
    def test_omega_depth_unprojection_matches_known_cameras(self):
        use_repository('vggt_omega')
        from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding
        h, w, n = 16, 32, 2
        poses = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)
        poses[1, :3, :3] = Rotation.from_euler('yx', [35, 20], degrees=True).as_matrix()
        poses[1, :3, 3] = [3, -5, 8]
        extrinsic = camera_to_extrinsic(poses)
        k = np.tile([[25., 0, w/2], [0, 20, h/2], [0, 0, 1]], (n, 1, 1)).astype(np.float32)
        encoding = extri_intri_to_pose_encoding(torch.from_numpy(extrinsic)[None], torch.from_numpy(k)[None], (h,w))
        native = {'pose_enc': encoding, 'depth': torch.full((1,n,h,w,1), 4.),
                  'depth_conf': torch.full((1,n,h,w), 1.5)}
        result = canonicalize_omega(native, torch.zeros(n, 3, h, w))
        validate_predictions(result, n)
        np.testing.assert_allclose(result['extrinsic'], extrinsic, atol=1e-6)
        np.testing.assert_allclose(result['intrinsic'], k, atol=1e-5)
        np.testing.assert_allclose(result['world_points'][:, h//2, w//2],
                                   poses[:, :3, 2]*4+poses[:, :3, 3], atol=2e-6)

    @unittest.skipUnless((MODEL_ROOT/'repos/vggt_omega').is_dir(), 'Omega source not downloaded')
    def test_omega_extreme_crop_and_mixed_shape_padding_preserve_pixels(self):
        with tempfile.TemporaryDirectory() as work:
            paths = []
            for i, size in enumerate([(1000, 200), (200, 800)]):
                path = Path(work)/f'{i}.png'
                Image.new('RGB', size, color=(80,120,160)).save(path)
                paths.append(path)
            prepared = prepare_images(paths, 'vggt_omega')
            for i, size in enumerate([(1000,200),(200,800)]):
                crop_wh, crop_xy, resized = omega_layout(*size)
                left, top, right, bottom = prepared.valid_bounds[i]
                np.testing.assert_allclose(prepared.pixel_transform[i] @ [crop_xy[0]+(crop_wh[0]-1)/2,
                                            crop_xy[1]+(crop_wh[1]-1)/2, 1],
                                           [(left+right)/2, (top+bottom)/2, 1])
                self.assertTrue((prepared.images[i, :, 0, 0] == 1).all())
                mid_y, mid_x = int((top+bottom)/2), int((left+right)/2)
                np.testing.assert_allclose(prepared.images[i, :, mid_y, mid_x], np.array([80,120,160])/255., atol=1e-7)

    def test_pi3_explicit_order_and_saved_affine_drive_calibrated_projection(self):
        with tempfile.TemporaryDirectory() as work:
            paths = []
            for name, color in [('10.png', (255,0,0)), ('2.png', (0,0,255))]:
                path = Path(work)/name
                Image.new('RGB',(1224,1024),color).save(path)
                paths.append(path)
            prepared = prepare_images(paths, 'pi3')
            h, w = prepared.images.shape[-2:]
            self.assertEqual((h % 14,w % 14),(0,0))
            self.assertLessEqual(h*w,255000)
            np.testing.assert_allclose(prepared.images[:, :, 0, 0], [[1,0,0],[0,0,1]])
            original_k = np.array([[730.,0,600],[0,730,510],[0,0,1]])
            options = {'intrinsics_source':'calibration','camera_matrix':original_k,
                       'camera_image_size':[1224,1024],'preprocess_mode':'crop',
                       'images_undistorted':True,'distortion':[]}
            k, _ = intrinsics_for_image(options, np.eye(3), paths[0], (h,w), prepared.pixel_transform[0])
            np.testing.assert_allclose(k, prepared.pixel_transform[0] @ original_k)
            self.assertNotEqual(w,518)  # Must not silently use the original VGGT resize.

    def test_cached_model_mismatch_is_rejected_and_legacy_defaults_to_vggt(self):
        self.assertEqual(saved_model({}), 'vggt')
        self.assertEqual(saved_model({'model':'pi3'}, 'pi3'),'pi3')
        with self.assertRaisesRegex(ValueError,'Saved predictions use pi3'):
            saved_model({'model':'pi3'}, 'vggt')
        self.assertEqual(list(chunk_ranges(9,5,2)), [(0,5),(3,8),(6,9)])

    def test_checkpoint_override_and_existing_config_are_respected(self):
        cfg = {'Weights': {'VGGT':'custom/vggt.pt', 'Pi3':'/tmp/pi3.safetensors'}}
        self.assertEqual(configured_checkpoint('vggt', cfg, '/tmp/override.pt'), Path('/tmp/override.pt'))
        self.assertEqual(configured_checkpoint('vggt', cfg), MODEL_ROOT.parent/'custom/vggt.pt')
        self.assertEqual(configured_checkpoint('pi3', cfg), Path('/tmp/pi3.safetensors'))
        self.assertEqual(configured_checkpoint('vggt_omega', cfg).parent, MODEL_ROOT/'weights/vggt_omega')

    def test_download_hash_rejects_corruption(self):
        import hashlib
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)/'weight'
            path.write_bytes(b'verified model')
            spec = {'size':14,'sha256':hashlib.sha256(b'verified model').hexdigest()}
            self.assertTrue(verify_checkpoint(path,spec))
            path.write_bytes(b'corrupt! model')
            self.assertFalse(verify_checkpoint(path,spec))

    def test_invalid_canonical_shape_and_rotation_are_rejected(self):
        arrays = {'world_points':np.zeros((1,4,5,3)), 'world_points_conf':np.ones((1,4,5)),
                  'depth':np.ones((1,4,5)), 'depth_conf':np.ones((1,4,5)),
                  'images':np.zeros((1,3,4,5)), 'intrinsic':np.eye(3)[None],
                  'extrinsic':np.zeros((1,3,4))}
        with self.assertRaisesRegex(ValueError,'rotations'):
            validate_predictions(arrays,1)
        arrays['world_points_conf'] = np.ones((4,5))
        with self.assertRaisesRegex(ValueError,'shape'):
            validate_predictions(arrays,1)


if __name__ == '__main__':
    unittest.main()
