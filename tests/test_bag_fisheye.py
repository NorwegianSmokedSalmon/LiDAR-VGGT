"""Rectified pixel calibration and re-recorded bag clock regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from tools.bag.images import BagImageProcessor, image_time_origin
from tools.bag.dataset import write_reconstruction_config
from tools.extract_bag_keyframes import extract

ROOT = Path(__file__).resolve().parents[1]


class FisheyeBagTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        self.config = yaml.safe_load((ROOT/'config/bag/lighthouse.yaml').read_text())

    def test_remap_matches_kannala_brandt_rays(self):
        processor = BagImageProcessor(self.config)
        k = np.asarray(self.config['camera_matrix']).reshape(3, 3)
        output_k = np.asarray(processor.calibration['camera_matrix']).reshape(3, 3)
        pixels = np.array([[100, 150], [1000, 900], [700, 500], [1200, 300]])
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(output_k).T
        radius = np.linalg.norm(rays[:, :2], axis=1)
        theta = np.arctan(radius)
        theta_distorted = theta.copy()
        for i, coefficient in enumerate(self.config['distortion']):
            theta_distorted += coefficient * theta**(2*i+3)
        distorted = rays[:, :2] * (theta_distorted/radius)[:, None]
        source = np.c_[distorted, np.ones(len(pixels))] @ k.T
        mapped = np.array([[processor.maps[0][y, x], processor.maps[1][y, x]] for x, y in pixels])
        np.testing.assert_allclose(mapped, source[:, :2], atol=1e-4, rtol=0)
        self.assertTrue(processor.calibration['images_undistorted'])
        self.assertEqual(processor.calibration['distortion'], [0.]*5)

    def test_explicit_clock_and_invalid_calibration(self):
        info = {'record_start': 2000., 'image': {'first_header_time': 1000.}}
        self.config['image_time_offset_s'] = .1
        self.assertEqual(image_time_origin(info, self.config), 1000.1)
        self.config['selection_time_basis'] = 'bag_record'
        self.assertEqual(image_time_origin(info, self.config), 2000.)
        self.config['selection_time_basis'] = 'guess'
        with self.assertRaisesRegex(ValueError, 'selection_time_basis'):
            image_time_origin(info, self.config)
        self.config['distortion'] = [0.]*3
        with self.assertRaisesRegex(ValueError, 'four coefficients'):
            BagImageProcessor(self.config)

    def test_rerecorded_bag_preserves_raw_pixels_and_rectified_intrinsics(self):
        config = self.config
        config.update(image_topic='/camera', camera_image_size=[320, 240],
                      camera_matrix=[280, 0, 160, 0, 280, 120, 0, 0, 1],
                      distortion=[-.07, .02, -.03, .01], image_time_offset_s=-.1,
                      lidar_to_camera_R=np.eye(3).ravel().tolist(), lidar_to_camera_t=[0., 0., 0.])
        config['selection'].update(tracking_width=320, min_tracks=30, min_track_ratio=.1, parallax_px=6)
        store = get_typestore(Stores.ROS1_NOETIC)
        Time = store.types['builtin_interfaces/msg/Time']
        Header = store.types['std_msgs/msg/Header']
        Image = store.types['sensor_msgs/msg/CompressedImage']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag, dataset = root/'replayed.bag', root/'prepared'
            (dataset/'frontend').mkdir(parents=True)
            payloads = []
            texture = cv2.GaussianBlur(np.random.default_rng(9).integers(0, 256, (240, 320, 3), dtype=np.uint8), (3, 3), 0)
            with Writer(bag) as writer:
                connection = writer.add_connection('/camera', Image.__msgtype__, typestore=store)
                for i in range(4):
                    shifted = cv2.warpAffine(texture, np.float32([[1, 0, 12*i], [0, 1, 0]]), (320, 240))
                    _, payload = cv2.imencode('.jpg', shifted)
                    payloads.append(payload)
                    message = Image(Header(i, Time(1000+i, 100000000), 'camera'), 'jpeg', payload)
                    writer.write(connection, (2000+i)*10**9, store.serialize_ros1(message, Image.__msgtype__))
            rows = [[i, 1000+i, float(i), 0, 0, 1, 0, 0, 0] for i in range(4)]
            np.savetxt(dataset/'frontend/pose.txt', rows)
            (dataset/'frontend/frames.json').write_text(json.dumps({'body_from_imu': np.eye(3).ravel().tolist()}))
            (dataset/'frontend/fastlio-config.yaml').write_text(yaml.safe_dump({'mapping': {
                'extrinsic_R': np.eye(3).ravel().tolist(), 'extrinsic_T': [0., 0., 0.]}}))
            with contextlib.redirect_stdout(io.StringIO()):
                report = extract(bag, dataset, config, start_offset=0, end_offset=3)
                self.assertEqual(report, extract(bag, dataset, config, start_offset=0, end_offset=3))
            self.assertEqual(report['flight_window']['time_origin'], 1000.)
            self.assertEqual(report['selected_images'], 4)
            processor = BagImageProcessor(config)
            for entry in report['selected']:
                payload = payloads[entry['bag_index']]
                self.assertEqual((dataset/entry['original_image']).read_bytes(), payload.tobytes())
                saved = cv2.imread(str(dataset/'images'/entry['image']))
                np.testing.assert_array_equal(saved, processor.decode(payload))
                self.assertAlmostEqual(entry['timestamp'], 1000+entry['bag_index'])
            projection_path = write_reconstruction_config(dataset, config, report, 4, 2)
            projection = yaml.safe_load(projection_path.read_text())['Projection']
            for key, expected in processor.calibration.items():
                self.assertEqual(projection[key], expected)
            # A completed cache cannot silently drop the original source images.
            (dataset/report['selected'][0]['original_image']).unlink()
            with self.assertRaisesRegex(ValueError, 'missing original'):
                extract(bag, dataset, config, start_offset=0, end_offset=3)


if __name__ == '__main__':
    unittest.main()
