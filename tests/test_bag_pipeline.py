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
from scipy.spatial.transform import Rotation

from tools.bag.keyframes import body_to_camera, compensated_displacements, flight_window, ParallaxSelector
from tools.bag.reader import inspect_bag, iter_images
from tools.extract_bag_keyframes import extract
from tools.run_bag_pipeline import write_reconstruction_config

ROOT = Path(__file__).resolve().parents[1]


from tools.bag.keyframes import selection_coverage


class BagPipelineTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)

    def test_body_extrinsics_preserve_the_same_physical_projection(self):
        rng = np.random.default_rng(0)
        r_cl = Rotation.from_euler('xyz', [.1, -.2, 1.2]).as_matrix()
        r_il = Rotation.from_euler('xyz', [.3, .2, -.1]).as_matrix()
        t_cl, t_il = np.array([.02, -.05, .1]), np.array([.04, .02, -.03])
        r_ci, t_ci = body_to_camera(r_cl, t_cl, r_il, t_il)
        lidar = rng.normal(size=(100, 3))
        body = lidar @ r_il.T + t_il
        np.testing.assert_allclose(body @ r_ci.T + t_ci, lidar @ r_cl.T + t_cl, atol=1e-12)

    def test_parallax_rejects_rotation_but_keeps_translation(self):
        k = np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]])
        points = np.array([[-1, -1, 10], [1, -1, 10], [-1, 1, 10], [1, 1, 10]], float)
        delta = Rotation.from_euler('xyz', [.04, -.06, .4]).as_matrix()
        project = lambda xyz: (xyz @ k.T)[:, :2] / xyz[:, 2, None]
        first = project(points)
        rotated = points @ delta.T
        errors = compensated_displacements(first, project(rotated), k, np.zeros(5), delta)
        np.testing.assert_allclose(errors, 0, atol=1e-10)
        translated = rotated + [.2, 0, 0]
        errors = compensated_displacements(first, project(translated), k, np.zeros(5), delta)
        np.testing.assert_allclose(errors, 80 / rotated[:, 2], atol=1e-10)

    def test_flight_gate_skips_climb_and_rejects_zero_telemetry(self):
        times = 1000 + np.arange(0, 100, .1)
        height = np.clip((times-1010)*2, 0, 80)
        window = flight_window(times, height, min_height=60, cruise_fraction=.5)
        self.assertGreaterEqual(window['start_time'], 1050)
        self.assertLess(window['start_time'], 1055)
        self.assertAlmostEqual(window['height_threshold_m'], 60)
        with self.assertRaisesRegex(ValueError, 'No stable airborne'):
            flight_window(times, np.zeros(len(times)))
        with self.assertRaises(ValueError):
            flight_window(times[::-1], height)

    def test_image_selection_uses_accumulated_parallax_not_stride(self):
        rng = np.random.default_rng(4)
        gray = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        k = np.array([[280., 0, 160], [0, 280, 120], [0, 0, 1]])
        selector = ParallaxSelector(k, np.zeros(5), tracking_width=320,
                                    parallax_px=12, min_tracks=30, min_baseline_m=.5)
        self.assertTrue(selector.consider(gray, np.eye(3), np.zeros(3))['accepted'])
        shifted8 = cv2.warpAffine(gray, np.float32([[1, 0, 8], [0, 1, 0]]), (320, 240))
        result = selector.consider(shifted8, np.eye(3), np.array([1., 0, 0]))
        self.assertFalse(result['accepted'])
        self.assertAlmostEqual(result['parallax_px'], 8, delta=.4)
        shifted16 = cv2.warpAffine(gray, np.float32([[1, 0, 16], [0, 1, 0]]), (320, 240))
        result = selector.consider(shifted16, np.eye(3), np.array([2., 0, 0]))
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['parallax_px'], 16, delta=.4)
        self.assertFalse(selector.consider(shifted16, np.eye(3), np.array([2., 0, 0]))['accepted'])

    def test_large_rotation_warp_retains_translation_parallax(self):
        rng = np.random.default_rng(10)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (320, 320), dtype=np.uint8), (3, 3), 0)
        k = np.array([[280., 0, 160], [0, 280, 160], [0, 0, 1]])
        delta = Rotation.from_euler('z', 65, degrees=True).as_matrix()
        homography = k @ delta @ np.linalg.inv(k)
        homography = np.array([[1, 0, 18], [0, 1, 0], [0, 0, 1]]) @ homography
        current = cv2.warpPerspective(gray, homography, (320, 320))
        selector = ParallaxSelector(k, np.zeros(5), tracking_width=320, min_tracks=30,
                                    parallax_px=12, min_track_ratio=.1)
        selector.consider(gray, np.eye(3), np.zeros(3))
        result = selector.consider(current, delta.T, np.array([1., 0, 0]))
        self.assertTrue(result['accepted'], result)
        self.assertAlmostEqual(result['parallax_px'], 18, delta=1)

    def test_coverage_allows_hover_but_rejects_missed_excursion_and_tail(self):
        times = np.arange(0., 101.)
        positions = np.zeros((len(times), 3))
        selected = [0., 10., 100.]
        result = selection_coverage(times, positions, selected, 0., 100.)
        self.assertTrue(result['complete'])
        positions[:, 0] = 80*np.maximum(0, 1-np.abs(times-55)/35)
        result = selection_coverage(times, positions, selected, 0., 100.)
        self.assertFalse(result['complete'])
        self.assertEqual(result['maximum_interval_displacement_m'], 80)
        positions[:, 0] = times
        result = selection_coverage(times, positions, [0., 10., 20.], 0., 100.)
        self.assertFalse(result['complete'])
        self.assertEqual(result['last_image_to_end_s'], 80)

    def test_original_bag_payload_timestamps_and_reuse(self):
        store = get_typestore(Stores.ROS1_NOETIC)
        Time = store.types['builtin_interfaces/msg/Time']
        Header = store.types['std_msgs/msg/Header']
        Image = store.types['sensor_msgs/msg/CompressedImage']
        Height = store.types['std_msgs/msg/Float32']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag, dataset = root/'toy.bag', root/'prepared'
            (dataset/'frontend').mkdir(parents=True)
            payloads = []
            texture = cv2.GaussianBlur(np.random.default_rng(9).integers(0,256,(240,320),dtype=np.uint8),(3,3),0)
            with Writer(bag) as writer:
                images = writer.add_connection('/camera', Image.__msgtype__, typestore=store)
                heights = writer.add_connection('/height', Height.__msgtype__, typestore=store)
                for i in range(4):
                    stamp = 1000+i
                    shifted = cv2.warpAffine(texture,np.float32([[1,0,8*i],[0,1,0]]),(320,240))
                    _, payload = cv2.imencode('.jpg',shifted)
                    payloads.append(payload.tobytes())
                    message = Image(Header(i,Time(stamp,100000000),'camera'),'mono8; jpeg compressed',payload)
                    writer.write(images,stamp*10**9,store.serialize_ros1(message,Image.__msgtype__))
                    writer.write(heights,stamp*10**9,store.serialize_ros1(Height(0.),Height.__msgtype__))
            preset = yaml.safe_load((ROOT/'config/bag/amtown.yaml').read_text())
            preset.update(image_topic='/camera', height_topic='/height', camera_image_size=[320,240],
                          camera_matrix=[280,0,160,0,280,120,0,0,1],distortion=[0]*5,
                          lidar_to_camera_R=np.eye(3).reshape(-1).tolist(),lidar_to_camera_t=[0,0,0])
            preset['selection'].update(tracking_width=320,min_tracks=30,min_track_ratio=.1,parallax_px=6)
            poses=np.array([[i,1000+i,float(i),0,0,1,0,0,0] for i in range(4)])
            np.savetxt(dataset/'frontend/pose.txt',poses)
            (dataset/'frontend/frames.json').write_text(json.dumps({'body_from_imu':np.eye(3).reshape(-1).tolist()}))
            (dataset/'frontend/fastlio-config.yaml').write_text(yaml.safe_dump({'mapping':{'extrinsic_R':np.eye(3).reshape(-1).tolist(),'extrinsic_T':[0,0,0]}}))
            with contextlib.redirect_stdout(io.StringIO()):
                result=extract(bag,dataset,preset,start_offset=0,end_offset=3)
                reused=extract(bag,dataset,preset,start_offset=0,end_offset=3)
            self.assertEqual(result,reused)
            self.assertEqual(result['selected_images'],4,result)
            for i,entry in enumerate(result['selected']):
                self.assertEqual((dataset/'images'/entry['image']).read_bytes(),payloads[i])
                self.assertAlmostEqual(entry['timestamp'],1000+i)
            timestamps=np.loadtxt(dataset/'timestamps.txt',dtype=str)
            np.testing.assert_allclose(timestamps[:,1].astype(float),np.arange(1000,1004))
            with self.assertRaisesRegex(ValueError,'settings changed'):
                extract(bag,dataset,preset,start_offset=.5,end_offset=3)
            projection=write_reconstruction_config(dataset,preset,result,8,5)
            config=yaml.safe_load(projection.read_text())
            base=yaml.safe_load(Path(config['base_config']).read_text())
            self.assertEqual(base['Data']['lio_pose_file'],'frontend/pose.txt')
            self.assertEqual(base['calibration']['source_frame'],'imu_body')
            self.assertFalse(config['Projection']['images_undistorted'])


if __name__ == '__main__':
    unittest.main()
