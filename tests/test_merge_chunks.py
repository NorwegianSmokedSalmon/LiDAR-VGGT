import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d
from tools.merge_chunks import merge_result


class MergeChunksTest(unittest.TestCase):
    def make_result(self, root):
        final = root / 'vggt_chunk_reg/final'
        final.mkdir(parents=True)
        chunks, registration = [], []
        points = [np.array([[10., 0, 0], [10., 1, 0]]), np.array([[10., 1, 0], [20., 2, 0]])]
        colors = [np.array([[1., 0, 0], [0., 1, 0]]), np.array([[0., 1, 0], [0., 0, 1]])]
        for idx in range(2):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))
            cloud.colors = o3d.utility.Vector3dVector(colors[idx])
            o3d.io.write_point_cloud(str(final / f'{idx}_pc.ply'), cloud)
            chunks.append({'index': idx, 'images': [f'{idx}.jpg', f'{idx + 1}.jpg']})
            registration.append({'vggt_file': f'{idx}_pc.ply'})
        (root / 'summary.json').write_text(json.dumps({'inference': {'chunks': chunks}, 'registration': registration}))
        return np.concatenate(points), np.concatenate(colors)

    def test_all_chunks_keep_world_coordinates_and_colors(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            points, colors = self.make_result(root)
            report = merge_result(root, voxel_size=0.1)
            cloud = o3d.io.read_point_cloud(str(root / 'merged_map.ply'))
            np.testing.assert_allclose(cloud.points, points)
            np.testing.assert_allclose(cloud.colors, colors)
            self.assertEqual(report['chunks'], 2)
            self.assertEqual(report['unique_images'], 3)
            self.assertEqual(report['merged_points'], 4)
            self.assertEqual(report['voxel_points'], 3)
            pcd = o3d.io.read_point_cloud(report['voxel_pcd'])
            self.assertEqual(len(pcd.points), 3)
            with self.assertRaises(FileExistsError):
                merge_result(root)

    def test_missing_chunk_fails_before_writing_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_result(root)
            (root / 'vggt_chunk_reg/final/1_pc.ply').unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                merge_result(root)
            self.assertFalse((root / 'merged_map.ply').exists())


if __name__ == '__main__':
    unittest.main()
