"""Canonical geometry: OpenCV world->camera extrinsics and camera-z depth."""
import numpy as np


def camera_to_extrinsic(poses):
    poses = np.asarray(poses)
    rotation = poses[..., :3, :3].swapaxes(-1, -2)
    translation = -np.einsum('...ij,...j->...i', rotation, poses[..., :3, 3])
    return np.concatenate([rotation, translation[..., None]], axis=-1).astype(np.float32)


def depth_to_world(depth, extrinsics, intrinsics):
    depth = np.asarray(depth, dtype=np.float32)
    n, height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    world = np.empty((n, height, width, 3), dtype=np.float32)
    for frame in range(n):
        k, ext = intrinsics[frame], extrinsics[frame]
        local = np.stack([(x-k[0, 2])/k[0, 0]*depth[frame],
                          (y-k[1, 2])/k[1, 1]*depth[frame], depth[frame]], axis=-1)
        world[frame] = (local-ext[:, 3]) @ ext[:, :3]
    return world


def fit_intrinsics_from_local_points(local_points, confidence):
    """Pi3 has no K head: robust approximate pinhole K fitted to local rays.

    Calibrated LiDAR projection does not use this approximation. Fit against
    integer pointmap pixel indices, the same convention as all other backends.
    """
    n, height, width, _ = local_points.shape
    y, x = np.mgrid[:height, :width]
    pixels = np.stack([x, y], axis=-1).reshape(-1, 2)
    matrices = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    for frame in range(n):
        xyz = local_points[frame].reshape(-1, 3)
        conf = confidence[frame].reshape(-1)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(conf) & (xyz[:, 2] > 0)
        valid &= conf >= max(0.1, np.percentile(conf[valid], 25)) if valid.any() else False
        selected = np.flatnonzero(valid)[::8]
        if len(selected) < 16:
            raise ValueError('Not enough Pi3 local rays to estimate approximate intrinsics')
        rays = xyz[selected, :2] / xyz[selected, 2:]
        for axis in range(2):
            a = np.column_stack([rays[:, axis], np.ones(len(rays))])
            target = pixels[selected, axis]
            mask = np.ones(len(a), dtype=bool)
            for _ in range(4):
                solution, _, rank, _ = np.linalg.lstsq(a[mask], target[mask], rcond=None)
                error = np.abs(a @ solution-target)
                updated = error <= max(1., 3.*np.median(error))
                if updated.sum() < 16 or np.array_equal(updated, mask):
                    break
                mask = updated
            if rank < 2 or not np.isfinite(solution).all() or solution[0] <= 0:
                raise ValueError('Pi3 local rays produced invalid approximate intrinsics')
            matrices[frame, axis, axis], matrices[frame, axis, 2] = solution
    return matrices


def validate_predictions(arrays, frames):
    if 'world_points' not in arrays or arrays['world_points'].ndim != 4:
        raise ValueError('Expected world_points with shape (N, H, W, 3)')
    n, h, w, xyz = arrays['world_points'].shape
    if n != frames or xyz != 3:
        raise ValueError('Pointmap frame count does not match the input images')
    expected = {'world_points_conf': (n,h,w), 'depth': (n,h,w), 'depth_conf': (n,h,w),
                'extrinsic': (n,3,4), 'intrinsic': (n,3,3), 'images': (n,3,h,w)}
    for key, shape in expected.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f'Invalid {key} shape; expected {shape}')
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
            raise ValueError(f'Non-finite/non-array prediction: {key}')
    if np.any(arrays['depth'] <= 0) or np.any(arrays['world_points_conf'] < 0):
        raise ValueError('Expected positive camera-z depth and nonnegative confidence')
    if arrays['images'].min() < 0 or arrays['images'].max() > 1:
        raise ValueError('Expected RGB images in [0, 1]')
    if np.any(arrays['intrinsic'][:, [0,1], [0,1]] <= 0):
        raise ValueError('Expected positive focal lengths')
    r = arrays['extrinsic'][:, :, :3]
    if not np.allclose(r @ r.swapaxes(1,2), np.eye(3), atol=2e-3) or np.any(np.linalg.det(r) <= 0):
        raise ValueError('Extrinsic rotations must be proper orthogonal rotations')
