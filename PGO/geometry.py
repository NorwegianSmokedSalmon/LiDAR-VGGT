"""Metric SE(3), with column transforms T_target_from_source."""
import numpy as np
from scipy.spatial.transform import Rotation


def invert(pose):
    pose = np.asarray(pose)
    result = np.broadcast_to(np.eye(4), pose.shape).copy()
    result[..., :3, :3] = pose[..., :3, :3].swapaxes(-1, -2)
    result[..., :3, 3] = -np.einsum('...ij,...j->...i', result[..., :3, :3], pose[..., :3, 3])
    return result


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def metric_camera_poses(extrinsics, sim3):
    """Scale camera centers, never rotation matrices. Return world<-camera, scale."""
    matrix = np.asarray(sim3, dtype=float)
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    if not np.isfinite(matrix).all() or scale <= 0 or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Invalid positive-scale Sim(3)')
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError('Sim(3) has anisotropic scale or an invalid rotation')
    ext = np.asarray(extrinsics, dtype=float)
    cameras = np.tile(np.eye(4), (len(ext), 1, 1))
    # Remove float32 orthogonality roundoff before composing poses.
    camera_rotation = Rotation.from_matrix(ext[:, :3, :3]).inv().as_matrix()
    centers = -np.einsum('nij,nj->ni', camera_rotation, ext[:, :3, 3])
    cameras[:, :3, :3] = rotation @ camera_rotation
    cameras[:, :3, 3] = scale * (centers @ rotation.T) + matrix[:3, 3]
    return cameras, float(scale)


def fit_rigid(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Need at least three matching 3D point pairs')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite correspondence')
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(y.T @ x)
    if singular[0] <= 1e-12 or singular[1] < singular[0] * 1e-7:
        raise ValueError('Degenerate correspondence geometry')
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    pose = np.eye(4)
    pose[:3, :3] = (u * correction) @ vt
    pose[:3, 3] = target.mean(0) - pose[:3, :3] @ source.mean(0)
    return pose


def correspondence_icp(source, target, config, seed=0):
    """Known-pixel RANSAC initialization followed by gated point-to-point ICP.

    Inputs contain fitting pixels only; tuning/test pixels never reach ICP.
    The returned transform maps source camera coordinates into target camera.
    """
    import open3d as o3d
    source, target = np.asarray(source, float), np.asarray(target, float)
    minimum = int(config['min_pairs'])
    if len(source) < minimum:
        raise ValueError(f'Too few fitting correspondences: {len(source)} < {minimum}')
    threshold = float(config['ransac_threshold_m'])
    rng = np.random.default_rng(seed)
    best, best_score = None, (-1, -np.inf)
    for iteration in range(int(config['ransac_iterations'])):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            pose = fit_rigid(source[sample], target[sample])
        except ValueError:
            continue
        errors = np.linalg.norm(transform(source, pose) - target, axis=1)
        inliers = errors < threshold
        score = (int(inliers.sum()), -float(np.minimum(errors, threshold).sum()))
        if score > best_score:
            best, best_score = pose, score
    if best is None:
        raise ValueError('No nondegenerate RANSAC hypothesis')
    for _ in range(6):
        errors = np.linalg.norm(transform(source, best) - target, axis=1)
        inliers = errors < threshold
        if inliers.sum() < minimum or inliers.mean() < config['min_inlier_ratio']:
            raise ValueError('Insufficient RANSAC consensus')
        best = fit_rigid(source[inliers], target[inliers])
    before = np.linalg.norm(transform(source, best) - target, axis=1)
    cloud_s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source[inliers]))
    cloud_t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target[inliers]))
    refined = o3d.pipelines.registration.registration_icp(
        cloud_s, cloud_t, float(config['max_correspondence_distance_m']), best,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(config['iterations'])))
    after = np.linalg.norm(transform(source, refined.transformation) - target, axis=1)
    # NN matching on a nearly planar surface can slide tangentially. Known
    # correspondences arbitrate that refinement, using fitting pixels only.
    use_icp = (refined.fitness >= config['min_inlier_ratio']
               and np.median(after) <= np.median(before))
    pose = np.asarray(refined.transformation) if use_icp else best
    errors = after if use_icp else before
    ratio = float(np.mean(errors < threshold))
    if ratio < config['min_inlier_ratio'] or np.median(errors) > config['max_median_error_m']:
        raise ValueError('ICP correspondence quality gate failed')
    return pose, {'fit_pairs': len(source), 'fit_inlier_ratio': ratio,
                  'fit_median_m': float(np.median(errors)), 'nn_icp_accepted': bool(use_icp),
                  'icp_fitness': float(refined.fitness), 'icp_rmse_m': float(refined.inlier_rmse)}
