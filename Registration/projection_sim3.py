"""Pixel-correspondence Sim(3) registration. All residual thresholds are in meters.

Column convention: target_world = scale * rotation @ vggt_world + translation.
No initial transform or nearest-neighbor correspondence search is used.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Sim3Estimate:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    residuals: np.ndarray
    iterations: int

    @property
    def matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def fit_sim3(source, target):
    """Umeyama fit; accept planar data, reject collinear/constant data."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Need at least three corresponding 3D point pairs')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Point pairs must be finite')
    sx, sy = source.mean(axis=0), target.mean(axis=0)
    x, y = source - sx, target - sy
    covariance = y.T @ x / len(x)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[0] < 1e-15 or singular[1] < singular[0] * 1e-8:
        raise ValueError('Degenerate (constant or collinear) correspondence sample')
    correction = np.ones(3)
    correction[-1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = (u * correction) @ vt
    variance = np.mean(np.sum(x * x, axis=1))
    scale = float(np.sum(singular * correction) / variance)
    translation = sy - scale * rotation @ sx
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Sim3 scale must be finite and positive')
    return scale, rotation, translation


def residuals(source, target, scale, rotation, translation):
    return np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)


def error_metrics(errors, threshold):
    """Summarize paired residuals in meters using an explicit inlier threshold."""
    errors = np.asarray(errors)
    inside = errors <= threshold
    return {'pairs': len(errors), 'inliers': int(inside.sum()),
            'inlier_ratio': float(inside.mean()) if len(errors) else 0.,
            'rmse_all_m': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'median_m': float(np.median(errors)) if len(errors) else None,
            'inlier_rmse_m': float(np.sqrt(np.mean(errors[inside]**2))) if inside.any() else None}


def ransac_sim3(source, target, threshold=2.0, max_iterations=5000, confidence=0.999,
                 min_inliers=30, min_inlier_ratio=0.1, scoring_points=10000, seed=0):
    """Three-pair RANSAC, adaptive stopping, then refit on the full consensus.

    Hypotheses are scored on a fixed random subset for speed; the chosen model
    and every refinement are verified against ALL input pairs before returning.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('source and target must have matching (N, 3) shapes')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('RANSAC input must be finite')
    if not np.isfinite(threshold) or threshold <= 0 or max_iterations < 1 or scoring_points < 3:
        raise ValueError('Invalid RANSAC threshold, iteration count, or scoring size')
    if not 0 < confidence < 1 or not 0 <= min_inlier_ratio <= 1 or min_inliers < 3:
        raise ValueError('Invalid RANSAC confidence or consensus requirement')
    required = max(min_inliers, math.ceil(len(source) * min_inlier_ratio))
    if len(source) < required:
        raise ValueError(f'Only {len(source)} pairs; need at least {required}')
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(source), min(scoring_points, len(source)), replace=False)
    x, y = source[selected], target[selected]
    best, best_score, limit = None, (-1, -np.inf), max_iterations
    iteration = 0
    while iteration < limit:
        iteration += 1
        sample = rng.choice(len(source), 3, replace=False)
        try:
            candidate = fit_sim3(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = residuals(x, y, *candidate)
        mask = errors <= threshold
        count = int(mask.sum())
        score = (count, -float(np.minimum(errors, threshold).sum()))
        if score > best_score:
            # Local optimization on the subset improves noisy minimal samples.
            if count >= 3:
                try:
                    refined = fit_sim3(x[mask], y[mask])
                    refined_errors = residuals(x, y, *refined)
                    refined_score = (int((refined_errors <= threshold).sum()),
                                     -float(np.minimum(refined_errors, threshold).sum()))
                    if refined_score > score:
                        candidate, score = refined, refined_score
                except (ValueError, np.linalg.LinAlgError):
                    pass
            best, best_score = candidate, score
            ratio = score[0] / len(x)
            if ratio >= 1:
                needed = 1
            elif ratio > 0:
                needed = math.ceil(math.log1p(-confidence) / math.log1p(-ratio**3))
            else:
                needed = max_iterations
            limit = min(limit, max(64, needed))
    if best is None:
        raise ValueError('RANSAC found no nondegenerate Sim3 hypothesis')
    errors = residuals(source, target, *best)
    mask = errors <= threshold
    if mask.sum() < required:
        raise ValueError(f'RANSAC consensus too small: {mask.sum()}/{len(mask)}, need {required}')
    # Least-squares refit on ALL consensus pairs, reclassifying until stable.
    # A tiny decrease in inlier count must not prevent the full consensus fit.
    for _ in range(10):
        candidate = fit_sim3(source[mask], target[mask])
        updated = residuals(source, target, *candidate)
        updated_mask = updated <= threshold
        if updated_mask.sum() < required:
            raise ValueError('Full-consensus refit no longer meets the RANSAC acceptance threshold')
        unchanged = np.array_equal(updated_mask, mask)
        best, errors, mask = candidate, updated, updated_mask
        if unchanged:
            break
    return Sim3Estimate(*best, mask, errors, iteration)


def preprocess_affine(original_wh, output_hw, mode='crop', target_size=518):
    """Original pixel centers -> exact VGGT/PIL resized, cropped/padded centers."""
    width, height = original_wh
    out_h, out_w = output_hw
    if min(width, height, out_h, out_w) <= 0 or mode not in ('crop', 'pad'):
        raise ValueError('Invalid image dimensions or preprocessing mode')
    if mode == 'pad' and height > width:
        new_h = target_size
        new_w = round(width * target_size / height / 14) * 14
    else:
        new_w = target_size
        new_h = round(height * target_size / width / 14) * 14
    cropped_h = min(new_h, target_size) if mode == 'crop' else new_h
    pad_h = target_size if mode == 'pad' else cropped_h
    pad_w = target_size if mode == 'pad' else new_w
    if new_h <= 0 or new_w <= 0 or out_h < pad_h or out_w < pad_w:
        raise ValueError('Pointmap shape does not match VGGT preprocessing')
    offset_x = (pad_w - new_w) // 2 + (out_w - pad_w) // 2
    offset_y = -(new_h - cropped_h) // 2 + (pad_h - cropped_h) // 2 + (out_h - pad_h) // 2
    sx, sy = new_w / width, new_h / height
    return np.array([[sx, 0, offset_x + (sx - 1) / 2],
                     [0, sy, offset_y + (sy - 1) / 2], [0, 0, 1.]])


class LidarTrajectory:
    """Rows: frame_id timestamp x y z qw qx qy qz; rotations are world<-LiDAR."""
    def __init__(self, rows):
        self.rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        if self.rows.shape[1] != 9 or len(self.rows) < 2 or not np.isfinite(self.rows).all():
            raise ValueError('Expected at least two finite LiDAR pose rows with nine columns')
        self.times = self.rows[:, 1]
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('LiDAR timestamps must be strictly increasing')
        self.rotations = Rotation.from_quat(self.rows[:, [6, 7, 8, 5]])
        self.slerp = Slerp(self.times, self.rotations)

    def closest(self, stamp, tolerance):
        idx = int(np.searchsorted(self.times, stamp))
        choices = [i for i in (idx - 1, idx) if 0 <= i < len(self.times)]
        idx = min(choices, key=lambda i: abs(self.times[i] - stamp))
        if abs(self.times[idx] - stamp) > tolerance:
            raise ValueError(f'No LiDAR scan within {tolerance}s of image time {stamp}')
        return idx

    def at(self, stamp):
        if not self.times[0] <= stamp <= self.times[-1]:
            raise ValueError('Image timestamp lies outside the LiDAR trajectory')
        rotation = self.slerp([stamp]).as_matrix()[0]
        translation = np.array([np.interp(stamp, self.times, self.rows[:, axis]) for axis in (2, 3, 4)])
        return rotation, translation

    def world_and_camera(self, points, scan_idx, image_stamp, r_cl, t_cl, compensate=True):
        world = points @ self.rotations[scan_idx].as_matrix().T + self.rows[scan_idx, 2:5]
        if compensate:
            r_wl, t_wl = self.at(image_stamp)
            image_lidar = (world - t_wl) @ r_wl
        else:
            image_lidar = points
        camera = image_lidar @ np.asarray(r_cl).T + np.asarray(t_cl)
        return world, camera


def project_visible(camera_points, intrinsic, image_hw, distortion=None, min_depth=0.1, max_depth=np.inf):
    """Project camera-frame points and keep the frontmost point per pixel (z-buffer)."""
    points = np.asarray(camera_points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    height, width = image_hw
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsic matrix')
    if not np.allclose(intrinsic[2], [0, 0, 1]) or not np.isclose(intrinsic[0, 1], 0):
        raise ValueError('Expected a zero-skew pinhole camera matrix')
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > min_depth) & (points[:, 2] < max_depth)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return indices, np.empty((0, 2))
    p = points[indices]
    if distortion is not None and np.any(distortion):
        uv = cv2.projectPoints(p, np.zeros(3), np.zeros(3), intrinsic,
                               np.asarray(distortion, dtype=np.float64))[0].reshape(-1, 2)
    else:
        uv = (p @ intrinsic.T)[:, :2] / p[:, 2:]
    valid = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    indices, uv = indices[valid], uv[valid]
    pixels = np.floor(uv + 0.5).astype(np.int64)
    flat = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((points[indices, 2], flat))
    _, first = np.unique(flat[order], return_index=True)
    selected = order[first]
    return indices[selected], uv[selected]


def sample_pointmap(pointmap, confidence, depth, uv, min_confidence=0., max_depth_jump=0.05):
    """Bilinear pointmap lookup; reject low-confidence/depth-edge/invalid samples.

    Returns sampled 3D coordinates and a mask into the supplied UV array.
    Relative depth discontinuity is scale invariant (VGGT depth is not metric).
    """
    pointmap, confidence, depth = np.asarray(pointmap), np.asarray(confidence), np.asarray(depth)
    if pointmap.ndim != 3 or pointmap.shape[2] != 3 or confidence.shape != pointmap.shape[:2] or depth.shape != confidence.shape:
        raise ValueError('Pointmap, confidence and depth shapes do not agree')
    height, width = confidence.shape
    uv = np.asarray(uv)
    valid = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] < height - 1)
    selected = np.flatnonzero(valid)
    coords = uv[selected]
    x, y = np.floor(coords).astype(int).T
    fx, fy = (coords - np.floor(coords)).T
    corners = np.stack([pointmap[y, x], pointmap[y, x + 1], pointmap[y + 1, x], pointmap[y + 1, x + 1]], axis=1)
    confs = np.stack([confidence[y, x], confidence[y, x + 1], confidence[y + 1, x], confidence[y + 1, x + 1]], axis=1)
    depths = np.stack([depth[y, x], depth[y, x + 1], depth[y + 1, x], depth[y + 1, x + 1]], axis=1)
    good = np.isfinite(corners).all(axis=(1, 2)) & np.isfinite(confs).all(axis=1) & np.isfinite(depths).all(axis=1)
    good &= (confs >= min_confidence).all(axis=1) & (depths > 0).all(axis=1)
    if max_depth_jump > 0:
        good &= (depths.max(axis=1) - depths.min(axis=1)) <= max_depth_jump * np.maximum(depths.min(axis=1), 1e-12)
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    samples = np.sum(corners[good] * weights[good, :, None], axis=1)
    valid[:] = False
    valid[selected[good]] = True
    return samples, valid
