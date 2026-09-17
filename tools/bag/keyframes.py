"""Takeoff gating and rotation-compensated parallax keyframes."""
import cv2
import numpy as np


def flight_window(times, heights, *, min_height=20., cruise_fraction=.8,
                  max_climb_speed=1., stable_seconds=3.):
    times, heights = np.asarray(times, float), np.asarray(heights, float)
    if len(times) < 3 or times.shape != heights.shape or not np.isfinite([times, heights]).all():
        raise ValueError('Need finite height samples for takeoff detection')
    if np.any(np.diff(times) <= 0) or min_height <= 0 or not 0 < cruise_fraction <= 1 or stable_seconds <= 0:
        raise ValueError('Invalid takeoff samples or thresholds')
    baseline = float(np.median(heights[times <= times[0]+3]))
    relative = heights - baseline
    threshold = max(min_height, cruise_fraction * float(np.percentile(relative, 75)))
    velocity = (np.interp(times+.5, times, heights) - np.interp(times-.5, times, heights))
    eligible = (relative >= threshold) & (np.abs(velocity) <= max_climb_speed)
    start, stable_start = None, None
    for t, ok in zip(times, eligible):
        if not ok:
            stable_start = None
        elif stable_start is None:
            stable_start = t
        elif t - stable_start >= stable_seconds:
            start = float(t)
            break
    if start is None:
        raise ValueError('No stable airborne interval; specify --start-offset or adjust takeoff thresholds')
    above = times[(times >= start) & (relative >= threshold)]
    if not len(above):
        raise ValueError('No usable flight interval')
    return {'start_time': start, 'end_time': float(above[-1]), 'baseline_height_m': baseline,
            'height_threshold_m': threshold, 'max_climb_speed_m_s': max_climb_speed,
            'stable_seconds': stable_seconds}



def selection_coverage(pose_times, positions, selected_times, start, end, *, max_unobserved_motion_m=30.):
    """Detect missed moving segments while allowing long stationary hovers.

    For every interval between selected images (including the flight boundaries),
    measure the largest LIO displacement from the interval's start. This also
    catches excursions that return to the same position before the next image.
    """
    pose_times, positions = np.asarray(pose_times, float), np.asarray(positions, float)
    selected_times = np.asarray(selected_times, float)
    if (len(pose_times) < 2 or positions.shape != (len(pose_times), 3)
            or len(selected_times) < 3 or np.any(np.diff(pose_times) <= 0)
            or np.any(np.diff(selected_times) <= 0) or not start < end
            or not np.isfinite(pose_times).all() or not np.isfinite(positions).all()
            or not np.isfinite(selected_times).all() or not max_unobserved_motion_m > 0):
        raise ValueError('Invalid trajectory or selected timestamps for coverage check')
    if start < pose_times[0] or end > pose_times[-1] or selected_times[0] < start or selected_times[-1] > end:
        raise ValueError('Selection and flight interval must be inside the trajectory')
    boundaries = np.r_[start, selected_times, end]
    intervals = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        origin = np.array([np.interp(lo, pose_times, positions[:, axis]) for axis in range(3)])
        endpoint = np.array([np.interp(hi, pose_times, positions[:, axis]) for axis in range(3)])
        inside = positions[(pose_times > lo) & (pose_times < hi)]
        excursion = float(np.linalg.norm(np.vstack([inside, endpoint])-origin, axis=1).max())
        intervals.append({'start_time': float(lo), 'end_time': float(hi),
                          'seconds': float(hi-lo), 'max_displacement_m': excursion})
    worst = max(intervals, key=lambda interval: interval['max_displacement_m'])
    failures = [interval for interval in intervals if interval['max_displacement_m'] > max_unobserved_motion_m]
    return {'complete': not failures, 'max_unobserved_motion_m': max_unobserved_motion_m,
            'maximum_interval_displacement_m': worst['max_displacement_m'], 'worst_interval': worst,
            'first_image_delay_s': float(selected_times[0]-start),
            'last_image_to_end_s': float(end-selected_times[-1]),
            'intervals_checked': len(intervals), 'failed_intervals': failures}


def body_to_camera(r_cl, t_cl, r_il, t_il):
    """Compose camera<-LiDAR with LiDAR<-IMU; exported body equals IMU."""
    r_cl, r_il = np.asarray(r_cl).reshape(3, 3), np.asarray(r_il).reshape(3, 3)
    r_ci = r_cl @ r_il.T
    return r_ci, np.asarray(t_cl) - r_ci @ np.asarray(t_il)


def compensated_displacements(previous, current, intrinsic, distortion, current_from_previous):
    """Undistorted pixel displacement after removing camera rotation."""
    previous = np.asarray(previous, np.float64).reshape(-1, 1, 2)
    current = np.asarray(current, np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(previous, intrinsic, distortion).reshape(-1, 2)
    rays = np.column_stack([rays, np.ones(len(rays))]) @ current_from_previous.T
    observed = cv2.undistortPoints(current, intrinsic, distortion, P=intrinsic).reshape(-1, 2)
    projected = rays @ intrinsic.T
    valid = (rays[:, 2] > 0) & np.isfinite(projected).all(axis=1)
    expected = projected[valid, :2] / projected[valid, 2, None]
    return np.linalg.norm(observed[valid] - expected, axis=1)


class ParallaxSelector:
    """Rotation-warped LK, with SIFT + geometric verification for sharp turns.

    Both methods only establish pixel correspondences. Parallax is always the
    displacement after subtracting the known camera rotation, and acceptance
    always requires sufficient physical camera translation.
    """
    def __init__(self, intrinsic, distortion, *, tracking_width=640, parallax_px=12.,
                 min_tracks=60, min_track_ratio=.25, min_baseline_m=.5, max_features=800):
        if tracking_width < 32 or parallax_px <= 0 or min_tracks < 4 or not 0 < min_track_ratio <= 1 or min_baseline_m < 0:
            raise ValueError('Invalid parallax selector thresholds')
        self.intrinsic = np.asarray(intrinsic, float).reshape(3, 3)
        self.distortion = np.asarray(distortion, float)
        self.width, self.threshold, self.min_tracks = tracking_width, parallax_px, min_tracks
        self.min_ratio, self.min_baseline, self.max_features = min_track_ratio, min_baseline_m, max_features
        self.gray = self.remap = self.descriptors = None
        self.sift = cv2.SIFT_create(nfeatures=2000)

    def _lk(self, gray, k, delta):
        homography = k @ delta @ np.linalg.inv(k)
        warped = cv2.warpPerspective(self.gray, homography, (gray.shape[1], gray.shape[0]))
        predicted = cv2.perspectiveTransform(self.features, homography)
        xy = predicted.reshape(-1, 2)
        inside = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 2) & (xy[:, 0] < gray.shape[1]-2)
        inside &= (xy[:, 1] >= 2) & (xy[:, 1] < gray.shape[0]-2)
        candidates = predicted[inside]
        if len(candidates) < self.min_tracks:
            return None
        tracked, ok, _ = cv2.calcOpticalFlowPyrLK(warped, gray, candidates, None, winSize=(31, 31), maxLevel=4)
        if tracked is None:
            return None
        reverse, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, warped, tracked, None, winSize=(31, 31), maxLevel=4)
        if reverse is None:
            return None
        fb = np.linalg.norm(reverse-candidates, axis=2).ravel()
        xy = tracked.reshape(-1, 2)
        valid = (ok.ravel() > 0) & (back_ok.ravel() > 0) & (fb <= 1.5)
        valid &= np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1])
        valid &= (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
        ratio = float(valid.sum()/len(self.features))
        if valid.sum() < self.min_tracks or ratio < self.min_ratio:
            return None
        return self.features[inside][valid], tracked[valid], ratio

    def _descriptor_matches(self, gray):
        if self.descriptors is None:
            self.descriptors = self.sift.detectAndCompute(self.gray, None)
        anchor_kp, anchor_desc = self.descriptors
        kp, desc = self.sift.detectAndCompute(gray, None)
        if anchor_desc is None or desc is None or len(desc) < 2:
            return None
        matches = cv2.BFMatcher().knnMatch(anchor_desc, desc, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .75*m[1].distance]
        if len(good) < self.min_tracks:
            return None
        source = np.float32([anchor_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if inliers is None:
            return None
        valid = inliers.ravel() > 0
        if valid.sum() < self.min_tracks or valid.mean() < .5:
            return None
        # Distributed support avoids accepting a single repeating image patch.
        cells = (target[valid, 0] / np.array([gray.shape[1]/4, gray.shape[0]/4])).astype(int)
        if len(np.unique(cells, axis=0)) < 4:
            return None
        return source[valid], target[valid], float(valid.mean())

    def consider(self, gray, rotation_wc, camera_center):
        h, w = gray.shape
        new_h = round(h*self.width/w)
        gray = cv2.resize(gray, (self.width, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = self.width/w, new_h/h
        k = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]]) @ self.intrinsic
        if self.remap is None:
            self.remap = cv2.initUndistortRectifyMap(k, self.distortion, np.eye(3), k,
                                                    (self.width, new_h), cv2.CV_32FC1)
        gray = cv2.remap(gray, *self.remap, interpolation=cv2.INTER_LINEAR)
        report = {'accepted': False, 'reason': 'insufficient_parallax', 'parallax_px': None,
                  'tracked_points': 0, 'baseline_m': None, 'tracking_width': self.width}
        if self.gray is not None:
            if gray.shape != self.gray.shape:
                raise ValueError('Image dimensions changed within the bag')
            baseline = float(np.linalg.norm(camera_center-self.center))
            report['baseline_m'] = baseline
            if baseline < self.min_baseline:
                report['reason'] = 'insufficient_baseline'
                return report
            delta = rotation_wc.T @ self.rotation
            matches = self._lk(gray, k, delta)
            method = 'rotation_warped_lk'
            if matches is None:
                matches = self._descriptor_matches(gray)
                method = 'sift_geometric_recovery'
            if matches is None:
                report['reason'] = 'insufficient_tracks'
                return report
            source, target, ratio = matches
            distances = compensated_displacements(source, target, k, np.zeros(5), delta)
            if len(distances) < self.min_tracks:
                report['reason'] = 'insufficient_visible_tracks'
                return report
            parallax = float(np.median(distances))
            report.update(parallax_px=parallax, tracked_points=len(distances),
                          tracking_method=method, track_ratio=ratio)
            if parallax < self.threshold:
                return report
            report.update(accepted=True, reason='parallax')
        else:
            report.update(accepted=True, reason='initial_anchor')
        features = cv2.goodFeaturesToTrack(gray, self.max_features, .01, 8, blockSize=7)
        if features is None or len(features) < self.min_tracks:
            report.update(accepted=False, reason='insufficient_features')
            return report
        self.gray, self.features, self.descriptors = gray, features, None
        self.rotation, self.center = rotation_wc.copy(), camera_center.copy()
        return report
