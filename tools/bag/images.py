"""Camera calibration and optional Kannala–Brandt to pinhole rectification."""
import cv2
import numpy as np


class BagImageProcessor:
    """Use the same output intrinsics for keyframe tracking and LiDAR projection.

    Brown/pinhole inputs retain their compressed payload. Kannala–Brandt inputs
    are remapped at the original resolution and saved as lossless PNGs; callers
    retain the original compressed images separately for provenance.
    """

    def __init__(self, config):
        model = config.get('camera_model', 'plumb_bob').lower()
        if model not in ('plumb_bob', 'kannala_brandt'):
            raise ValueError(f'Unsupported bag camera model: {model}')
        self.rectified = model == 'kannala_brandt'
        size = tuple(config['camera_image_size'])
        k = np.asarray(config['camera_matrix'], np.float64).reshape(3, 3)
        d = np.asarray(config['distortion'], np.float64)
        if not np.isfinite(k).all() or not np.isfinite(d).all() or min(size) <= 0:
            raise ValueError('Invalid camera calibration')
        self.maps = None
        if self.rectified:
            if config.get('images_undistorted', False):
                raise ValueError('Kannala–Brandt input must contain the original distorted images')
            balance = float(config.get('rectification_balance', 0.0))
            if d.shape != (4,) or not 0 <= balance <= 1:
                raise ValueError('Kannala–Brandt requires four coefficients and balance in [0, 1]')
            k_out = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k, d, size, np.eye(3), balance=balance, new_size=size)
            self.maps = cv2.fisheye.initUndistortRectifyMap(
                k, d, np.eye(3), k_out, size, cv2.CV_32FC1)
            k, d = k_out, np.zeros(5)
        self.calibration = {
            'camera_matrix': k.reshape(-1).tolist(), 'distortion': d.tolist(),
            'camera_image_size': list(size),
            'images_undistorted': self.rectified or config.get('images_undistorted', False),
        }
        self.metadata = {
            'method': 'kannala_brandt_to_pinhole' if self.rectified else 'original_payload',
            'input_camera_model': model, 'output_calibration': self.calibration,
        }

    def decode(self, payload):
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Compressed image decode failed')
        if list(image.shape[1::-1]) != self.calibration['camera_image_size']:
            raise ValueError('Image dimensions differ from calibration')
        if self.maps is not None:
            image = cv2.remap(image, *self.maps, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        return image


def image_time_origin(info, config):
    """Select an explicit clock for user offsets; never infer clock corrections."""
    basis = config.get('selection_time_basis', 'bag_record')
    if basis == 'bag_record':
        return info['record_start']
    if basis == 'image_header':
        return info['image']['first_header_time'] + config['image_time_offset_s']
    raise ValueError(f'Unsupported selection_time_basis: {basis}')
