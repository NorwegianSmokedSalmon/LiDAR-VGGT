"""Build shared dataset and calibrated projection configs from raw-bag exports."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def write_reconstruction_config(dataset, preset, selection, chunk_size, chunk_step):
    """Build a complete standalone dataset config, retaining the shared model setup."""
    base = yaml.safe_load((ROOT/'config/config.yaml').read_text())
    base['Data'] = {'base_path': str(dataset.parent), 'dataset': dataset.name,
                    'images_dir': 'images', 'body_pcd_dir': 'frontend/pcd',
                    'world_pcd_dir': 'world_pcd', 'npy_dir': 'npy',
                    'img_time_file': 'timestamps.txt', 'lio_pose_file': 'frontend/pose.txt'}
    base['Chunk'].update(chunk_size=chunk_size, chunk_step=chunk_step, start_idx=0)
    base['calibration'] = {'R': selection['body_to_camera_R'], 't': selection['body_to_camera_t'],
                           'source_frame': 'imu_body', 'target_frame': 'camera'}
    base_path = dataset/'dataset_config.yaml'
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    projection = yaml.safe_load((ROOT/'config/projection_amtown.yaml').read_text())
    projection['base_config'] = str(base_path)
    effective = selection.get('image_processing', {}).get('output_calibration', {
        'camera_matrix': preset['camera_matrix'], 'distortion': preset['distortion'],
        'camera_image_size': preset['camera_image_size'],
        'images_undistorted': preset.get('images_undistorted', False),
    })
    projection['Projection'].update(effective)
    projection['Projection'].update(calibration_source=preset['sources'],
        calibration_note='Calibration matches the saved reconstruction images; see selection.json for processing. '
                         'Body<-LiDAR extrinsic composed into camera<-body. '
                         'Image timestamps already include the configured offset exactly once.')
    path = dataset/'projection.yaml'
    path.write_text(yaml.safe_dump(projection, sort_keys=False))
    return path
