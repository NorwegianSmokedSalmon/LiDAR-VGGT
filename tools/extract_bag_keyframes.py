"""Extract calibrated camera keyframes after takeoff using compensated parallax."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import yaml
from Registration.projection_sim3 import LidarTrajectory
from tools.bag.reader import bag_identity, inspect_bag, iter_images
from tools.bag.images import BagImageProcessor, image_time_origin
from tools.bag.keyframes import ParallaxSelector, body_to_camera, flight_window, selection_coverage



def validate_coverage(dataset, report):
    rows = np.loadtxt(dataset/'frontend/pose.txt', ndmin=2)
    window = report['flight_window']
    coverage = selection_coverage(rows[:, 1], rows[:, 2:5],
        [entry['timestamp'] for entry in report['selected']], window['start_time'], window['end_time'],
        max_unobserved_motion_m=report['request']['config'].get('max_unobserved_motion_m', 30.))
    (dataset/'selection_coverage.json').write_text(json.dumps(coverage, indent=2, allow_nan=False)+'\n')
    if not coverage['complete']:
        raise ValueError('Image selection missed a moving segment; inspect selection_coverage.json and '
                         'selection_candidates.jsonl before reconstruction')
    return coverage


def extract(bag, dataset, config, *, start_offset=None, end_offset=None):
    bag, dataset = Path(bag).resolve(), Path(dataset).resolve()
    dataset.mkdir(parents=True, exist_ok=True)
    identity = bag_identity(bag)
    request = {'schema_version': 2, 'selector': 'rotation_warp_lk_sift', 'bag': identity, 'config': config,
               'start_offset_s': start_offset, 'end_offset_s': end_offset}
    report_path = dataset / 'selection.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('request') != request:
            raise ValueError('Extraction settings changed; select a new dataset directory')
        if previous.get('complete'):
            names = [f['image'] for f in previous['selected']]
            if not all((dataset/'images'/name).is_file() for name in names) or not (dataset/'timestamps.txt').is_file():
                raise ValueError('Completed selection has missing images or timestamps')
            if not all((dataset/entry['original_image']).is_file() for entry in previous['selected'] if 'original_image' in entry):
                raise ValueError('Completed rectified selection has missing original images')
            stamps = np.loadtxt(dataset/'timestamps.txt', dtype=str, ndmin=2)
            if stamps.shape != (len(names), 2) or stamps[:, 0].tolist() != names or not np.allclose(
                    stamps[:, 1].astype(float), [f['timestamp'] for f in previous['selected']], rtol=0, atol=1e-6):
                raise ValueError('Saved image timestamps differ from the completed selection')
            validate_coverage(dataset, previous)
            return previous
    images = dataset/'images'
    if images.exists() and any(images.iterdir()):
        raise ValueError('Incomplete image selection exists; preserve it and use a new dataset directory')
    info_path = dataset/'bag_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if any(info.get(key) != value for key, value in identity.items()):
            raise ValueError('Bag identity differs from saved metadata')
    else:
        info = inspect_bag(bag, config['image_topic'], config['height_topic'])
        info_path.write_text(json.dumps(info, indent=2)+'\n')
    if info['image']['size_wh'] != config['camera_image_size']:
        raise ValueError('Original bag images do not match calibration resolution')
    rows = np.loadtxt(dataset/'frontend/pose.txt')
    trajectory = LidarTrajectory(rows)
    frames = json.loads((dataset/'frontend/frames.json').read_text())
    if not np.allclose(np.asarray(frames['body_from_imu']).reshape(3, 3), np.eye(3)):
        raise ValueError('Expected IMU/body point clouds from the bag frontend')
    time_origin = image_time_origin(info, config)
    processor = BagImageProcessor(config)
    heights = np.asarray(info['height_samples'])
    telemetry_valid = (len(heights) > 3 and np.isfinite(heights).all() and
                       heights[-1, 0] >= trajectory.times[0] and heights[0, 0] <= trajectory.times[-1] and
                       np.percentile(heights[:, 1], 90)-np.percentile(heights[:, 1], 10) >= config['takeoff']['min_height'])
    if start_offset is None:
        if telemetry_valid:
            window = flight_window(heights[:, 0], heights[:, 1], **config['takeoff'])
            window['source'] = 'height_topic_record_time'
        else:
            window = flight_window(rows[:, 1], rows[:, 4], **config['takeoff'])
            window['source'] = 'gravity_aligned_lio_height'
            window['height_topic_rejected'] = 'missing, invalid or insufficient height variation'
    else:
        window = {'start_time': time_origin+start_offset, 'end_time': trajectory.times[-1],
                  'source': 'manual_'+config.get('selection_time_basis', 'bag_record')+'_offset'}
    if end_offset is not None:
        window['end_time'] = time_origin+end_offset
    window['start_time'] = max(window['start_time'], trajectory.times[0])
    window['end_time'] = min(window['end_time'], trajectory.times[-1])
    if window['end_time'] <= window['start_time']:
        raise ValueError('Empty extraction interval')
    window['start_offset_s'] = window['start_time']-time_origin
    window['end_offset_s'] = window['end_time']-time_origin
    window['time_origin'] = time_origin
    window['time_basis'] = config.get('selection_time_basis', 'bag_record')
    fastlio = yaml.safe_load((dataset/'frontend/fastlio-config.yaml').read_text())
    r_cb, t_cb = body_to_camera(config['lidar_to_camera_R'], config['lidar_to_camera_t'],
                                fastlio['mapping']['extrinsic_R'], fastlio['mapping']['extrinsic_T'])
    selection = dict(config['selection'])
    candidate_interval = selection.pop('candidate_interval_s')
    effective = processor.calibration
    selector = ParallaxSelector(effective['camera_matrix'], effective['distortion'], **selection)
    if processor.rectified:
        (dataset/'images_raw').mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    report = {'request': request, 'complete': False, 'flight_window': window,
              'body_to_camera_R': r_cb.reshape(-1).tolist(), 'body_to_camera_t': t_cb.tolist(),
              'selected': [], 'counts': {}, 'image_payload': ('rectified PNG; original JPEG/PNG bytes retained in images_raw' if processor.rectified
                                              else 'original JPEG/PNG bytes; no resize or re-encoding'),
              'image_processing': processor.metadata,
              'timestamp_formula': 'camera_header_stamp + image_time_offset_s (applied once)'}
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    counts, last_candidate, last_stamp = Counter(), -np.inf, -np.inf
    tick = time.perf_counter()
    print(f"Flight window: {window['start_offset_s']:.2f}–{window['end_offset_s']:.2f}s; {window['source']}", flush=True)
    with (dataset/'selection_candidates.jsonl').open('w') as diagnostics, (dataset/'timestamps.txt').open('w') as stamps:
        stamps.write('# image_name camera_header_time_plus_offset\n')
        for bag_index, header, recorded, image_format, payload in iter_images(bag, config['image_topic']):
            counts['bag_images'] += 1
            stamp = header+config['image_time_offset_s']
            if stamp <= last_stamp:
                raise ValueError('Image header timestamps must be strictly increasing')
            last_stamp = stamp
            if not window['start_time'] <= stamp <= window['end_time']:
                counts['outside_flight_window'] += 1
                continue
            if stamp-last_candidate < candidate_interval-1e-5:
                counts['candidate_rate_limit'] += 1
                continue
            last_candidate = stamp
            try:
                scan = trajectory.closest(stamp, .06)
            except ValueError:
                counts['no_nearby_lidar_scan'] += 1
                continue
            rectified = processor.decode(payload) if processor.rectified else None
            gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if processor.rectified else cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f'Image decode failed at bag index {bag_index}')
            if list(gray.shape[::-1]) != config['camera_image_size']:
                raise ValueError('Camera resolution changes inside the bag')
            r_wb, t_wb = trajectory.at(stamp)
            r_wc = r_wb @ r_cb.T
            center = t_wb-r_wc @ t_cb
            decision = selector.consider(gray, r_wc, center)
            decision.update(bag_index=bag_index, timestamp=stamp, raw_header_time=header,
                            record_time=recorded, scan_id=int(rows[scan, 0]),
                            scan_time_difference_s=float(trajectory.times[scan]-stamp))
            counts[decision['reason']] += 1
            if decision['accepted']:
                content = payload.tobytes()
                suffix = '.jpg' if content.startswith(b'\xff\xd8') else '.png' if content.startswith(b'\x89PNG') else None
                if suffix is None:
                    raise ValueError(f'Unsupported original compressed payload: {image_format}')
                name = f"{len(report['selected']):06d}{suffix}"
                if processor.rectified:
                    original = Path('images_raw')/name
                    (dataset/original).write_bytes(content)
                    decision['original_image'] = str(original)
                    name = str(Path(name).with_suffix('.png'))
                    if not cv2.imwrite(str(images/name), rectified):
                        raise OSError('Failed to save rectified image: '+name)
                else:
                    (images/name).write_bytes(content)
                stamps.write(f'{name} {stamp:.12f}\n')
                decision['image'] = name
                report['selected'].append(decision)
                if len(report['selected']) % 100 == 0:
                    print(f"Selected {len(report['selected'])} images; bag t={stamp-time_origin:.1f}s", flush=True)
            diagnostics.write(json.dumps(decision, allow_nan=False)+'\n')
    if len(report['selected']) < 3:
        raise ValueError('Fewer than three usable keyframes')
    times = np.array([r['timestamp'] for r in report['selected']])
    validate_coverage(dataset, report)
    report.update(complete=True, counts=dict(counts), selected_images=len(times),
                  elapsed_s=time.perf_counter()-tick, max_keyframe_gap_s=float(np.diff(times).max()),
                  median_keyframe_gap_s=float(np.median(np.diff(times))))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f"Selected {len(times)} / {counts['bag_images']} camera keyframes; max gap {report['max_keyframe_gap_s']:.2f}s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--start-offset', type=float, help='Override takeoff start in seconds; origin follows selection_time_basis in the preset')
    parser.add_argument('--end-offset', type=float)
    parser.add_argument('--parallax-px', type=float)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(0)
    config = yaml.safe_load(args.config.read_text())
    if args.parallax_px is not None:
        config['selection']['parallax_px'] = args.parallax_px
    extract(args.bag, args.dataset_dir, config, start_offset=args.start_offset, end_offset=args.end_offset)


if __name__ == '__main__':
    main()
