#!/usr/bin/env python3
"""ROS1 container entry: FAST-LIO -> paired body PCD, indexed poses and TUM.

Adapted from the local LEMON-Mapping pipeline/run_frontend.py runner. This
pipeline initializes from the beginning of the bag; image selection skips takeoff.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xmlrpc.client

import numpy as np
import rosbag
import rospy
import rosparam
import yaml


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def master_ready(master, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if master.getPid('/lemon_pipeline')[0] == 1:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError('ROS master failed to start')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('/work/config/bag/fastlio_avia.yaml'))
    parser.add_argument('--initialization', type=Path)
    parser.add_argument('--start', type=float, default=0.0, help='Seconds from rosbag record start; nonzero requires initialization')
    parser.add_argument('--duration', type=float, default=float('inf'))
    parser.add_argument('--rate', type=float, default=4.0)
    args = parser.parse_args()
    if args.start < 0 or args.duration <= 0 or args.rate <= 0:
        parser.error('start >= 0, duration > 0 and rate > 0 are required')
    if args.start > 0 and not args.initialization:
        parser.error('Initialize FAST-LIO at bag start; crop image time separately')
    if args.output.exists() and (args.output / 'pose.json').exists():
        raise RuntimeError('Output already has poses; use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    lidar_topic = config['common']['lid_topic']
    imu_topic = config['common']['imu_topic']
    with rosbag.Bag(args.bag) as bag:
        start = bag.get_start_time() + args.start
        end = min(start + args.duration, bag.get_end_time())
        # Re-recorded bags can end with seconds of unrelated estimator output.
        # Find the last raw scan rather than assuming it is in the last second.
        scan_end, final_scan = end, None
        while scan_end > start and final_scan is None:
            scan_start = max(start, scan_end - 5.0)
            for scan in bag.read_messages(topics=[lidar_topic],
                    start_time=rospy.Time.from_sec(scan_start), end_time=rospy.Time.from_sec(scan_end)):
                final_scan = scan
            scan_end = scan_start
        if final_scan is None:
            raise RuntimeError('No LiDAR messages in the requested interval: ' + lidar_topic)
        expected_stamp = final_scan.message.header.stamp.to_sec()
    del final_scan
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:%d' % port
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ['ROS_HOME'] = str(args.output / 'ros')
    if args.initialization:
        config['initialization'] = json.loads(args.initialization.read_text())
    config['use_sim_time'] = True
    config['lemon_export']['directory'] = str(args.output)
    (args.output / 'fastlio-config.yaml').write_text(yaml.safe_dump(config))
    metadata = dict(bag=args.bag, record_start_offset_s=args.start, duration_s=end-start,
                    record_start=start, record_end=end, expected_last_lidar_stamp=expected_stamp,
                    playback_rate=args.rate, independent_initialization=True)
    (args.output / 'sequence.json').write_text(json.dumps(metadata, indent=2))
    core = front = player = None
    try:
        with (args.output / 'roscore.log').open('w') as log:
            core = subprocess.Popen(['roscore', '-p', str(port)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        master = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
        master_ready(master)
        rosparam.upload_params('/', config)
        with (args.output / 'fastlio.log').open('w') as log:
            front = subprocess.Popen(['rosrun', 'fast_lio', 'fastlio_mapping'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if front.poll() is not None:
                raise RuntimeError('FAST-LIO exited before subscribing; see fastlio.log')
            subscriptions = {topic for topic, _ in master.getSystemState('/lemon_pipeline')[2][1]}
            if {lidar_topic, imu_topic} <= subscriptions:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('FAST-LIO subscriptions did not become ready')
        cmd = ['rosbag', 'play', '--clock', '--rate', str(args.rate), '--start', str(args.start),
               '--duration', str(end-start), '--delay', '1', args.bag, '--topics', lidar_topic, imu_topic]
        with (args.output / 'rosbag-play.log').open('w') as log:
            player = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_notice = 0
        while player.poll() is None:
            if front.poll() is not None:
                raise RuntimeError('FAST-LIO exited during bag playback')
            if time.monotonic() - last_notice > 15:
                print('Playback running: start=%.1f s, output=%s' % (args.start, args.output), flush=True)
                last_notice = time.monotonic()
            time.sleep(0.5)
        if player.returncode:
            raise RuntimeError('rosbag play failed: %s' % player.returncode)
        deadline = time.monotonic() + 300
        progress = {}
        while time.monotonic() < deadline:
            if front.poll() is not None:
                raise RuntimeError('FAST-LIO exited while draining its input queue')
            status = args.output / 'progress.json'
            if status.exists():
                progress = json.loads(status.read_text())
                if progress['last_lidar_stamp'] >= expected_stamp - 0.7:
                    time.sleep(2)
                    break
            time.sleep(1)
        else:
            raise RuntimeError('FAST-LIO did not process the final scans: %s' % progress)
        stop(front)
        if front.returncode not in (0, -signal.SIGINT):
            raise RuntimeError('FAST-LIO exited abnormally: %s' % front.returncode)
        poses = np.loadtxt(args.output / 'pose.json', ndmin=2)
        stamps = np.loadtxt(args.output / 'timestamps.txt', ndmin=1)
        files = list((args.output / 'pcd').glob('*.pcd'))
        assert poses.shape == (len(stamps), 7) and len(stamps) == len(files)
        assert len(stamps) > 10 and np.isfinite(poses).all() and np.all(np.diff(stamps) > 0)
        assert np.max(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1)) < 1e-5
        assert all((args.output / 'pcd' / ('%d.pcd' % i)).exists() for i in range(len(stamps)))
        speed = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / np.diff(stamps)
        if speed.max() > 30:
            raise RuntimeError('FAST-LIO trajectory diverged: speed exceeds 30 m/s')
        tum = np.column_stack([stamps, poses[:, :3], poses[:, 4:7], poses[:, 3]])
        indexed = np.column_stack([np.arange(len(stamps)), stamps, poses])
        np.savetxt(args.output / 'trajectory.tum', tum, fmt='%.12f', header='timestamp x y z qx qy qz qw')
        np.savetxt(args.output / 'pose.txt', indexed, fmt=['%d'] + ['%.12f']*8,
                   header='frame_id timestamp x y z qw qx qy qz; world_from_body')
        frames = json.loads((args.output / 'frames.json').read_text())
        if not np.allclose(np.asarray(frames['body_from_imu']).reshape(3,3), np.eye(3)):
            raise RuntimeError('Bag pipeline expects IMU/body PCD coordinates')
        metadata.update(pose_convention='world_from_imu_body', scan_timestamp='lidar_scan_end',
                        tum_quaternion_order='xyzw', indexed_quaternion_order='wxyz',
                        max_speed_m_s=float(speed.max()), keyframes=len(stamps), first_stamp=float(stamps[0]), last_stamp=float(stamps[-1]),
                        trajectory_length_m=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()),
                        position_min=poses[:, :3].min(axis=0).tolist(), position_max=poses[:, :3].max(axis=0).tolist(),
                        progress=progress, complete=True)
        (args.output / 'sequence.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
    finally:
        stop(player)
        stop(front)
        stop(core)


if __name__ == '__main__':
    main()
