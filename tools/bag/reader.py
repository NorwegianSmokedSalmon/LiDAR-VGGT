"""ROS1 bag access on the host, without importing ROS or requiring a ROS install."""
from pathlib import Path
import cv2
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


def header_time(message):
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def bag_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def inspect_bag(path, image_topic, height_topic):
    store = get_typestore(Stores.ROS1_NOETIC)
    with Reader(Path(path)) as bag:
        report = {**bag_identity(path), 'record_start': bag.start_time / 1e9,
                  'record_end': bag.end_time / 1e9, 'duration_s': bag.duration / 1e9,
                  'topics': {c.topic: {'type': c.msgtype, 'messages': c.msgcount} for c in bag.connections}}
        connections = [c for c in bag.connections if c.topic == image_topic]
        if not connections:
            raise ValueError(f'Image topic missing: {image_topic}')
        c, record, raw = next(bag.messages(connections=connections))
        if c.msgtype != 'sensor_msgs/msg/CompressedImage':
            raise ValueError('Expected sensor_msgs/CompressedImage for lossless payload extraction')
        msg = store.deserialize_ros1(raw, c.msgtype)
        decoded = cv2.imdecode(msg.data, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError('First image cannot be decoded')
        report['image'] = {'topic': image_topic, 'format': msg.format, 'size_wh': list(decoded.shape[::-1]),
                           'first_header_time': header_time(msg), 'first_record_time': record / 1e9}
        heights = []
        connections = [c for c in bag.connections if c.topic == height_topic]
        for c, record, raw in bag.messages(connections=connections) if connections else []:
            msg = store.deserialize_ros1(raw, c.msgtype)
            # Height-above-takeoff is an unheadered Float32; use its record time.
            if not heights or record / 1e9 - heights[-1][0] >= .09:
                heights.append([record / 1e9, float(msg.data)])
        report['height_topic'] = height_topic if heights else None
        report['height_samples'] = heights
    return report


def iter_images(path, topic):
    store = get_typestore(Stores.ROS1_NOETIC)
    with Reader(Path(path)) as bag:
        connections = [c for c in bag.connections if c.topic == topic]
        if not connections or any(c.msgtype != 'sensor_msgs/msg/CompressedImage' for c in connections):
            raise ValueError(f'Missing or unsupported compressed image topic: {topic}')
        for index, (connection, recorded, raw) in enumerate(bag.messages(connections=connections)):
            msg = store.deserialize_ros1(raw, connection.msgtype)
            yield index, header_time(msg), recorded / 1e9, msg.format, msg.data
