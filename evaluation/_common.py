"""Shared input/output conventions for the paper's color map metrics."""
import json
from pathlib import Path

import numpy as np
import open3d as o3d

PAPER = "https://arxiv.org/html/2511.01186v1#S3.SS4"


def colored_arrays(cloud, name="point cloud"):
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if len(points) == 0:
        raise ValueError(f"{name} is empty")
    if colors.shape != points.shape:
        raise ValueError(f"{name} must have RGB colors for every point")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError(f"{name} contains non-finite coordinates or colors")
    if np.any(colors < 0) or np.any(colors > 1):
        raise ValueError(f"{name} RGB channels must be normalized to [0, 1]")
    return points, colors


def read_cloud(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    cloud = o3d.io.read_point_cloud(str(path))
    colored_arrays(cloud, str(path))
    return cloud


def validate_search(thread_num, batch_size):
    if not isinstance(thread_num, int) or (thread_num != -1 and thread_num < 1):
        raise ValueError("threads must be a positive integer or -1")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def write_report(path, report):
    """Write strict JSON; perfect fidelity is represented by the string '+inf'."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return "+inf" if value == np.inf else "-inf" if value == -np.inf else None
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean({"paper": PAPER, "rgb_channels": "[0, 1]", **report}),
                               indent=2, allow_nan=False) + "\n")
