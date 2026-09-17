"""Plot the flight gate, camera keyframe coverage and measured parallax."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def preview(dataset):
    report = json.loads((dataset/'selection.json').read_text())
    info = json.loads((dataset/'bag_info.json').read_text())
    poses = np.loadtxt(dataset/'frontend/pose.txt', ndmin=2)
    window = report['flight_window']
    zero = window.get('time_origin', info['record_start'])
    selected = report['selected']
    stamps = np.array([s['timestamp'] for s in selected])
    xyz = np.column_stack([np.interp(stamps, poses[:, 1], poses[:, i]) for i in range(2, 5)])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    height = poses[:, 4] - window.get('baseline_height_m', 0)
    ax.plot(poses[:, 1]-zero, height, lw=1, label='Gravity-aligned LIO height')
    ax.axvspan(window['start_time']-zero, window['end_time']-zero, alpha=.18, color='green', label='Selected flight window')
    if 'height_threshold_m' in window:
        ax.axhline(window['height_threshold_m'], ls='--', color='gray', label='Height gate')
    ax.set(xlabel='Seconds from selection time origin', ylabel='Height (m)', title='Takeoff / landing gate')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(poses[:, 2], poses[:, 3], c='.7', lw=1)
    points = ax.scatter(xyz[:, 0], xyz[:, 1], c=stamps-zero, s=5, cmap='viridis')
    fig.colorbar(points, ax=ax, label='Seconds from selection time origin')
    ax.set(xlabel='LIO world X (m)', ylabel='LIO world Y (m)', title='Selected image positions', aspect='equal')
    ax = axes[1, 0]
    measured = [s for s in selected if s.get('parallax_px') is not None]
    for method, color in [('rotation_warped_lk', '#246da8'), ('sift_geometric_recovery', '#d87925')]:
        values = [s for s in measured if s.get('tracking_method') == method]
        if values:
            ax.scatter([s['timestamp']-zero for s in values], [s['parallax_px'] for s in values], s=5, color=color, label=method)
    ax.axhline(report['request']['config']['selection']['parallax_px'], ls='--', color='gray', label='Acceptance threshold')
    width = report['request']['config']['selection']['tracking_width']
    ax.set(xlabel='Seconds from selection time origin', ylabel=f'Pixels at {width}px width', title='Rotation-compensated parallax')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(stamps[1:]-zero, np.diff(stamps), '.', ms=3)
    ax.set(xlabel='Seconds from selection time origin', ylabel='Seconds', title='Time between selected images (hovers create gaps)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"{dataset.name} | {len(selected):,} selected / {report['counts']['bag_images']:,} bag images")
    path = dataset/'selection_preview.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_dir', type=Path)
    preview(parser.parse_args().dataset_dir)
