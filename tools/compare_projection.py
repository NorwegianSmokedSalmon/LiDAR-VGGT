"""Compare both transforms on the SAME held-out pixel correspondences (meters)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Registration.projection_sim3 import error_metrics
from models.predictions import prediction_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    args = parser.parse_args()
    root, baseline_dir = args.result_dir.resolve(), args.baseline_dir.resolve()
    summary = json.loads((root / 'summary.json').read_text())
    if summary.get('method') != 'projection_ransac_sim3':
        raise ValueError('Expected correspondence registration output')
    if prediction_directory(baseline_dir) != prediction_directory(root):
        raise ValueError('Both methods must use the exact same VGGT predictions')
    baseline = np.loadtxt(baseline_dir / 'sim3_total.txt').reshape(-1, 4, 4)
    entries = summary['registration']
    if len(entries) != len(summary['inference']['chunks']) or len(baseline) < len(entries):
        raise ValueError('Incomplete registration or baseline')
    threshold = summary['residual_threshold_m']
    new_errors, old_errors, per_chunk = [], [], []
    for entry in entries:
        idx = entry['index']
        with np.load(root / 'correspondences' / f'chunk_{idx}.npz') as pairs:
            test = pairs['validation']
            source = pairs['source_vggt'][test].astype(np.float64)
            target = pairs['target_lidar_world'][test]
            errors = pairs['residual_m'][test].astype(np.float64)
        old_xyz = source @ baseline[idx, :3, :3].T + baseline[idx, :3, 3]
        old = np.linalg.norm(old_xyz - target, axis=1)
        new_errors.append(errors)
        old_errors.append(old)
        per_chunk.append({'index': idx, 'projection': error_metrics(errors, threshold),
                          'baseline': error_metrics(old, threshold)})
    new_errors, old_errors = np.concatenate(new_errors), np.concatenate(old_errors)
    report = {'metric': 'held-out 3D pixel-correspondence residual in LiDAR world (meters)',
              'note': '20% of selected pairs per frame are excluded from new Sim3 fitting. '
                      'Overlapping chunks share images; this is a paired registration diagnostic, not a map accuracy benchmark.',
              'threshold_m': threshold, 'chunks': len(entries), 'baseline_dir': str(baseline_dir),
              'projection': error_metrics(new_errors, threshold), 'baseline': error_metrics(old_errors, threshold),
              'chunks_with_lower_median': sum(e['projection']['median_m'] < e['baseline']['median_m'] for e in per_chunk),
              'per_chunk': per_chunk}
    (root / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    x = [e['index'] for e in per_chunk]
    for method, label, color in [('baseline', 'Original registration', '#d66b32'),
                                 ('projection', 'Projection + RANSAC Sim3', '#218752')]:
        axes[0].plot(x, [e[method]['median_m'] for e in per_chunk], '.-', color=color, label=label)
        axes[1].plot(x, [100*e[method]['inlier_ratio'] for e in per_chunk], '.-', color=color, label=label)
    axes[0].set_ylabel('Held-out paired median (m)')
    axes[0].legend()
    axes[1].set(xlabel='Chunk index', ylabel=f'Held-out paired inliers <= {threshold:g} m (%)', ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Same held-out pixel pairs: {len(new_errors):,} pairs, {len(entries)} chunks\n"
                 f"Global median: original {report['baseline']['median_m']:.3f} m -> projection {report['projection']['median_m']:.3f} m")
    fig.savefig(root / 'correspondence_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_chunk'}, indent=2))


if __name__ == '__main__':
    main()
