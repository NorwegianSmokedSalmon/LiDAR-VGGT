"""Audit completed Python PGO exports and write a before/after comparison index."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from PGO.graph import check_connected
from PGO.pipeline import odometry_metrics
from PGO.inputs import digest


def summarize(root):
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.rglob('report.json')):
        report = json.loads(path.read_text())
        if 'source_sim3_sha256' not in report:
            continue
        assert report['complete'] and report['selected']['converged'], path
        assert report['anchors_fixed'] and report['chunk_scales_fixed'], path
        assert 0 < report['selected']['icp_weight'] < report['selected']['odometry_weight'], path
        assert digest(Path(report['source_result'])/'sim3_total.txt') == report['source_sim3_sha256'], path
        assert digest(Path(report['compact_inputs'])/'inputs.json') == report['input_manifest_sha256'], path
        with np.load(path.parent/'graph.npz', allow_pickle=False) as graph:
            initial, optimized = graph['initial'], graph['optimized']
            assert initial.shape == optimized.shape == (report['nodes'], 4, 4)
            assert np.isfinite(optimized).all()
            assert np.array_equal(initial[report['fixed_nodes']], optimized[report['fixed_nodes']])
            components = check_connected(len(initial), graph['sources'], graph['targets'], report['fixed_nodes'])
            assert components == report['graph_components']
            actual_odometry = odometry_metrics(graph, optimized)
        assert len(report['maps']) == 3
        for entry in report['maps']:
            assert entry['finite_xyz_rgb'] and entry['points'] > 0
            assert digest(path.parent/entry['file']) == entry['sha256'], path
        before, after = report['test_before']['overall_m'], report['test_after']['overall_m']
        assert before['count'] == after['count'] > 0
        improving = sum(a['rmse'] < b['rmse'] for a, b in zip(report['test_after']['boundaries'], report['test_before']['boundaries']))
        rows.append({'scene': Path(report['dataset']).name.removesuffix('_raw'), 'model': report['model'],
                     'method': Path(report['source_result']).name, 'prior': report['prior_mode'],
                     'chunks': report['chunks'], 'nodes': report['nodes'], 'applied': report['applied'],
                     'test_pairs': before['count'], 'test_rmse_before_m': before['rmse'], 'test_rmse_after_m': after['rmse'],
                     'rmse_reduction_percent': 100*(1-after['rmse']/before['rmse']),
                     'improved_boundaries': improving, 'total_boundaries': len(report['test_before']['boundaries']),
                     'icp_weight': report['selected']['icp_weight'], 'world_displacement_p95_m': report['node_translation_change_m']['p95'],
                     'odometry_translation_p95_m': actual_odometry['translation_m']['p95'],
                     'odometry_rotation_p95_deg': actual_odometry['rotation_deg']['p95'],
                     'graph_components': components, 'rejected_overlap_edges': report['rejected_overlap_edges'],
                     'output': str(path.parent)})
    if not rows:
        raise ValueError(f'No PGO reports in {root}')
    (root/'validation.json').write_text(json.dumps({'complete': True, 'runs': rows}, indent=2, allow_nan=False)+'\n')
    with (root/'validation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Python PGO validation', '',
             f'{len(rows)} completed runs verified, including fixed poses, graph anchoring, source provenance and all final-map checksums.', '',
             '| Scene | Model | Source | Prior | Applied | Chunks | Test RMSE before → after (m) | ICP weight | Report |',
             '| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in rows:
        link = Path(row['output']).relative_to(root)/'README.md'
        lines.append(f'| {row["scene"]} | {row["model"]} | {row["method"]} | {row["prior"]} | {row["applied"]} | {row["chunks"]} | '
                     f'{row["test_rmse_before_m"]:.3f} → {row["test_rmse_after_m"]:.3f} | {row["icp_weight"]:g} | [result]({link}) |')
    lines += ['', 'Every fixed prior has infinite weight; model odometry weight is 10 and every ICP weight is lower.', '',
              'Pixel partitions: 60% fit, 20% tune, 20% test. Both measurements use identical test pixels, including rejected-edge observations. '
              'These are within-model overlap diagnostics, not ground-truth accuracy or a controlled ranking of models.', '',
              '`sequence_first` fixes only the first camera. Later poses can drift from LiDAR; inspect `world_displacement_p95_m` in the CSV. '
              '`chunk_first` fixes each chunk start. Rejected edges can leave multiple components, each explicitly anchored. '
              'SE(3) cannot remove scale inconsistencies inside predicted pointmaps; not every boundary necessarily improves.', '',
              'The tuning guard retains the original geometry when no candidate improves tuning RMSE. '
              'Each run preserves the resolved YAML, accepted/rejected factors, graph poses, trajectories and before/after preview.', '',
              'Details: [CSV](validation.csv), [JSON](validation.json).', '']
    (root/'README.md').write_text('\n'.join(lines))
    print(f'{len(rows)} PGO runs audited: {root/"README.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/pgo')
    summarize(parser.parse_args().root)
