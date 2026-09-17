"""Summarize completed full-sequence model x scene x registration validations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.registry import MODEL_NAMES, model_spec, checkpoint_path, use_repository
from tools.download_models import verify_checkpoint


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_models(report_dir):
    records = []
    for model in MODEL_NAMES:
        spec, path = model_spec(model), checkpoint_path(model)
        use_repository(model)  # Checks the actual Git revision against the registry.
        require(verify_checkpoint(path, spec), f'Checkpoint hash/size mismatch: {model}')
        records.append({'model': model, 'weight': str(path), 'size': path.stat().st_size,
                        'sha256': spec['sha256'], 'code_revision': spec['code_revision'], 'verified': True})
        print(f'{model}: checkpoint SHA-256 and source revision verified', flush=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir/'weights.json').write_text(json.dumps(records, indent=2)+'\n')


def summarize(output_root, scenes):
    report_dir = output_root/'model_validation'
    if not (report_dir/'weights.json').exists():
        verify_models(report_dir)
    weights = read(report_dir/'weights.json')
    for model in MODEL_NAMES:
        records = [entry for entry in weights if entry['model'] == model]
        spec = model_spec(model)
        require(len(records) == 1 and records[0]['verified'] and records[0]['sha256'] == spec['sha256']
                and records[0]['code_revision'] == spec['code_revision'], f'Unverified weights/code: {model}')
    rows, inputs = [], {}
    for scene in scenes:
        for model in MODEL_NAMES:
            folder = output_root/scene/model
            comparison = read(folder/'correspondence/comparison.json')
            require(comparison['baseline_dir'] == str((folder/'baseline').resolve()),
                    f'Comparison baseline mismatch: {folder}')
            require(comparison['chunks'] == len(comparison['per_chunk']), f'Incomplete comparison: {folder}')
            require(comparison['baseline']['pairs'] == comparison['projection']['pairs'],
                    f'Methods compared on different pair counts: {folder}')
            shared = []
            for method in ('baseline', 'correspondence'):
                result = folder/method
                run, merge = read(result/'bag_run.json'), read(result/'merge_summary.json')
                validation, summary = read(result/'validation.json'), read(result/'summary.json')
                require(run['complete'] and run['full_sequence'] and validation['complete'], f'Incomplete result: {result}')
                require(run['request']['model'] == summary['model'] == model and run['method'] == method,
                        f'Method/model mismatch: {result}')
                require(run['chunks'] == merge['chunks'] == validation['chunks'] == len(summary['registration']) == comparison['chunks'],
                        f'Chunk count mismatch: {result}')
                require(run['selected_images'] == run['reconstructed_images'] == merge['unique_images']
                        == validation['original_image_payloads_checked'], f'Missing image coverage: {result}')
                require(run['request']['bag'] == validation['bag'], f'Validation bag mismatch: {result}')
                require(run['merged_points'] == merge['merged_points'], f'Point count mismatch: {result}')
                require(len(validation['maps']) == 3 and all(v['finite_xyz_rgb'] for v in validation['maps']),
                        f'Incomplete map read-back: {result}')
                for key, count in [('merged_file', 'merged_points'), ('voxel_ply', 'voxel_points'), ('voxel_pcd', 'voxel_points')]:
                    path = Path(merge[key])
                    checked = [v for v in validation['maps'] if v['file'] == path.name]
                    require(path.is_file() and len(checked) == 1 and checked[0]['points'] == merge[count],
                            f'Missing or unvalidated map: {path}')
                prediction = Path(run['prediction_dir'])
                digest = hashlib.sha256((prediction/'manifest.json').read_bytes()).hexdigest()
                require(digest == run['request']['prediction_manifest_sha256'], f'Prediction manifest changed: {result}')
                manifest = read(prediction/'manifest.json')
                require(manifest['model'] == model and len(manifest['chunks']) == run['chunks'], f'Prediction model/count mismatch: {result}')
                spec = model_spec(model)
                require(manifest.get('code_revision') == spec['code_revision'] and manifest.get('preprocess') == spec['preprocess']
                        and Path(manifest['checkpoint']).resolve() == checkpoint_path(model).resolve(),
                        f'Prediction model provenance mismatch: {result}')
                shared.append((prediction.resolve(), digest))
                identity = (run['request']['selection_sha256'], run['request']['frontend_pose_sha256'])
                if scene in inputs:
                    require(inputs[scene] == identity, f'Models use different images/trajectory: {scene}')
                inputs[scene] = identity
                paired = comparison['baseline' if method == 'baseline' else 'projection']
                rows.append({'scene': scene, 'model': model, 'method': method, 'validated': True,
                    'images': run['reconstructed_images'], 'chunks': run['chunks'], 'points': merge['merged_points'],
                    'voxel_points': merge['voxel_points'], 'paired_points': paired['pairs'],
                    'paired_median_m': paired['median_m'], 'paired_inlier_ratio': paired['inlier_ratio'],
                    'worst_chunk_paired_median_m': max(c['baseline' if method == 'baseline' else 'projection']['median_m']
                                                       for c in comparison['per_chunk']),
                    'paired_rmse_all_m': paired['rmse_all_m'],
                    'peak_inference_gpu_gib': max(c.get('peak_gpu_gib', 0) for c in manifest['chunks']),
                    'output': str(result), 'prediction_dir': str(prediction)})
            require(shared[0] == shared[1], f'Methods used different predictions: {scene}/{model}')
    report = {'complete': True, 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'scenes': scenes, 'models': list(MODEL_NAMES), 'methods': ['baseline', 'correspondence'],
              'validated_runs': len(rows), 'validated_map_files': 3*len(rows),
              'registered_chunks': sum(row['chunks'] for row in rows), 'weights': weights,
              'metric_note': 'Paired errors compare both methods on the same held-out pixels within each model. '
                             'Confidence/depth filtering differs across models; this is not a controlled model ranking or ground-truth map benchmark.',
              'runs': rows}
    (report_dir/'matrix.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (report_dir/'matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Three-model full-sequence validation', '',
             f"{len(rows)} / {len(rows)} runs passed; {3*len(rows)} merged/voxel map files read back completely.", '',
             'Models: VGGT-1B, original Pi3, VGGT-Omega-1B-512. All checkpoint SHA-256 values and code revisions verified.', '',
             'Shared bag exports and selected original images; chunk size 40, step 25, sample ratio 0.05. '
             'Baseline uses the existing 5 Sim(3) / 1 ICP iterations and no optional straight-line refinement.', '',
             '| Scene | Model | Method | Images | Chunks | Points | Paired median (m) | Preview |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        link=f"../{row['scene']}/{row['model']}/{row['method']}/map_preview.png"
        lines.append(f"| {row['scene']} | {row['model']} | {row['method']} | {row['images']:,} | {row['chunks']} | "
                     f"{row['points']:,} | {row['paired_median_m']:.3f} | [map]({link}) |")
    lines += ['', report['metric_note'], '',
              'Checks: source PCD/TUM association, unchanged original image bytes, timestamps, selected-image coverage, '
              'all chunks, merged point counts, finite XYZ/RGB, shared input hashes and prediction manifests.', '',
              'Machine-readable details: [matrix.json](matrix.json), [matrix.csv](matrix.csv), [weights.json](weights.json).', '',
              'Each method directory contains its own `validation.json`, `bag_run.json`, `summary.json`, '
              '`merged_map.ply`, and `merged_map_0.1m.ply/.pcd`. Each model directory contains `methods_summary.json`.', '']
    (report_dir/'README.md').write_text('\n'.join(lines))
    for row in rows:
        write_run_report(row)
    print(f'{len(rows)} validated runs, {3*len(rows)} map files: {report_dir/"README.md"}')
    return report


def write_run_report(row):
    result = Path(row['output'])
    run = read(result/'bag_run.json')
    lines = [f"# {row['scene']} / {row['model']} / {row['method']}", '',
             'Full sequence completed and read back successfully.', '',
             f"- Original bag: `{run['request']['bag']['path']}`",
             f"- Shared dataset: `{run['dataset']}`",
             f"- Shared predictions: `{row['prediction_dir']}`",
             f"- FAST-LIO paired PCD / poses: {run['frontend_scans']:,}.",
             f"- Reconstructed images: {row['images']:,}; completed chunks: {row['chunks']}.",
             f"- Merged points: {row['points']:,}; 0.1 m voxel points: {row['voxel_points']:,}.",
             '- All body PCDs, TUM poses, original selected image bytes, timestamps and three map files: validated.',
             f"- Registration, merge and preview: {run['registration_merge_preview_seconds']:.1f} seconds "
             '(excludes shared input preparation/inference; concurrent runs affect wall-clock time).',
             f"- Paired median residual: {row['paired_median_m']:.3f} m on {row['paired_points']:,} held-out pairs; "
             f"inlier ratio at 2 m: {row['paired_inlier_ratio']:.2%}.",
             f"- Worst chunk paired median residual: {row['worst_chunk_paired_median_m']:.3f} m.", '',
             'Baseline uses pose initialization, regularized nearest-neighbor Sim(3), and ICP. '
             'Correspondence uses calibrated projection, RANSAC and Umeyama Sim(3). No loop closure/PGO was run.', '',
             'The comparison in `../correspondence/comparison.json` uses the same model predictions and pixel pairs. '
             'These residuals are registration diagnostics, not an independent map accuracy benchmark.', '',
             'See `bag_run.json`, `validation.json`, `summary.json`, `config.yaml`, and `map_preview.png`.', '']
    (result/'RUN_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/from_bag')
    parser.add_argument('--scenes', nargs='+', default=['AMtown02', 'HKisland03'])
    parser.add_argument('--verify-weights', action='store_true', help='Refresh the checkpoint/code verification snapshot')
    args = parser.parse_args()
    if args.verify_weights:
        verify_models(args.output_root.resolve()/'model_validation')
    summarize(args.output_root.resolve(), args.scenes)
