"""Shared model inference and separate baseline / correspondence reconstruction runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml
from models.inference import natural_key
from models.predictions import load_prediction_manifest
from models.registry import configured_checkpoint, model_spec
from tools.bag.reader import bag_identity

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('baseline', 'correspondence')
RUNNERS = {'baseline': 'run_pipeline.py', 'correspondence': 'run_projection_pipeline.py'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_logged(command, log_path):
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with log_path.open('w') as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}); see {log_path}')


def ensure_predictions(root, dataset, model, chunk_size, chunk_step, max_chunks, selection_hash):
    """Infer once for both methods; reuse compatible historical bag predictions."""
    if (root/'retention.json').exists():
        raise ValueError(f'Final maps are retained at {root}, but intermediate predictions were intentionally removed. '
                         'Use a new --output-root to reconstruct again.')
    config = yaml.safe_load((dataset/'dataset_config.yaml').read_text())
    checkpoint = Path(configured_checkpoint(model, config)).resolve()
    spec = model_spec(model)
    request = {'selection_sha256': selection_hash, 'model': model, 'checkpoint': str(checkpoint),
               'code_revision': spec['code_revision'], 'preprocess': spec['preprocess'],
               'chunk_size': chunk_size, 'chunk_step': chunk_step, 'max_chunks': max_chunks}
    plan_file = root/'prediction_request.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != request:
        raise ValueError('Prediction input/settings changed; select a new output root')
    images = sorted((p for p in (dataset/'images').iterdir()
                     if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), key=natural_key)
    prediction_dir = root/'predictions'
    legacy = root/'npy'
    if not prediction_dir.exists() and (legacy/'manifest.json').exists():
        old_run = json.loads((root/'bag_run.json').read_text())
        if old_run['request']['selection_sha256'] != selection_hash or old_run['request']['model'] != model:
            raise ValueError('Historical predictions belong to a different input/model')
        prediction_dir = legacy
    if (prediction_dir/'manifest.json').exists():
        manifest = load_prediction_manifest(prediction_dir, images, model, chunk_size=chunk_size,
            chunk_step=chunk_step, max_chunks=max_chunks, require_all=max_chunks is None)
        if (Path(manifest['checkpoint']).resolve() != checkpoint or manifest.get('code_revision') != spec['code_revision']
                or manifest.get('preprocess') != spec['preprocess']):
            raise ValueError('Saved predictions use different weights or model code/preprocessing')
        print(f'Reusing {len(manifest["chunks"])} {model} prediction chunks: {prediction_dir}', flush=True)
    else:
        if prediction_dir.exists() and any(prediction_dir.iterdir()):
            raise ValueError('Incomplete prediction cache; preserve it and use a new output root')
        plan_file.write_text(json.dumps(request, indent=2)+'\n')
        command = [sys.executable, str(ROOT/'tools/run_inference.py'), '--model', model,
                   '--config', str(dataset/'dataset_config.yaml'), '--image-dir', str(dataset/'images'),
                   '--output-dir', str(prediction_dir), '--chunk-size', str(chunk_size),
                   '--overlap', str(chunk_size-chunk_step)]
        if max_chunks is not None:
            command += ['--max-chunks', str(max_chunks)]
        print(f'Inferring shared {model} predictions; log: {root/"inference.log"}', flush=True)
        run_logged(command, root/'inference.log')
        manifest = load_prediction_manifest(prediction_dir, images, model, chunk_size=chunk_size,
            chunk_step=chunk_step, max_chunks=max_chunks, require_all=max_chunks is None)
    plan_file.write_text(json.dumps(request, indent=2)+'\n')
    return prediction_dir.resolve(), manifest


def method_command(method, dataset, output, predictions, model, sample_ratio, max_chunks, refine_line=False):
    if method not in METHODS:
        raise ValueError(f'Unknown registration method: {method}')
    config = dataset/('dataset_config.yaml' if method == 'baseline' else 'projection.yaml')
    command = [sys.executable, str(ROOT/'tools'/RUNNERS[method]), '--config', str(config),
               '--model', model, '--prediction-dir', str(predictions), '--output-dir', str(output),
               '--merge', '--sample-ratio', str(sample_ratio)]
    command += ['--max-chunks', str(max_chunks)] if max_chunks is not None else ['--all-images']
    if method == 'baseline' and refine_line:
        command += ['--refine-line']
    return command


def completed_run(output, request, expected_images, expected_chunks):
    path = output/'bag_run.json'
    if not path.exists():
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'Incomplete result at {output}; preserve it and use a new output root')
        return None
    report = json.loads(path.read_text())
    if report.get('request') != request:
        raise ValueError(f'Reconstruction settings changed at {output}; select a new output root')
    if not report.get('complete'):
        raise ValueError(f'Incomplete result at {output}')
    merge = json.loads((output/'merge_summary.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    paths = [Path(merge[key]) for key in ('merged_file', 'voxel_ply', 'voxel_pcd') if merge.get(key)]
    paths += [output/'vggt_chunk_reg/final'/f'{i}_pc.ply' for i in range(expected_chunks)]
    if (merge['unique_images'] != expected_images or merge['chunks'] != expected_chunks
            or len(summary['registration']) != expected_chunks
            or not all(p.is_file() and p.stat().st_size > 0 for p in paths)):
        raise ValueError(f'Completed result has missing maps/chunks: {output}')
    return report


def reconstruct(bag, dataset, output_root, selected, frontend, *, model, method,
                chunk_size, chunk_step, max_chunks, sample_ratio, refine_line=False):
    """Both methods consume exactly the same immutable model predictions."""
    started = time.perf_counter()
    tag = model + (f'_first{max_chunks}' if max_chunks is not None else '')
    root = output_root/bag.stem/tag
    root.mkdir(parents=True, exist_ok=True)
    selection_hash = sha256(dataset/'selection.json')
    prediction_dir, manifest = ensure_predictions(root, dataset, model, chunk_size, chunk_step, max_chunks, selection_hash)
    methods = METHODS if method == 'both' else (method,)
    expected_images = len({image for entry in manifest['chunks'] for image in entry['images']})
    if max_chunks is None and expected_images != selected['selected_images']:
        raise ValueError('Shared predictions do not cover every selected image')
    common = {'schema_version': 2, 'bag': bag_identity(bag), 'selection_sha256': selection_hash,
              'frontend_pose_sha256': sha256(dataset/'frontend/pose.txt'), 'model': model,
              'prediction_manifest_sha256': sha256(prediction_dir/'manifest.json'),
              'chunk_size': chunk_size, 'chunk_step': chunk_step, 'max_chunks': max_chunks,
              'sample_ratio': sample_ratio, 'base_config_sha256': sha256(dataset/'dataset_config.yaml')}
    reports = {}
    for name in methods:
        output = root/name
        request = dict(common, method=name)
        if name == 'baseline':
            request['refine_line'] = refine_line
        else:
            request['projection_config_sha256'] = sha256(dataset/'projection.yaml')
        previous = completed_run(output, request, expected_images, len(manifest['chunks']))
        if previous is not None:
            print(f'[{bag.stem}] Reusing completed {name}: {output}', flush=True)
            reports[name] = previous
            continue
        command = method_command(name, dataset, output, prediction_dir, model, sample_ratio, max_chunks, refine_line)
        log_path = root/(name+'.log')
        print(f'[{bag.stem}] {name} registration; log: {log_path}', flush=True)
        tick = time.perf_counter()
        run_logged(command, log_path)
        merge = json.loads((output/'merge_summary.json').read_text())
        if merge['unique_images'] != expected_images or merge['chunks'] != len(manifest['chunks']):
            raise ValueError('Merged map is missing selected images or chunks')
        subprocess.run([sys.executable, str(ROOT/'tools/preview_map.py'), str(output)], check=True)
        report = {'request': request, 'complete': True, 'full_sequence': max_chunks is None,
                  'method': name, 'dataset': str(dataset), 'output': str(output), 'prediction_dir': str(prediction_dir),
                  'frontend_scans': frontend['keyframes'], 'selected_images': selected['selected_images'],
                  'reconstructed_images': expected_images, 'chunks': merge['chunks'], 'merged_points': merge['merged_points'],
                  'registration_merge_preview_seconds': time.perf_counter()-tick}
        (output/'bag_run.json').write_text(json.dumps(report, indent=2)+'\n')
        reports[name] = report
        print(f'[{bag.stem}] {name}: {merge["chunks"]} chunks, {merge["merged_points"]:,} points', flush=True)
    if method == 'both':
        run_logged([sys.executable, str(ROOT/'tools/compare_projection.py'), str(root/'correspondence'),
                    '--baseline-dir', str(root/'baseline')], root/'comparison.log')
        comparison = json.loads((root/'correspondence/comparison.json').read_text())
        (root/'methods_summary.json').write_text(json.dumps({
            'complete': True, 'model': model, 'dataset': str(dataset), 'prediction_dir': str(prediction_dir),
            'methods': {name: {'output': item['output'], 'chunks': item['chunks'], 'points': item['merged_points']}
                        for name, item in reports.items()},
            'paired_comparison': {key: comparison[key] for key in ('metric', 'note', 'baseline', 'projection')},
            'seconds_this_invocation': time.perf_counter()-started}, indent=2)+'\n')
        print(f'[{bag.stem}] Both methods completed; comparison: {root/"methods_summary.json"}', flush=True)
    return reports
