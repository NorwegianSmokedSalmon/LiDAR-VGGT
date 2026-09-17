"""Compact per-frame PGO inputs; regenerate pruned predictions without saving them."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from models.geometry import validate_predictions
from models.inference import run_inference
from models.registry import model_spec
from Registration.projection_sim3 import sample_pointmap

SCHEMA = 1


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_result(result_dir):
    root = Path(result_dir).resolve()
    summary = json.loads((root/'summary.json').read_text())
    manifest = summary['inference']
    entries = manifest['chunks']
    if len(entries) < 2 or [e['index'] for e in entries] != list(range(len(entries))):
        raise ValueError('PGO requires at least two complete, consecutively indexed chunks')
    if len(summary['registration']) != len(entries):
        raise ValueError('Registration did not finish every chunk')
    transforms = np.loadtxt(root/'sim3_total.txt').reshape(-1, 4, 4)
    if len(transforms) != len(entries) or not np.isfinite(transforms).all():
        raise ValueError('Incomplete/non-finite metric alignment transforms')
    order = [int(Path(r['vggt_file']).stem.split('_')[0]) for r in summary['registration']]
    if order != list(range(len(entries))):
        raise ValueError('Registration transform order does not match chunk order')
    # Overlaps must identify the same original image, not merely frame indices.
    image_at_index = {}
    for entry in entries:
        if entry['end'] - entry['start'] != len(entry['images']):
            raise ValueError('Invalid image/chunk bounds')
        for index, name in enumerate(entry['images'], entry['start']):
            path = str(Path(name).resolve(strict=True))
            if index in image_at_index and image_at_index[index] != path:
                raise ValueError('Overlapping chunks refer to different images')
            image_at_index[index] = path
    if sorted(image_at_index) != list(range(len(image_at_index))):
        raise ValueError('Chunk image coverage contains gaps')
    return root, summary, manifest, transforms


def compact_chunk(arrays, options, min_confidence, seed):
    n, height, width, _ = arrays['world_points'].shape
    validate_predictions(arrays, n)
    ext = arrays['extrinsic'].astype(float).copy()
    ext[:, :3, :3] = Rotation.from_matrix(ext[:, :3, :3]).as_matrix()
    side = int(options['grid_side'])
    yy, xx = np.mgrid[:side, :side]
    normalized = np.column_stack([(xx.ravel()+.5)/side, (yy.ravel()+.5)/side])
    points = np.zeros((n, side*side, 3), np.float32)
    valid = np.zeros((n, side*side), bool)
    sampled, colors, offsets = [], [], [0]
    rng = np.random.default_rng(seed)
    for frame in range(n):
        conf = arrays['world_points_conf'][frame]
        positive = conf[np.isfinite(conf) & (conf > 0)]
        if not len(positive):
            raise ValueError('Frame has no valid confidence')
        cutoff = max(min_confidence, float(np.percentile(positive, options['confidence_percentile'])))
        original_uv = normalized * arrays['image_original_size'][frame] - .5
        affine = arrays['image_pixel_transform'][frame]
        uv = original_uv @ affine[:2, :2].T + affine[:2, 2]
        xyz, mask = sample_pointmap(arrays['world_points'][frame], conf, arrays['depth'][frame], uv,
                                    cutoff, options['max_relative_depth_jump'])
        points[frame, mask] = xyz @ ext[frame, :3, :3].T + ext[frame, :3, 3]
        valid[frame] = mask
        # Retain uniformly sampled local pointmaps for recoloring/re-export at
        # any optimized pose. They are independent of the overlap fitting mask.
        xyz = arrays['world_points'][frame].reshape(-1, 3)
        flat_conf = conf.ravel()
        good = np.flatnonzero(np.isfinite(xyz).all(1) & np.isfinite(flat_conf)
                              & (flat_conf >= min_confidence) & (flat_conf > 0))
        selected = rng.choice(good, int(len(good)*options['export_sample_ratio']), replace=False)
        if not len(selected):
            raise ValueError('Empty frame export sample')
        sampled.append((xyz[selected] @ ext[frame, :3, :3].T + ext[frame, :3, 3]).astype(np.float32))
        rgb = arrays['images'][frame].transpose(1, 2, 0).reshape(-1, 3)
        colors.append(np.rint(np.clip(rgb[selected], 0, 1)*255).astype(np.uint8))
        offsets.append(offsets[-1] + len(selected))
    return {'extrinsic': ext, 'overlap_points': points, 'overlap_valid': valid,
            'sample_points': np.concatenate(sampled), 'sample_colors': np.concatenate(colors),
            'sample_offsets': np.array(offsets, np.int64)}


def ensure_inputs(result_dir, cache_dir, config, regenerate=False, device='auto'):
    root, summary, manifest, transforms = load_result(result_dir)
    model = summary['model']
    spec = model_spec(model)
    for key in ('code_revision', 'preprocess'):
        if manifest.get(key) != spec[key]:
            raise ValueError(f'Result uses another {key}; cannot regenerate compatible predictions')
    images = sorted({p for e in manifest['chunks'] for p in e['images']},
                    key=lambda p: next(e['start']+e['images'].index(p) for e in manifest['chunks'] if p in e['images']))
    provenance = {'schema': SCHEMA, 'model': model, 'code_revision': spec['code_revision'],
                  'preprocess': spec['preprocess'], 'checkpoint_sha256': digest(manifest['checkpoint']),
                  'images': [{'path': p, 'sha256': digest(p)} for p in images],
                  'chunks': [{k: e[k] for k in ('index', 'start', 'end', 'images')} for e in manifest['chunks']],
                  'observations': config['observations']}
    # A known pinned checkpoint is required when restoring a deleted cache.
    if regenerate and provenance['checkpoint_sha256'] != spec['sha256']:
        raise ValueError('Checkpoint checksum differs from the pinned model')
    cache = Path(cache_dir).resolve()
    request_path = cache/'inputs.json'
    if request_path.exists():
        previous = json.loads(request_path.read_text())
        if previous.get('request') != provenance or not previous.get('complete'):
            raise ValueError('PGO input cache is incomplete or incompatible; use another --cache-dir')
        for item in previous['files']:
            if digest(cache/item['file']) != item['sha256']:
                raise ValueError(f'PGO input checksum mismatch: {item["file"]}')
        print(f'Reusing {len(previous["files"])} compact {model} input chunks: {cache}', flush=True)
        return cache, summary, transforms
    if cache.exists() and any(cache.iterdir()):
        raise ValueError('PGO input cache is nonempty but incomplete; use another --cache-dir')
    source = Path(summary['prediction_dir'])
    available = all((source/e['file']).is_file() for e in manifest['chunks'])
    if not available and not regenerate:
        raise ValueError('Dense predictions were removed. Use --regenerate-predictions to build compact PGO inputs.')
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = {'complete': False, 'request': provenance, 'files': [],
                'source': 'saved_predictions' if available else 'regenerated_predictions'}
    request_path.write_text(json.dumps(snapshot, indent=2)+'\n')

    def write_chunk(index, arrays):
        entry = manifest['chunks'][index]
        if len(arrays['extrinsic']) != len(entry['images']):
            raise ValueError('Prediction frame count mismatch')
        compact = compact_chunk(arrays, config['observations'], spec['default_min_confidence'],
                                int(config['observations']['seed'])+index)
        path = cache/f'chunk_{index}.npz'
        np.savez_compressed(path, **compact)
        snapshot['files'].append({'file': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size})
        print(f'Compact PGO chunk {index+1}/{len(manifest["chunks"])}: {path.stat().st_size/1024**2:.1f} MiB', flush=True)
        return path.name

    if available:
        for entry in manifest['chunks']:
            write_chunk(entry['index'], np.load(source/entry['file'], allow_pickle=True).item())
    else:
        regenerated = run_inference(images, cache, manifest['checkpoint'], manifest['chunk_size'],
                                    manifest['overlap'], len(manifest['chunks']), device, model,
                                    chunk_writer=write_chunk)
        actual = [{k: e[k] for k in ('index', 'start', 'end', 'images')} for e in regenerated['chunks']]
        if actual != provenance['chunks']:
            raise ValueError('Regenerated prediction layout differs from source registration')
    snapshot['complete'] = True
    request_path.write_text(json.dumps(snapshot, indent=2)+'\n')
    return cache, summary, transforms
