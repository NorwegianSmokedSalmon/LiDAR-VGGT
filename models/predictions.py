"""Validate cached model predictions before either registration pipeline uses them."""
import json
from pathlib import Path

from models.inference import chunk_ranges, saved_model


def load_prediction_manifest(directory, images, requested_model=None, *, chunk_size=None,
                             chunk_step=None, max_chunks=None, require_all=False):
    """Return a verified contiguous prefix, with exact image order and chunk bounds.

    Cached chunk layout is authoritative unless the caller supplies an expected
    layout. A full run must cover every image; a limited run must have every
    requested chunk. This never treats interrupted inference as a complete run.
    """
    directory = Path(directory).resolve()
    if (directory/'inputs.json').exists():
        raise ValueError('This is a compact PGO observation cache, not canonical predictions; use tools/run_pgo.py')
    manifest = json.loads((directory/'manifest.json').read_text())
    saved_model(manifest, requested_model)
    size, overlap = manifest['chunk_size'], manifest['overlap']
    if chunk_size is not None and chunk_size != size:
        raise ValueError('Saved prediction chunk size differs from the requested layout')
    if chunk_step is not None and chunk_step != size-overlap:
        raise ValueError('Saved prediction chunk step differs from the requested layout')
    expected_images = [str(Path(p).resolve()) for p in images]
    ranges = list(chunk_ranges(len(expected_images), size, overlap))
    entries = manifest['chunks']
    if not entries or len(entries) > len(ranges):
        raise ValueError('Saved inference has an invalid number of chunks')
    if require_all and len(entries) != len(ranges):
        raise ValueError('Saved inference does not cover all configured images')
    if max_chunks is not None:
        if max_chunks < 1:
            raise ValueError('max_chunks must be positive')
        needed = min(max_chunks, len(ranges))
        if len(entries) < needed:
            raise ValueError('Saved inference is missing requested chunks')
        entries = entries[:needed]
    for index, entry in enumerate(entries):
        start, end = ranges[index]
        if entry['index'] != index or (entry['start'], entry['end']) != (start, end):
            raise ValueError('Saved inference has inconsistent chunk indices or bounds')
        if entry['images'] != expected_images[start:end]:
            raise ValueError('Saved inference images do not match the configured dataset/order')
        prediction = directory/entry['file']
        if not prediction.is_file() or prediction.stat().st_size == 0:
            raise ValueError(f'Missing or empty prediction file: {prediction}')
    return dict(manifest, chunks=entries)


def prediction_directory(result_dir):
    """Support historical local npy/ and new shared prediction directories."""
    result_dir = Path(result_dir).resolve()
    summary_path = result_dir/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return Path(summary.get('prediction_dir', result_dir/'npy')).resolve()
