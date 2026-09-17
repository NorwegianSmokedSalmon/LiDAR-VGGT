"""Shared chunk inference and canonical prediction serialization for all backends."""
from contextlib import nullcontext
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from models.backends import ReconstructionModel
from models.geometry import validate_predictions
from models.preprocessing import prepare_images
from models.registry import model_spec


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', Path(path).name)]


def chunk_ranges(count, size, overlap):
    if size < 1 or not 0 <= overlap < size:
        raise ValueError('Require chunk_size > 0 and 0 <= overlap < chunk_size')
    start = 0
    while start < count:
        end = min(start + size, count)
        yield start, end
        if end == count:
            break
        start += size-overlap


def saved_model(manifest, requested=None):
    """Older manifests predate model tagging and contain original VGGT predictions."""
    name = manifest.get('model', 'vggt')
    model_spec(name)
    if requested is not None and name != requested:
        raise ValueError(f'Saved predictions use {name}, but --model requested {requested}')
    return name


def run_inference(image_paths, output_dir, checkpoint=None, chunk_size=40, overlap=15,
                  max_chunks=None, device='auto', model_name='vggt', chunk_writer=None):
    if not image_paths:
        raise ValueError('No input images found')
    spec = model_spec(model_name)
    ranges = list(chunk_ranges(len(image_paths), chunk_size, overlap))
    if max_chunks is not None:
        if max_chunks < 1:
            raise ValueError('max_chunks must be positive')
        ranges = ranges[:max_chunks]
    output_dir = Path(output_dir)
    if any(output_dir.glob('chunk_*.npy')) or (output_dir/'manifest.json').exists():
        raise FileExistsError(f'Prediction directory already contains results: {output_dir}')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    print(f'Loading {model_name} on {device}', flush=True)
    model = ReconstructionModel(model_name, device, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 2, 'model': model_name, 'checkpoint': str(model.checkpoint),
                'code_revision': spec['code_revision'], 'preprocess': spec['preprocess'],
                'confidence_type': spec['confidence'], 'default_min_confidence': spec['default_min_confidence'],
                'intrinsic_source': 'fitted_from_local_rays' if model_name == 'pi3' else 'camera_head',
                'coordinate_convention': 'world_points=chunk_world; extrinsic=OpenCV_camera_from_world; depth=camera_z',
                'device': str(device), 'chunk_size': chunk_size, 'overlap': overlap, 'chunks': []}
    for idx, (start, end) in enumerate(ranges):
        paths = image_paths[start:end]
        tick = time.perf_counter()
        print(f'{model_name} chunk {idx+1}/{len(ranges)}: {Path(paths[0]).name} .. {Path(paths[-1]).name}', flush=True)
        prepared = prepare_images(paths, model_name)
        images = prepared.images.to(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        amp = torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
        with torch.inference_mode(), amp:
            arrays = model.predict(images)
        arrays['image_pixel_transform'] = prepared.pixel_transform
        arrays['image_original_size'] = prepared.original_size
        arrays['image_valid_bounds'] = prepared.valid_bounds
        # Padding is never treated as an observed image pixel in downstream matching.
        height, width = images.shape[-2:]
        y, x = np.mgrid[:height, :width]
        for frame, (left, top, right, bottom) in enumerate(prepared.valid_bounds):
            invalid = (x < left) | (x > right) | (y < top) | (y > bottom)
            arrays['world_points_conf'][frame][invalid] = 0
            arrays['depth_conf'][frame][invalid] = 0
        validate_predictions(arrays, len(paths))
        if chunk_writer is None:
            filename = f'chunk_{idx}.npy'
            temporary = output_dir/(filename+'.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, arrays)
            temporary.replace(output_dir/filename)
        else:
            # A streaming consumer can retain compact downstream observations
            # without writing multi-gigabyte dense prediction caches.
            filename = chunk_writer(idx, arrays)
        entry = {'index': idx, 'file': filename, 'start': start, 'end': end,
                 'images': [str(Path(p).resolve()) for p in paths],
                 'seconds': time.perf_counter()-tick,
                 'shapes': {k: list(v.shape) for k, v in arrays.items()}}
        if device.type == 'cuda':
            entry['peak_gpu_gib'] = torch.cuda.max_memory_allocated(device)/1024**3
        manifest['chunks'].append(entry)
        staged = output_dir/'manifest.json.tmp'
        staged.write_text(json.dumps(manifest, indent=2)+'\n')
        staged.replace(output_dir/'manifest.json')
        print(f'Saved {output_dir/filename} ({entry["seconds"]:.1f}s)', flush=True)
        del arrays, images, prepared
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return manifest
