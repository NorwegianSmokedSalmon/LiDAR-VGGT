"""Pinned official model sources and checkpoint metadata."""
import json
from pathlib import Path
import subprocess
import sys

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
REGISTRY = json.loads((MODEL_ROOT / 'registry.json').read_text())
MODEL_NAMES = tuple(REGISTRY)


def model_spec(name):
    if name not in REGISTRY:
        raise ValueError(f'Unknown model {name!r}; choose one of {MODEL_NAMES}')
    return dict(REGISTRY[name])


def checkpoint_path(name):
    return MODEL_ROOT / 'weights' / name / model_spec(name)['filename']


def use_repository(name):
    spec = model_spec(name)
    path = MODEL_ROOT / 'repos' / name
    if not path.is_dir():
        raise FileNotFoundError(f'Missing {name} source. Run: python tools/download_models.py --models {name}')
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != spec['code_revision']:
        raise ValueError(f'{name} source revision differs from models/registry.json: {revision}')
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def configured_checkpoint(name, config, override=None):
    """CLI override, then existing Weights config, then the model registry path."""
    if override is not None:
        return Path(override)
    key = {'vggt': 'VGGT', 'pi3': 'Pi3', 'vggt_omega': 'VGGT_Omega'}[name]
    value = config.get('Weights', {}).get(key)
    if value:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return checkpoint_path(name)
