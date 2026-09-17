"""Download pinned official source and verified weights into models/ (all by default)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Standard HTTPS resumes into the explicit local directory; no global weight cache.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from models.registry import MODEL_NAMES, MODEL_ROOT, checkpoint_path, model_spec
from models.downloads import download_modelscope, select_source


def verify_checkpoint(path, spec):
    if not path.is_file() or path.stat().st_size != spec['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == spec['sha256']


def download_model(name, sources_only=False, source='auto'):
    spec = model_spec(name)
    repo = MODEL_ROOT / 'repos' / name
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', spec['repository'], str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', spec['code_revision']], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != spec['code_revision']:
        raise ValueError(f'{repo} is at {revision}; expected {spec["code_revision"]}. Existing checkouts are not overwritten.')
    print(f'{name}: source {revision}', flush=True)
    if sources_only:
        return {'model': name, 'status': 'source_ready', 'revision': revision}
    path = checkpoint_path(name)
    provider = select_source(spec, source)
    info_path = path.parent / 'download_info.json'
    provenance = json.loads(info_path.read_text()) if info_path.exists() else {'source': 'existing_verified'}
    if path.exists():
        if not verify_checkpoint(path, spec):
            raise ValueError(f'Existing checkpoint failed size/SHA-256 verification: {path}; move it aside before retrying.')
        print(f'{name}: existing checkpoint SHA-256 OK', flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'{name}: downloading from {provider}', flush=True)
        if provider == 'modelscope':
            downloaded = download_modelscope(spec, path, verify_checkpoint)
            provenance = {'source': provider, 'repository': spec['modelscope_repo'],
                          'revision': spec['modelscope_revision']}
        else:
            for attempt in range(6):
                try:
                    downloaded = hf_hub_download(repo_id=spec['hf_repo'], filename=spec['filename'],
                        revision=spec['weight_revision'], local_dir=path.parent)
                    break
                except GatedRepoError:
                    raise
                except HfHubHTTPError as error:
                    if error.response is not None and error.response.status_code in (401, 403, 404):
                        raise
                    if attempt == 5:
                        raise
                    print(f'{name}: HTTP download interrupted; resume attempt {attempt + 2}/6', flush=True)
                    time.sleep(min(2**attempt, 16))
                except (OSError, ConnectionError):
                    if attempt == 5:
                        raise
                    print(f'{name}: download interrupted; resume attempt {attempt + 2}/6', flush=True)
                    time.sleep(min(2**attempt, 16))
            provenance = {'source': provider, 'repository': spec['hf_repo'], 'revision': spec['weight_revision']}
        if Path(downloaded).resolve() != path.resolve() or not verify_checkpoint(path, spec):
            raise ValueError(f'Downloaded checkpoint failed SHA-256 verification: {path}')
        print(f'{name}: downloaded; SHA-256 OK', flush=True)
    provenance.update(filename=spec['filename'], size=spec['size'], sha256=spec['sha256'])
    info_path.write_text(json.dumps(provenance, indent=2) + '\n')
    return {'model': name, 'status': 'ready', 'download': provenance, 'checkpoint': str(path), 'sha256': spec['sha256'],
            'code_revision': revision, 'weight_revision': spec['weight_revision']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['all', *MODEL_NAMES], default=['all'])
    parser.add_argument('--sources-only', action='store_true', help='Fetch code without model weights')
    parser.add_argument('--source', choices=['auto', 'huggingface', 'modelscope'], default='auto',
                        help='Default: ModelScope for Omega, Hugging Face for VGGT/Pi3')
    args = parser.parse_args()
    names = MODEL_NAMES if 'all' in args.models else list(dict.fromkeys(args.models))
    results = []
    for name in names:
        try:
            results.append(download_model(name, args.sources_only, args.source))
        except GatedRepoError:
            message = (f'Access to https://huggingface.co/{model_spec(name)["hf_repo"]} is required. '
                       'Request access with your Hugging Face account, then log in locally with hf auth login '
                       'and rerun this command. For Omega, --source modelscope selects the verified public mirror. '
                       'An authorized local checkpoint can also be passed with --checkpoint.')
            print(f'{name}: BLOCKED: {message}', file=sys.stderr, flush=True)
            results.append({'model': name, 'status': 'access_required', 'message': message})
        except Exception as error:
            # Never print headers or tokens. Other models can still be installed.
            print(f'{name}: FAILED: {type(error).__name__}: {str(error).splitlines()[0]}', file=sys.stderr, flush=True)
            results.append({'model': name, 'status': 'failed', 'error_type': type(error).__name__})
    weights = MODEL_ROOT / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    (weights / 'download_report.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)
    return int(any(r['status'] not in ('ready', 'source_ready') for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
