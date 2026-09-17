"""Verified public ModelScope downloads with resumable, atomic checkpoint writes."""
import re
import time
from urllib.parse import quote, urlencode

import requests


def select_source(spec, requested='auto'):
    source = spec.get('preferred_source', 'huggingface') if requested == 'auto' else requested
    if source not in ('huggingface', 'modelscope'):
        raise ValueError(f'Unknown download source: {source}')
    if source == 'modelscope' and not (spec.get('modelscope_repo') and spec.get('modelscope_revision')):
        raise ValueError('No verified ModelScope mirror is configured for this model; use --source auto or huggingface')
    return source


def modelscope_url(spec):
    return ('https://modelscope.cn/api/v1/models/' + quote(spec['modelscope_repo'], safe='/') + '/repo?'
            + urlencode({'Revision': spec['modelscope_revision'], 'FilePath': spec['filename']}))


def download_modelscope(spec, destination, verify, attempts=6):
    """Fetch only the pinned file; verify SHA-256 before exposing the final path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Destination already exists: {destination}')
    part = destination.with_name(destination.name + '.part')
    expected_size = spec['size']
    url = modelscope_url(spec)
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            raise ValueError(f'Partial file exceeds expected size: {part}; move it aside before retrying')
        if offset == expected_size:
            break
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                        raise ValueError('ModelScope returned an inconsistent Content-Range')
                    mode = 'ab' if offset else 'wb'
                elif response.status_code == 200:
                    # A server may ignore Range: restart, never append a full response.
                    offset, mode = 0, 'wb'
                else:
                    raise ValueError(f'Unexpected download status: {response.status_code}')
                last_print = time.monotonic()
                received = offset
                with part.open(mode) as stream:
                    for block in response.iter_content(chunk_size=1024*1024):
                        if not block:
                            continue
                        received += len(block)
                        if received > expected_size:
                            raise ValueError('Download exceeds pinned checkpoint size')
                        stream.write(block)
                        if time.monotonic()-last_print >= 10:
                            print(f'ModelScope: {received/1024**3:.2f}/{expected_size/1024**3:.2f} GiB ({received/expected_size:.1%})', flush=True)
                            last_print = time.monotonic()
                if received != expected_size:
                    raise OSError(f'Incomplete download: {received}/{expected_size} bytes')
            break
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code in (401, 403, 404):
                raise
            if attempt == attempts-1:
                raise
            print(f'ModelScope: HTTP interruption, retry {attempt+2}/{attempts}', flush=True)
            time.sleep(min(2**attempt, 16))
        except (requests.RequestException, OSError):
            if attempt == attempts-1:
                raise
            print(f'ModelScope: interrupted, resuming attempt {attempt+2}/{attempts}', flush=True)
            time.sleep(min(2**attempt, 16))
    if not verify(part, spec):
        raise ValueError(f'ModelScope checkpoint failed SHA-256 verification: {part}; final file was not installed')
    part.replace(destination)
    return destination
