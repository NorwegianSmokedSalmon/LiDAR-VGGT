"""Transport failure checks for large checkpoint downloads; no network required."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.downloads import download_modelscope, select_source
from tools.download_models import verify_checkpoint


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body


class ModelDownloadsTest(unittest.TestCase):
    def setUp(self):
        self.payload = b'checkpoint-payload'
        self.spec = {'size': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest(),
                     'filename': 'model.pt', 'modelscope_repo': 'test/model', 'modelscope_revision': 'pinned'}
        self.work = tempfile.TemporaryDirectory()
        self.path = Path(self.work.name)/'model.pt'
        self.part = self.path.with_name('model.pt.part')
        self.addCleanup(self.work.cleanup)

    def test_partial_range_resumes_and_installs_only_verified_file(self):
        self.part.write_bytes(self.payload[:5])
        response = Response(206, self.payload[5:], {'Content-Range': f'bytes 5-{len(self.payload)-1}/{len(self.payload)}'})
        with patch('models.downloads.requests.get', return_value=response) as get:
            download_modelscope(self.spec, self.path, verify_checkpoint)
        self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=5-')
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertFalse(self.part.exists())

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        self.part.write_bytes(b'old-partial')
        with patch('models.downloads.requests.get', return_value=Response(200, self.payload)):
            download_modelscope(self.spec, self.path, verify_checkpoint)
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_wrong_hash_is_never_exposed_as_final_checkpoint(self):
        bad = b'x'*len(self.payload)
        with patch('models.downloads.requests.get', return_value=Response(200, bad)):
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                download_modelscope(self.spec, self.path, verify_checkpoint)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.part.read_bytes(), bad)

    def test_inconsistent_resume_response_is_rejected(self):
        self.part.write_bytes(self.payload[:5])
        response = Response(206, self.payload, {'Content-Range': f'bytes 0-{len(self.payload)-1}/{len(self.payload)}'})
        with patch('models.downloads.requests.get', return_value=response):
            with self.assertRaisesRegex(ValueError, 'Content-Range'):
                download_modelscope(self.spec, self.path, verify_checkpoint)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.part.read_bytes(), self.payload[:5])

    def test_existing_valid_partial_finishes_without_network(self):
        self.part.write_bytes(self.payload)
        with patch('models.downloads.requests.get') as get:
            download_modelscope(self.spec, self.path, verify_checkpoint)
            get.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_default_source_and_explicit_override(self):
        self.assertEqual(select_source({}), 'huggingface')
        spec = dict(self.spec, preferred_source='modelscope')
        self.assertEqual(select_source(spec), 'modelscope')
        self.assertEqual(select_source(spec, 'huggingface'), 'huggingface')
        with self.assertRaisesRegex(ValueError, 'No verified ModelScope'):
            select_source({}, 'modelscope')


if __name__ == '__main__':
    unittest.main()
