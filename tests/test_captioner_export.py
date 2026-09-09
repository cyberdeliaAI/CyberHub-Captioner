"""Run with python3 -m unittest discover -s tests; no CyberHub install needed."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import zipfile

module_path = Path(os.environ.get('CAPTIONER_SOURCE', Path(__file__).parents[1] / 'modules/captioner/__init__.py'))
core = types.ModuleType('core')
core.Module = object
server = types.ModuleType('core.server')
server.build_shell = None
spec = importlib.util.spec_from_file_location('captioner_test_module', module_path)
captioner = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'core': core, 'core.server': server}):
    spec.loader.exec_module(captioner)


class Handler:
    def __init__(self, items):
        self.data = {'items': items}
        self.status = 200

    def read_body_json(self, _length):
        return self.data

    def respond_json(self, data, status=200):
        self.response = data
        self.status = status

    def respond_binary(self, data, content_type, download_name):
        self.response = data
        self.content_type = content_type
        self.filename = download_name


class CaptionExportTests(unittest.TestCase):
    def export(self, items):
        handler = Handler(items)
        captioner.CaptionerModule()._sidecars(handler, 0, 'application/json')
        return handler

    def test_zip_contains_latest_caption_and_preserves_unicode(self):
        handler = self.export([{'name': 'portrait.png', 'caption': '  Hand-edited caption: café, soft light.  '}])
        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.filename, 'caption-files.zip')
        with zipfile.ZipFile(io.BytesIO(handler.response)) as archive:
            self.assertEqual(archive.namelist(), ['portrait.txt'])
            self.assertEqual(archive.read('portrait.txt').decode(), 'Hand-edited caption: café, soft light.\n')

    def test_zip_keeps_captions_separate_for_duplicate_filenames(self):
        handler = self.export([{'name': '../portrait.png', 'caption': 'First edit'}, {'name': 'portrait.jpg', 'caption': 'Second edit'}])
        with zipfile.ZipFile(io.BytesIO(handler.response)) as archive:
            self.assertEqual(len(archive.namelist()), 2)
            self.assertEqual({archive.read(name).decode() for name in archive.namelist()}, {'First edit\n', 'Second edit\n'})
            self.assertTrue(all('/' not in name for name in archive.namelist()))

    def test_empty_captions_are_not_exported(self):
        handler = self.export([{'name': 'empty.png', 'caption': '  '}])
        self.assertEqual(handler.status, 400)


if __name__ == '__main__':
    unittest.main()
