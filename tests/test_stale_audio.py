"""Bounded CPU regressions: audited functions only, no application/model imports."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(os.environ.get('ALEXANDRIA_TEST_ROOT', Path(__file__).resolve().parents[1]))


def load_functions(relative, names, namespace, class_name=None):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if len(selected) != len(names):
        raise AssertionError(f'Missing functions in {relative}: {names}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)
    return namespace

class StaleAudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / 'old.wav').write_bytes(b'existing audio')
        self.loads = []
        audio = SimpleNamespace(from_file=lambda path: self.loads.append(path) or path)
        self.ns = load_functions('app/project.py', ['update_chunk', '_load_chunks_with_audio'],
                                 dict(os=os, AudioSegment=audio), 'ProjectManager')
        self.chunks = [dict(text='Old text', speaker='A', instruct='Calm', status='done', audio_path='old.wav')]
        self.manager = SimpleNamespace(root_dir=self.tmp.name, load_chunks=lambda: self.chunks,
                                       save_chunks=lambda chunks: None)

    def test_each_synthesis_field_invalidates_and_old_file_is_not_loaded(self):
        for key, value in [('text', 'New text'), ('speaker', 'B'), ('instruct', 'Urgent')]:
            with self.subTest(key=key):
                self.setUpChunk()
                self.ns['update_chunk'](self.manager, 0, {key: value})
                self.assertEqual(self.chunks[0]['status'], 'pending')
                self.assertIsNone(self.chunks[0]['audio_path'])
                self.assertEqual(self.ns['_load_chunks_with_audio'](self.manager), [])
                self.assertTrue((Path(self.tmp.name) / 'old.wav').exists())

    def setUpChunk(self):
        self.chunks[:] = [dict(text='Old text', speaker='A', instruct='Calm', status='done', audio_path='old.wav')]

    def test_same_values_and_pause_edits_retain_current_audio(self):
        for update in ({'text': 'Old text', 'speaker': 'A', 'instruct': 'Calm'},
                       {'pause_after': 300}, {'pause_after': None}):
            self.ns['update_chunk'](self.manager, 0, update)
            self.assertEqual(self.chunks[0]['status'], 'done')
            self.assertEqual(len(self.ns['_load_chunks_with_audio'](self.manager)), 1)

    def test_legacy_references_require_done_status_and_regeneration_restores_audio(self):
        for status in ('pending', 'generating', 'error', None):
            self.chunks[0]['status'] = status
            self.assertEqual(self.ns['_load_chunks_with_audio'](self.manager), [])
        self.assertEqual(self.loads, [])
        self.chunks[0]['status'] = 'done'
        self.assertEqual(len(self.ns['_load_chunks_with_audio'](self.manager)), 1)


if __name__ == '__main__':
    unittest.main()
