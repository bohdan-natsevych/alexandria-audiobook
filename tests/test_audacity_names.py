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
import wave
import zipfile


class Segment:
    """Tiny 1 kHz PCM segment using stdlib wave; no ffmpeg or audio dependencies."""
    def __init__(self, samples=b''):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __add__(self, other):
        return Segment(self.samples + other.samples)
    @staticmethod
    def empty():
        return Segment()
    @staticmethod
    def silent(duration):
        return Segment(b'\x80' * duration)
    def export(self, buffer, format):
        with wave.open(buffer, 'wb') as wav:
            wav.setparams((1, 1, 1000, 0, 'NONE', 'not compressed'))
            wav.writeframes(self.samples)


class AudacityNamesTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(os=os, re=re, io=io, zipfile=zipfile, AudioSegment=Segment,
                       DEFAULT_PAUSE_MS=0, SAME_SPEAKER_PAUSE_MS=0)
        load_functions('app/tts.py', ['sanitize_filename', 'audacity_track_filenames', 'compute_timeline'], self.ns)
        load_functions('app/project.py', ['export_audacity'], self.ns, 'ProjectManager')

    def test_portable_unique_bounded_and_deterministic(self):
        speakers = ['A B', 'A_B', 'a_b', 'CON', 'nul', '', '中文', 'Привет', 'a' * 400,
                    '..', 'x/y', 'x\\y', '"quoted"', 'é', 'e\u0301']
        names = self.ns['audacity_track_filenames'](speakers)
        self.assertEqual(names, self.ns['audacity_track_filenames'](speakers[::-1]))
        self.assertEqual(len(set(n.casefold() for n in names.values())), len(speakers))
        for name in names.values():
            self.assertRegex(name, r'^track_\d+_[a-z0-9_-]+\.wav$')
            self.assertLess(len(name), 100)

    def test_zip_and_lof_refer_to_distinct_correct_wav_tracks(self):
        pairs = [(dict(speaker='A B', text='First'), Segment(b'\x81' * 5)),
                 (dict(speaker='A_B', text='Second'), Segment(b'\x82' * 5)),
                 (dict(speaker='a_b', text='Third'), Segment(b'\x83' * 5))]
        with tempfile.TemporaryDirectory() as tmp:
            manager = SimpleNamespace(root_dir=tmp, _load_chunks_with_audio=lambda: pairs,
                                      _load_pause_defaults=lambda: (0, 0))
            success, path = self.ns['export_audacity'](manager)
            self.assertTrue(success)
            names = self.ns['audacity_track_filenames']([c['speaker'] for c, _ in pairs])
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                self.assertEqual(len(members), len(set(members)))
                refs = [line[6:-1] for line in archive.read('project.lof').decode().splitlines()]
                self.assertEqual(refs, [names[c['speaker']] for c, _ in pairs])
                for index, (chunk, segment) in enumerate(pairs):
                    with wave.open(io.BytesIO(archive.read(names[chunk['speaker']])), 'rb') as wav:
                        samples = wav.readframes(wav.getnframes())
                    self.assertEqual(samples[index * 5:(index + 1) * 5], segment.samples)
                    self.assertEqual(len(samples), 15)
                    self.assertEqual(set(samples[:index * 5] + samples[(index + 1) * 5:]), {128})


if __name__ == '__main__':
    unittest.main()
