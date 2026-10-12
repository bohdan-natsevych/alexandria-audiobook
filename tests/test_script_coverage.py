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
from difflib import SequenceMatcher


def entry(text):
    return dict(speaker='NARRATOR', text=text, instruct='Neutral narration.')


class ScriptCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        app = Path(self.tmp.name) / 'app'
        app.mkdir()
        self.ns = dict(json=json, re=re, os=os, sys=sys, argparse=__import__('argparse'),
                       SequenceMatcher=SequenceMatcher, __file__=str(app / 'generate_script.py'),
                       DEFAULT_SYSTEM_PROMPT='Return JSON', DEFAULT_USER_PROMPT='{context}\n{chunk}')
        load_functions('app/generate_script.py',
                       ['clean_json_string', 'repair_json_array', 'salvage_json_entries',
                        '_coverage_text', 'source_coverage_passes', 'process_chunk'], self.ns)

    def test_complete_string_content_is_not_rewritten(self):
        text = 'Literal }, { brackets [ ] <think>quoted</think> "yes" and \\path\nline'
        raw = json.dumps([entry(text)])
        cleaned = self.ns['clean_json_string']('<think>planning</think>\n```json\n' + raw + '\n```')
        self.assertEqual(self.ns['repair_json_array'](cleaned), [entry(text)])

    def test_partial_and_malformed_arrays_never_salvage(self):
        raw = json.dumps([entry('First.'), entry('Last.')])
        for bad in (raw[:-1], raw[:-12], raw.replace('}, {', '} {'),
                    raw[:-1] + ', "dropped"]', '[{"speaker":"N","text":17,"instruct":""}]'):
            self.assertIsNone(self.ns['repair_json_array'](self.ns['clean_json_string'](bad)))
            self.assertIsNone(self.ns['salvage_json_entries'](bad))

    def test_coverage_rejects_missing_middle_end_and_equal_length_replacement(self):
        source = ' '.join(f'The lantern lit room number {i}.' for i in range(25)) + ' Finally she vanished.'
        for text in (source.replace(' Finally she vanished.', ''),
                     source.replace('The lantern lit room number 12. ', ''),
                     source.replace('Finally she vanished.', 'Another tale happened.')):
            self.assertFalse(self.ns['source_coverage_passes'](source, [entry(text)]))
        self.assertTrue(self.ns['source_coverage_passes'](source, [entry(source)]))

    def test_default_prompt_formatting_and_attribution_policy(self):
        cases = [('Chapter I\nDr. Smith & Mr. Jones came 3rd.', 'Chapter One\nDoctor Smith and Mister Jones came third.'),
                 ('"Go," he said.', 'Go.'),
                 ('"There is nothing to tell," said Marcus, unable to meet her gaze.',
                  'There is nothing to tell. Marcus was unable to meet her gaze.'),
                 ('Привет, мир. Она вернулась домой.', 'Привет, мир. Она вернулась домой.')]
        for source, spoken in cases:
            self.assertTrue(self.ns['source_coverage_passes'](source, [entry(spoken)]), source)

    def test_length_response_retried_even_with_complete_json(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps([entry('Full source.')])),
                                                           finish_reason='length' if len(calls) == 1 else 'stop')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.ns['process_chunk'](client, 'fake', 'Full source.', 1, 1)
        self.assertEqual(result, [entry('Full source.')])
        self.assertEqual(len(calls), 2)

    def test_incomplete_or_uncovered_chunk_exhausts_retries(self):
        for raw, finish_reason in [(json.dumps([entry('First sentence.')])[:-1], 'stop'),
                                   (json.dumps([entry('First sentence.')]), 'stop'),
                                   (json.dumps([entry('First sentence. Missing ending.')]), 'length')]:
            calls = []
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason=finish_reason)])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.ns['process_chunk'](client, 'fake', 'First sentence. Missing ending.', 1, 1), [])
            self.assertEqual(len(calls), 3)

    def test_main_failed_chunk_preserves_existing_outputs(self):
        # Main is executed in a temporary synthetic project, with a stub LLM boundary.
        load_functions('app/generate_script.py', ['main', 'fix_mojibake', 'split_into_chunks', '_write_script_output'], self.ns)
        self.ns['OpenAI'] = lambda **kw: object()
        self.ns['process_chunk'] = lambda *a, **kw: []
        project = Path(self.tmp.name)
        (project / 'source.txt').write_text('Source must survive.')
        for name in ('annotated_script.json', 'chunks.json'):
            (project / name).write_text('existing')
        old_cwd, old_argv = os.getcwd(), sys.argv
        try:
            os.chdir(project / 'app')
            sys.argv = ['generate_script.py', str(project / 'source.txt')]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                self.ns['main']()
            self.assertEqual(exit_info.exception.code, 1)
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv
        for name in ('annotated_script.json', 'chunks.json'):
            self.assertEqual((project / name).read_text(), 'existing')


if __name__ == '__main__':
    unittest.main()
