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


def entry(text, speaker='NARRATOR'):
    return dict(speaker=speaker, text=text, instruct='Neutral, even narration.')


class ReviewOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(os=os, sys=sys, json=json, re=re, argparse=__import__('argparse'),
                       REVIEW_SYSTEM_PROMPT='Return JSON', REVIEW_USER_PROMPT='{context}\n{batch}')
        load_functions('app/review_script.py', ['parse_review_response', 'review_outcome', 'review_batch',
            'main', 'check_text_loss', 'normalize_text', 'diff_entries',
            'merge_consecutive_narrators', '_is_section_break'], self.ns)

    def test_strict_complete_array_strings_and_types(self):
        raw = json.dumps([entry('Literal }, { [ ] <think>keep</think>')])
        self.assertEqual(self.ns['parse_review_response']('```json\n' + raw + '\n```'), json.loads(raw))
        for bad in (raw[:-1], raw.replace('}]', '}, "lost"]'),
                    '[{"speaker":"A","text":null,"instruct":""}]'):
            self.assertIsNone(self.ns['parse_review_response'](bad))

    def test_truncated_response_is_not_reviewed(self):
        calls = []
        def create(**kw):
            calls.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps([entry('Complete.')])), finish_reason='length')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            app = Path(tmp) / 'app'
            app.mkdir()
            self.ns['__file__'] = str(app / 'review_script.py')
            self.assertIsNone(self.ns['review_batch'](client, 'fake', [entry('Complete.')], 1, 1))
        self.assertEqual(len(calls), 3)

    def run_main(self, results, contextual=False, merge=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'app').mkdir()
            self.ns['__file__'] = str(root / 'app/review_script.py')
            original = [entry('First source sentence.'), entry('Second source sentence.')]
            script = root / 'annotated_script.json'
            original_bytes = json.dumps(original).encode()
            script.write_bytes(original_bytes)
            chunks = root / 'chunks.json'
            chunks.write_text('original chunks')
            (root / 'app/config.json').write_text(json.dumps({'generation': {'review_batch_size': 1, 'merge_narrators': merge}}))
            self.ns['OpenAI'] = lambda **kw: object()
            iterator = iter(results)
            self.ns['review_batch'] = lambda *a, **kw: next(iterator)
            old_argv = sys.argv
            output = io.StringIO()
            try:
                sys.argv = ['review_script.py'] + (['--context-window', '1'] if contextual else [])
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit_info:
                    self.ns['main']()
            finally:
                sys.argv = old_argv
            return exit_info.exception.code, output.getvalue(), json.loads(script.read_text()), script.read_bytes() == original_bytes, chunks.exists()

    def test_complete_clean_only_after_all_batches_succeeded(self):
        code, logs, entries, unchanged, chunks = self.run_main([[entry('First source sentence.')], [entry('Second source sentence.')]], merge=False)
        self.assertEqual(code, 0)
        self.assertIn('Review complete:', logs)
        self.assertIn('script looks clean', logs)
        self.assertIn('Review output saved.', logs)

    def test_partial_without_changes_cannot_claim_clean(self):
        code, logs, entries, unchanged, chunks = self.run_main([[entry('First source sentence.')], None])
        self.assertEqual(code, 2)
        self.assertNotIn('script looks clean', logs)
        self.assertNotIn('completed successfully', logs)

    def test_failed_keeps_original_script_and_chunks(self):
        for contextual in (False, True):
            code, logs, entries, unchanged, chunks = self.run_main([None, None], contextual)
            self.assertEqual(code, 1)
            self.assertTrue(unchanged)
            self.assertTrue(chunks)
            self.assertIn('Review failed:', logs)
            self.assertNotIn('script looks clean', logs)
            self.assertNotIn('completed successfully', logs)
            self.assertNotIn('Review output saved.', logs)

    def test_partial_saves_successes_preserves_failed_entries_and_skips_merging(self):
        for contextual in (False, True):
            code, logs, entries, unchanged, chunks = self.run_main([[entry('First source sentence.', 'A')], None], contextual)
            self.assertEqual(code, 2)
            self.assertEqual(entries, [entry('First source sentence.', 'A'), entry('Second source sentence.')])
            self.assertFalse(chunks)
            self.assertIn('Review partial:', logs)
            self.assertIn('Review output saved.', logs)
            self.assertNotIn('script looks clean', logs)
            self.assertNotIn('completed successfully', logs)

    def test_text_loss_rejection_counts_as_failed(self):
        code, logs, entries, unchanged, chunks = self.run_main([[entry('Missing.')], None])
        self.assertEqual(code, 1)
        self.assertTrue(unchanged)

    def test_runner_and_ui_partial_refresh_without_success(self):
        state = {'review': {'running': False, 'logs': []}}
        ns = dict(List=__import__('typing').List, process_state=state,
                  logger=SimpleNamespace(info=lambda *a: None, error=lambda *a: None),
                  _stream_subprocess_to_logs=lambda *a: 2, BASE_DIR='synthetic')
        load_functions('app/app.py', ['run_process'], ns)
        ns['run_process'](['fake'], 'review')
        self.assertIn('partially completed', state['review']['logs'][-1])
        self.assertNotIn('completed successfully', '\n'.join(state['review']['logs']))
        html = (ROOT / 'app/static/index.html').read_text()
        self.assertIn('l === "Review output saved."', html)


if __name__ == '__main__':
    unittest.main()
