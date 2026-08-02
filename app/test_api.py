"""Contract tests for the client-owned, job-only server."""

import json
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# The route contract does not require native audio libraries. Keep these tests
# runnable in a lightweight development interpreter as well as the full image.
try:
    import soundfile  # noqa: F401
except ImportError:
    module = types.ModuleType("soundfile")
    module.read = lambda *args, **kwargs: None
    module.write = lambda *args, **kwargs: None
    sys.modules["soundfile"] = module

try:
    import pydub  # noqa: F401
except ImportError:
    module = types.ModuleType("pydub")
    module.AudioSegment = type("AudioSegment", (), {})
    sys.modules["pydub"] = module

from fastapi.testclient import TestClient

server_spec = importlib.util.spec_from_file_location("alexandria_server", APP_DIR / "app.py")
server = importlib.util.module_from_spec(server_spec)
server_spec.loader.exec_module(server)
from jobs import JobManager


class JobOnlyRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_route_table_contains_no_project_state_endpoints(self):
        allowed_exact = {
            "/", "/favicon.ico", "/static", "/openapi.json", "/docs",
            "/docs/oauth2-redirect", "/redoc", "/api/config",
            "/api/default_prompts", "/api/system/stats", "/api/lora/models",
            "/api/lora/models/{adapter_id}", "/api/lora/download/{adapter_id}",
        }
        paths = {route.path for route in server.app.routes}
        unexpected = {
            path for path in paths
            if path not in allowed_exact and not path.startswith("/api/jobs/")
        }
        self.assertEqual(set(), unexpected)

    def test_index_serves_client_store(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        self.assertIn("alexandria-client-project", response.text)
        self.assertIn("/api/jobs/tts-chunk", response.text)

    def test_unknown_job_is_404(self):
        response = self.client.get("/api/jobs/not-a-job")
        self.assertEqual(404, response.status_code)

    def test_tts_rejects_empty_text_before_loading_model(self):
        request = {
            "chunk_id": "0", "text": "", "speaker": "NARRATOR",
            "instruct": "", "voice_config": {},
        }
        response = self.client.post(
            "/api/jobs/tts-chunk",
            data={"request_json": json.dumps(request)},
        )
        self.assertEqual(400, response.status_code)

    def test_script_job_rejects_unsupported_source(self):
        response = self.client.post(
            "/api/jobs/script-generate",
            files={"source_file": ("book.pdf", b"not a pdf", "application/pdf")},
        )
        self.assertEqual(400, response.status_code)

    def test_dataset_builder_requires_samples(self):
        response = self.client.post(
            "/api/jobs/dataset-builder",
            json={"description": "voice", "samples": []},
        )
        self.assertEqual(400, response.status_code)

    def test_export_requires_client_audio(self):
        response = self.client.post(
            "/api/jobs/export",
            data={"kind": "merge", "manifest_json": json.dumps({"chunks": []})},
        )
        self.assertEqual(400, response.status_code)

    def test_old_endpoints_are_404(self):
        for path in ("/api/chunks", "/api/voices", "/api/status/audio", "/api/scripts"):
            with self.subTest(path=path):
                self.assertEqual(404, self.client.get(path).status_code)


class JobManagerTests(unittest.TestCase):
    def test_result_artifact_and_delete_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, ttl_hours=1)

            def work(context):
                result = context.output_dir / "result.txt"
                result.write_text("done", encoding="utf-8")
                artifact = context.output_dir / "artifact.txt"
                artifact.write_text("part", encoding="utf-8")
                context.set_progress(1, 1, "complete")
                context.set_artifacts([{
                    "id": "part", "path": str(artifact.relative_to(context.workspace)),
                    "filename": artifact.name, "media_type": "text/plain",
                }])
                context.set_result(result, "text/plain", result.name)

            job = manager.submit("test", work)
            deadline = time.time() + 5
            completed = manager.get(job["id"])
            while completed["state"] not in {"succeeded", "failed", "cancelled"} and time.time() < deadline:
                time.sleep(0.01)
                completed = manager.get(job["id"])
            self.assertEqual("succeeded", completed["state"])
            result_path, _ = manager.result_path(job["id"])
            artifact_path, _ = manager.artifact_path(job["id"], "part")
            self.assertEqual("done", Path(result_path).read_text(encoding="utf-8"))
            self.assertEqual("part", Path(artifact_path).read_text(encoding="utf-8"))
            manager.delete(job["id"])
            with self.assertRaises(KeyError):
                manager.get(job["id"])

    def test_cancel_terminates_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, ttl_hours=1)

            def work(context):
                returncode = context.run_subprocess(
                    [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(10)"],
                    context.work_dir,
                )
                if not context.cancelled() and returncode:
                    raise RuntimeError(f"child exited {returncode}")

            job = manager.submit("cancel-test", work)
            deadline = time.time() + 2
            while manager.get(job["id"])["state"] != "running" and time.time() < deadline:
                time.sleep(0.01)
            manager.cancel(job["id"])
            deadline = time.time() + 5
            state = manager.get(job["id"])
            while state["state"] not in {"cancelled", "failed"} and time.time() < deadline:
                time.sleep(0.02)
                state = manager.get(job["id"])
            self.assertEqual("cancelled", state["state"])


if __name__ == "__main__":
    unittest.main()
