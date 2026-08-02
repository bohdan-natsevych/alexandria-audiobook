import os
import sys
import json
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import time
import threading
import zipfile
import subprocess
from pathlib import Path
from utils import atomic_json_write
from jobs import JobManager, JobContext
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

# Import ProjectManager
from project import ProjectManager, group_into_chunks
from tts import TTSEngine
from default_prompts import load_default_prompts
from review_prompts import load_review_prompts
from persona_prompts import load_persona_prompts
from hf_utils import fetch_builtin_manifest, download_builtin_adapter, is_adapter_downloaded

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlexandriaUI")

app = FastAPI(title="Alexandria Audiobook")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.environ.get("ALEXANDRIA_CONFIG_PATH", os.path.join(BASE_DIR, "config.json"))
LORA_MODELS_DIR = os.path.join(ROOT_DIR, "lora_models")
BUILTIN_LORA_DIR = os.path.join(ROOT_DIR, "builtin_lora")
PREPARER_SCRIPT_PATH = os.path.join(BASE_DIR, "alexandria_preparer.py")
JOB_WORKSPACES_DIR = os.path.join(ROOT_DIR, "job_workspaces")

os.makedirs(LORA_MODELS_DIR, exist_ok=True)
os.makedirs(JOB_WORKSPACES_DIR, exist_ok=True)

# Mount static files with absolute path
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Built-in and trained LoRA adapters are reusable compute assets. They are not
# exposed as static files; audio results always leave through isolated jobs.
os.makedirs(BUILTIN_LORA_DIR, exist_ok=True)


class ComputeEngineManager:
    """Lazily own only the shared model, never a server-side user project."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.engine = None
        self._lock = threading.Lock()

    def get_engine(self):
        if self.engine is not None:
            return self.engine
        with self._lock:
            if self.engine is None:
                try:
                    config = {}
                    if os.path.exists(self.config_path):
                        with open(self.config_path, "r", encoding="utf-8") as handle:
                            config = json.load(handle)
                    self.engine = TTSEngine(config)
                except Exception:
                    logger.exception("Failed to initialize TTS engine")
                    return None
        return self.engine


project_manager = ComputeEngineManager(CONFIG_PATH)
job_manager = JobManager(JOB_WORKSPACES_DIR, ttl_hours=int(os.environ.get("ALEXANDRIA_JOB_TTL_HOURS", "24")))
job_manager.cleanup_expired()
# Serialize GPU inference until a dedicated worker/queue is introduced.
tts_job_lock = threading.Lock()

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System Helpers ---

def get_gpu_stats():
    """Get current GPU memory and utilization stats."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    stats = {}
    try:
        # Memory stats (works for both NVIDIA and AMD ROCm)
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

        stats['allocated_gb'] = allocated
        stats['reserved_gb'] = reserved
        stats['total_gb'] = total
        stats['allocated_percent'] = (allocated / total * 100) if total > 0 else 0

        # Try to get utilization via rocm-smi for AMD GPUs
        try:
            result = subprocess.run(
                ['/opt/rocm/bin/rocm-smi', '--showuse', '--json'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for card_key, card_data in data.items():
                    if not isinstance(card_data, dict):
                        continue
                    for key in ('GPU use (%)', 'GPU Use (%)', 'GPU Activity'):
                        gpu_use_str = card_data.get(key)
                        if gpu_use_str is not None and gpu_use_str != 'N/A':
                            stats['utilization_percent'] = float(gpu_use_str)
                            break
                    break
        except Exception:
            stats['utilization_percent'] = None

    except Exception as e:
        logger.debug(f"Could not get GPU stats: {e}")
        return None

    return stats

def check_disk_space(path, required_gb):
    """Check if disk has enough space. Returns (has_space, free_gb)."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        return free_gb >= required_gb, free_gb
    except Exception:
        return True, 0

@app.get("/api/system/stats")
async def get_system_stats():
    """Return GPU and Disk statistics."""
    gpu = get_gpu_stats()
    # Check root dir for disk space
    has_space, free_gb = check_disk_space(ROOT_DIR, 1.0) # 1GB threshold for generic warning
    
    return {
        "gpu": gpu,
        "disk": {
            "free_gb": round(free_gb, 2),
            "low_space": not has_space
        }
    }

# Data Models
class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str

class TTSConfig(BaseModel):
    mode: str = "local"  # "local" or "external"
    url: str = "http://127.0.0.1:7860"  # external mode only
    device: str = "auto"  # local mode: "auto", "cuda:0", "cpu", etc.
    language: str = "English"  # TTS language
    parallel_workers: int = 2  # concurrent TTS workers
    batch_seed: Optional[int] = None  # Single seed for batch mode, None/-1 = random
    compile_codec: bool = False  # torch.compile the codec for ~3-4x batch throughput (slow first run)
    sub_batch_enabled: bool = True  # split batch by text length to reduce padding waste
    sub_batch_min_size: int = 4  # minimum chunks per sub-batch before allowing a split
    sub_batch_ratio: float = 5.0  # max longest/shortest length ratio before splitting
    sub_batch_max_items: int = 0  # hard cap on sequences per sub-batch (0 = auto from VRAM estimate)
    batch_group_by_type: bool = False  # group chunks by voice type for efficient batching
    pause_between_speakers_ms: int = 500  # silence (ms) between different speakers during merge
    pause_same_speaker_ms: int = 250  # silence (ms) when same speaker continues during merge

class GenerationConfig(BaseModel):
    chunk_size: int = 3000
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 0
    min_p: float = 0
    presence_penalty: float = 0.0
    banned_tokens: List[str] = []
    merge_narrators: bool = False

class PromptConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    review_system_prompt: Optional[str] = None
    review_user_prompt: Optional[str] = None
    persona_system_prompt: Optional[str] = None
    persona_user_prompt: Optional[str] = None
    persona_advanced_prompt: Optional[str] = None

class AppConfig(BaseModel):
    llm: LLMConfig
    tts: TTSConfig
    prompts: Optional[PromptConfig] = None
    generation: Optional[GenerationConfig] = None

class StatelessTTSRequest(BaseModel):
    """Client-owned chunk request used by the isolated job API.

    `voice_config` is supplied by the caller instead of being read from disk.
    A clone reference file, when required, is sent as the multipart
    `reference_audio` field.
    """
    chunk_id: str
    text: str
    speaker: str = "NARRATOR"
    instruct: str = ""
    voice_config: Dict[str, dict]

class StatelessTTSBatchChunk(BaseModel):
    chunk_id: str
    text: str
    speaker: str = "NARRATOR"
    instruct: str = ""

class StatelessTTSBatchRequest(BaseModel):
    chunks: List[StatelessTTSBatchChunk]
    voice_config: Dict[str, dict]
    batch_seed: int = -1

class ReviewJobRequest(BaseModel):
    script: List[dict]
    context_window: int = 0

class PersonaJobRequest(BaseModel):
    script: List[dict]
    voice_config: Dict[str, dict] = {}
    advanced: bool = False
    batch_size: int = 40

class VoiceDesignJobRequest(BaseModel):
    description: str
    sample_text: str
    language: Optional[str] = None
    seed: int = -1

class DatasetBuilderJobRequest(BaseModel):
    description: str
    samples: List[dict]
    seeds: Optional[List[int]] = None

class LoraTrainingRequest(BaseModel):
    name: str
    dataset_id: str
    epochs: int = 5
    lr: float = 5e-6
    batch_size: int = 1
    lora_r: int = 32
    lora_alpha: int = 128
    gradient_accumulation_steps: int = 8
    language: str = "english"

class PreparerConfig(BaseModel):
    audio_filename: str
    output_filename: str = "alexandria_dataset.zip"
    lang: str = "en"
    min_confidence: float = 0.85
    min_snr: int = 25

# Endpoints


class _EPUBTextExtractor(HTMLParser):
    BLOCK_TAGS = frozenset({"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "br", "hr", "tr", "section", "article"})
    SKIP_TAGS = frozenset({"style", "script"})

    def __init__(self):
        super().__init__()
        self.parts = []
        self.pending_newline = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self.pending_newline = True

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.pending_newline and self.parts:
            self.parts.append("\n")
            self.pending_newline = False
        self.parts.append(data)


def _extract_epub_text(epub_path: Path) -> str:
    with zipfile.ZipFile(epub_path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("Invalid EPUB container")
        opf_path = rootfile.get("full-path")
        opf = ET.fromstring(archive.read(opf_path))
        namespace = opf.tag.split("}")[0] + "}" if "}" in opf.tag else ""
        opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        manifest = {
            item.get("id"): opf_dir + item.get("href")
            for item in opf.findall(f".//{namespace}item")
            if item.get("id") and item.get("href") and "html" in item.get("media-type", "")
        }
        chapters = []
        for item in opf.findall(f".//{namespace}itemref"):
            href = manifest.get(item.get("idref"))
            if not href:
                continue
            extractor = _EPUBTextExtractor()
            extractor.feed(archive.read(href).decode("utf-8", errors="replace"))
            text = "".join(extractor.parts).strip()
            if text:
                chapters.append(text)
    return "\n\n".join(chapters)

@app.get("/")
async def read_index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/favicon.ico")
async def read_favicon():
    favicon_path = os.path.join(ROOT_DIR, "icon.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")


# --- Isolated job API -------------------------------------------------------
# The browser owns project JSON and audio. The server owns only shared model
# configuration/assets and disposable job workspaces.

def _job_response(job: dict) -> dict:
    """Return a job payload without exposing server workspace paths."""
    result = job.get("result")
    public = {key: value for key, value in job.items() if key not in {"result", "artifacts"}}
    public["status_url"] = f"/api/jobs/{job['id']}"
    public["logs_url"] = f"/api/jobs/{job['id']}/logs"
    public["cancel_url"] = f"/api/jobs/{job['id']}/cancel"
    if result:
        public["result"] = {
            "filename": result["filename"],
            "media_type": result["media_type"],
            "download_url": f"/api/jobs/{job['id']}/result",
        }
    else:
        public["result"] = None
    public["artifacts"] = [
        {
            "id": artifact["id"],
            "chunk_id": artifact.get("chunk_id"),
            "filename": artifact["filename"],
            "media_type": artifact["media_type"],
            "download_url": f"/api/jobs/{job['id']}/artifacts/{artifact['id']}",
        }
        for artifact in job.get("artifacts", [])
    ]
    return public


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        return _job_response(job_manager.get(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    try:
        job = job_manager.get(job_id)
        return {"id": job_id, "state": job["state"], "logs": job.get("logs", [])}
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    try:
        return _job_response(job_manager.cancel(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    try:
        job_manager.delete(job_id)
        return {"id": job_id, "status": "deleted"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/jobs/{job_id}/result")
async def download_job_result(job_id: str):
    try:
        path, result = job_manager.result_path(job_id)
        return FileResponse(path, filename=result["filename"], media_type=result["media_type"])
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError:
        raise HTTPException(status_code=409, detail="Job result is not ready")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job result has expired or is missing")


@app.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
async def download_job_artifact(job_id: str, artifact_id: str):
    try:
        path, artifact = job_manager.artifact_path(job_id, artifact_id)
        return FileResponse(path, filename=artifact["filename"], media_type=artifact["media_type"])
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job artifact has expired or is missing")


@app.post("/api/jobs/tts-chunk")
async def create_tts_chunk_job(
    request_json: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
):
    """Generate one chunk into a disposable workspace and return a job ID.

    The browser owns the input configuration and stores the returned WAV Blob.
    Clone references remain inside the isolated job workspace.
    """
    try:
        request = StatelessTTSRequest.model_validate_json(request_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request_json: {exc}")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Cannot generate audio for empty text")

    # Upload bytes once before queuing.  The path is patched into this job's
    # private copy of voice_config by the worker below.
    reference_bytes = await reference_audio.read() if reference_audio else None
    reference_name = os.path.basename(reference_audio.filename or "reference.wav") if reference_audio else None

    def run(context: JobContext):
        voice_config = json.loads(json.dumps(request.voice_config))
        if reference_bytes is not None:
            ref_path = context.input_dir / reference_name
            ref_path.write_bytes(reference_bytes)
            voice = voice_config.setdefault(request.speaker, {})
            voice["ref_audio"] = str(ref_path)
            context.log("Stored clone reference in isolated job workspace")

        if context.cancelled():
            return
        context.set_progress(0, 1, "Loading TTS engine")
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", request.chunk_id).strip("._") or "chunk"
        output_path = context.output_dir / f"{safe_id}.wav"
        context.log(f"Generating chunk {request.chunk_id}")
        with tts_job_lock:
            engine = project_manager.get_engine()
            if not engine:
                raise RuntimeError("TTS engine not initialized")
            success = engine.generate_voice(
                request.text,
                request.instruct,
                request.speaker,
                voice_config,
                str(output_path),
            )
        if context.cancelled():
            return
        if not success or not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("TTS generation failed")
        context.set_progress(1, 1, "Audio generated")
        context.set_result(output_path, "audio/wav", output_path.name)

    return _job_response(job_manager.submit("tts-chunk", run))


@app.post("/api/jobs/tts-batch")
async def create_tts_batch_job(
    request_json: str = Form(...),
    reference_audio: List[UploadFile] = File(default=[]),
):
    """Generate many client-owned chunks efficiently and return a ZIP result.

    Reference uploads are matched by their original filename against the
    `ref_audio` value in the supplied voice configuration.  The client should
    therefore use distinct filenames for clone references in a batch.
    """
    try:
        request = StatelessTTSBatchRequest.model_validate_json(request_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request_json: {exc}")
    chunks = [chunk for chunk in request.chunks if chunk.text.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No non-empty chunks supplied")

    uploads = []
    for file in reference_audio:
        uploads.append((os.path.basename(file.filename or "reference.wav"), await file.read()))

    def run(context: JobContext):
        voice_config = json.loads(json.dumps(request.voice_config))
        uploaded_paths = {}
        for filename, content in uploads:
            path = context.input_dir / filename
            path.write_bytes(content)
            uploaded_paths[filename] = str(path)
        for voice in voice_config.values():
            ref_audio = voice.get("ref_audio")
            if ref_audio:
                replacement = uploaded_paths.get(os.path.basename(ref_audio))
                if replacement:
                    voice["ref_audio"] = replacement

        if context.cancelled():
            return
        context.set_progress(0, len(chunks), "Generating batch")
        batch_input = [
            {"index": index, "text": chunk.text, "instruct": chunk.instruct, "speaker": chunk.speaker}
            for index, chunk in enumerate(chunks)
        ]
        with tts_job_lock:
            engine = project_manager.get_engine()
            if not engine:
                raise RuntimeError("TTS engine not initialized")
            generated = engine.generate_batch(batch_input, voice_config, str(context.work_dir), request.batch_seed)
        if context.cancelled():
            return

        manifest = {"completed": [], "failed": []}
        artifacts = []
        archive_path = context.output_dir / "tts_batch.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in generated["completed"]:
                source = context.work_dir / f"temp_batch_{index}.wav"
                if not source.is_file() or source.stat().st_size == 0:
                    manifest["failed"].append({"chunk_id": chunks[index].chunk_id, "error": "Generated file missing"})
                    continue
                safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", chunks[index].chunk_id).strip("._") or f"chunk_{index}"
                archive_name = f"audio/{safe_id}.wav"
                artifact_name = f"{index:04d}_{safe_id}.wav"
                artifact_path = context.output_dir / artifact_name
                shutil.copyfile(source, artifact_path)
                archive.write(artifact_path, archive_name)
                manifest["completed"].append({"chunk_id": chunks[index].chunk_id, "file": archive_name})
                artifacts.append({
                    "id": str(index),
                    "chunk_id": chunks[index].chunk_id,
                    "path": str(artifact_path.relative_to(context.workspace)),
                    "filename": artifact_name,
                    "media_type": "audio/wav",
                })
            for index, error in generated["failed"]:
                chunk_id = chunks[index].chunk_id if 0 <= index < len(chunks) else str(index)
                manifest["failed"].append({"chunk_id": chunk_id, "error": str(error)})
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        context.set_progress(len(manifest["completed"]), len(chunks), "Batch complete")
        context.set_artifacts(artifacts)
        context.set_result(archive_path, "application/zip", archive_path.name)

    return _job_response(job_manager.submit("tts-batch", run))


@app.post("/api/jobs/script-generate")
async def create_script_generation_job(source_file: UploadFile = File(...)):
    """Run the existing annotation pipeline against a job-local source file.

    The subprocess runs entirely inside the disposable job workspace.
    """
    filename = os.path.basename(source_file.filename or "source.txt")
    if not filename.lower().endswith((".txt", ".md", ".epub")):
        raise HTTPException(status_code=400, detail="Use EPUB, plain-text, or Markdown source")
    source_bytes = await source_file.read()
    if not source_bytes.strip():
        raise HTTPException(status_code=400, detail="Source file is empty")

    def run(context: JobContext):
        input_path = context.input_dir / filename
        input_path.write_bytes(source_bytes)
        if input_path.suffix.lower() == ".epub":
            text_path = context.input_dir / f"{input_path.stem}.txt"
            text_path.write_text(_extract_epub_text(input_path), encoding="utf-8")
            if not text_path.read_text(encoding="utf-8").strip():
                raise RuntimeError("No readable text found in EPUB")
            input_path = text_path
        context.set_progress(0, 1, "Generating annotated script")
        command = [sys.executable, "-u", os.path.join(BASE_DIR, "generate_script.py"), str(input_path)]
        context.log(f"Starting script generation for {filename}")
        returncode = context.run_subprocess(command, context.work_dir)
        if context.cancelled():
            return
        if returncode != 0:
            raise RuntimeError(f"Script generator exited with code {returncode}")
        generated_path = context.workspace / "annotated_script.json"
        if not generated_path.is_file():
            raise RuntimeError("Script generator did not produce annotated_script.json")
        with generated_path.open("r", encoding="utf-8") as handle:
            script = json.load(handle)
        chunks = group_into_chunks(script)
        for index, chunk in enumerate(chunks):
            chunk.update({"id": index, "status": "pending"})
        result_path = context.output_dir / "project.json"
        result_path.write_text(json.dumps({"script": script, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
        context.set_progress(1, 1, "Script generated")
        context.set_result(result_path, "application/json", result_path.name)

    return _job_response(job_manager.submit("script-generate", run))


@app.post("/api/jobs/script-review")
async def create_script_review_job(request: ReviewJobRequest):
    if not request.script:
        raise HTTPException(status_code=400, detail="Script is empty")
    window = max(0, min(request.context_window, 12))

    def run(context: JobContext):
        input_path = context.input_dir / "script.json"
        reviewed_path = context.work_dir / "reviewed_script.json"
        input_path.write_text(json.dumps(request.script, ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, "-u", os.path.join(BASE_DIR, "review_script.py"), "--input", str(input_path), "--output", str(reviewed_path)]
        if window:
            command.extend(["--context-window", str(window)])
        context.set_progress(0, 1, "Reviewing script")
        returncode = context.run_subprocess(command, context.work_dir)
        if context.cancelled(): return
        if returncode != 0:
            raise RuntimeError(f"Script reviewer exited with code {returncode}")
        reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
        chunks = group_into_chunks(reviewed)
        for index, chunk in enumerate(chunks): chunk.update({"id": index, "status": "pending"})
        result_path = context.output_dir / "project.json"
        result_path.write_text(json.dumps({"script": reviewed, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
        context.set_progress(1, 1, "Review complete")
        context.set_result(result_path, "application/json", result_path.name)

    return _job_response(job_manager.submit("script-review", run))


@app.post("/api/jobs/personas-generate")
async def create_persona_job(request: PersonaJobRequest):
    if not request.script:
        raise HTTPException(status_code=400, detail="Script is empty")

    def run(context: JobContext):
        (context.workspace / "annotated_script.json").write_text(json.dumps(request.script, ensure_ascii=False), encoding="utf-8")
        (context.workspace / "voice_config.json").write_text(json.dumps(request.voice_config, ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, "-u", os.path.join(BASE_DIR, "generate_personas.py"), "--root", str(context.workspace)]
        if request.advanced:
            command.extend(["--advanced", "--batch-size", str(max(1, min(request.batch_size, 200)))])
        context.set_progress(0, 1, "Generating personas")
        returncode = context.run_subprocess(command, context.work_dir)
        if context.cancelled(): return
        if returncode != 0:
            raise RuntimeError(f"Persona generator exited with code {returncode}")
        archive_path = context.output_dir / "personas.zip"
        artifacts = []
        config_path = context.workspace / "voice_config.json"
        config_result = context.output_dir / "voice_config.json"
        voice_config = json.loads(config_path.read_text(encoding="utf-8"))
        for speaker, voice in voice_config.items():
            ref_audio = voice.get("ref_audio")
            if not ref_audio:
                continue
            source = (context.workspace / ref_audio).resolve()
            if context.workspace.resolve() not in source.parents or not source.is_file():
                continue
            filename = source.name
            output_preview = context.output_dir / f"persona_{_sanitize_name(speaker)}_{filename}"
            shutil.copyfile(source, output_preview)
            voice["ref_audio"] = f"client_refs/{filename}"
            artifacts.append({
                "id": f"persona-{len(artifacts)}",
                "chunk_id": speaker,
                "path": str(output_preview.relative_to(context.workspace)),
                "filename": filename,
                "media_type": "audio/wav",
            })
        config_result.write_text(json.dumps(voice_config, ensure_ascii=False), encoding="utf-8")
        artifacts.append({"id": "voice-config", "path": str(config_result.relative_to(context.workspace)), "filename": config_result.name, "media_type": "application/json"})
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(config_result, "voice_config.json")
            for artifact in artifacts:
                if artifact["media_type"] == "audio/wav":
                    archive.write(context.workspace / artifact["path"], f"references/{artifact['filename']}")
        context.set_progress(1, 1, "Personas generated")
        context.set_artifacts(artifacts)
        context.set_result(archive_path, "application/zip", archive_path.name)

    return _job_response(job_manager.submit("personas-generate", run))


@app.post("/api/jobs/export")
async def create_export_job(
    kind: str = Form(...),
    manifest_json: str = Form(...),
    audio_zip: Optional[UploadFile] = File(None),
    audio_files: List[UploadFile] = File(default=[]),
    cover: Optional[UploadFile] = File(None),
):
    """Assemble client-held chunks with native server audio tooling.

    The manifest contains `chunks`, in playback order, with a `file` member
    naming the corresponding ZIP entry.  The uploaded ZIP and final output
    exist only for this job's TTL.
    """
    if kind not in {"merge", "m4b", "audacity"}:
        raise HTTPException(status_code=400, detail="kind must be merge, m4b, or audacity")
    try:
        manifest = json.loads(manifest_json)
        if not isinstance(manifest.get("chunks"), list): raise ValueError("chunks is required")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid manifest_json: {exc}")
    if audio_zip is None and not audio_files:
        raise HTTPException(status_code=400, detail="Provide audio_zip or audio_files")
    zip_bytes = await audio_zip.read() if audio_zip else None
    uploaded_audio = [(os.path.basename(file.filename or "audio.wav"), await file.read()) for file in audio_files]
    cover_bytes = await cover.read() if cover else None
    cover_name = os.path.basename(cover.filename or "cover.jpg") if cover else None

    def run(context: JobContext):
        extracted = context.input_dir / "chunks"
        extracted.mkdir()
        if zip_bytes is not None:
            source_zip = context.input_dir / "chunks.zip"
            source_zip.write_bytes(zip_bytes)
            with zipfile.ZipFile(source_zip) as archive:
                for member in archive.infolist():
                    target = (extracted / member.filename).resolve()
                    if extracted.resolve() not in target.parents or member.is_dir():
                        if member.is_dir(): continue
                        raise RuntimeError("Unsafe ZIP entry")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        else:
            for filename, content in uploaded_audio:
                (extracted / filename).write_bytes(content)
        voicelines = context.workspace / "voicelines"
        voicelines.mkdir(exist_ok=True)
        job_chunks = []
        for index, chunk in enumerate(manifest["chunks"]):
            source = (extracted / str(chunk.get("file", ""))).resolve()
            if extracted.resolve() not in source.parents or not source.is_file():
                raise RuntimeError(f"Missing audio ZIP entry for chunk {index}")
            filename = f"{index:05d}_{os.path.basename(source.name)}"
            shutil.copyfile(source, voicelines / filename)
            job_chunk = dict(chunk)
            job_chunk["audio_path"] = f"voicelines/{filename}"
            job_chunks.append(job_chunk)
        (context.workspace / "chunks.json").write_text(json.dumps(job_chunks, ensure_ascii=False), encoding="utf-8")
        job_app_dir = context.workspace / "app"
        job_app_dir.mkdir(exist_ok=True)
        if os.path.exists(CONFIG_PATH):
            shutil.copyfile(CONFIG_PATH, job_app_dir / "config.json")
        manager = ProjectManager(str(context.workspace))
        context.set_progress(0, 1, f"Creating {kind} export")
        if kind == "merge":
            success, message = manager.merge_audio()
            source = context.workspace / "cloned_audiobook.mp3"
            media_type = "audio/mpeg"
            filename = "audiobook.mp3"
        elif kind == "audacity":
            success, message = manager.export_audacity()
            source = Path(message) if success else context.workspace / "audacity_export.zip"
            media_type = "application/zip"
            filename = "audacity_export.zip"
        else:
            metadata = dict(manifest.get("metadata") or {})
            if cover_bytes is not None:
                cover_path = context.input_dir / cover_name
                cover_path.write_bytes(cover_bytes)
                metadata["cover_path"] = str(cover_path)
            success, message = manager.merge_m4b(bool(manifest.get("per_chunk_chapters")), metadata)
            source = context.workspace / "audiobook.m4b"
            media_type = "audio/mp4"
            filename = "audiobook.m4b"
        if not success or not source.is_file(): raise RuntimeError(str(message))
        result = context.output_dir / filename
        shutil.copyfile(source, result)
        context.set_progress(1, 1, "Export complete")
        context.set_result(result, media_type, filename)

    return _job_response(job_manager.submit(f"export-{kind}", run))


@app.post("/api/jobs/voice-design")
async def create_voice_design_job(request: VoiceDesignJobRequest):
    if not request.description.strip() or not request.sample_text.strip():
        raise HTTPException(status_code=400, detail="description and sample_text are required")

    def run(context: JobContext):
        context.set_progress(0, 1, "Generating voice design")
        with tts_job_lock:
            engine = project_manager.get_engine()
            if not engine: raise RuntimeError("TTS engine not initialized")
            wav_path, _ = engine.generate_voice_design(request.description, request.sample_text, request.language, request.seed)
        result = context.output_dir / "voice_design_preview.wav"
        shutil.copyfile(wav_path, result)
        try: os.remove(wav_path)
        except OSError: pass
        context.set_progress(1, 1, "Preview generated")
        context.set_result(result, "audio/wav", result.name)

    return _job_response(job_manager.submit("voice-design", run))


@app.post("/api/jobs/dataset-builder")
async def create_dataset_builder_job(request: DatasetBuilderJobRequest):
    samples = [sample for sample in request.samples if str(sample.get("text", "")).strip()]
    if not samples: raise HTTPException(status_code=400, detail="At least one non-empty sample is required")

    def run(context: JobContext):
        archive_path = context.output_dir / "voice_dataset_samples.zip"
        metadata = []
        artifacts = []
        with tts_job_lock:
            engine = project_manager.get_engine()
            if not engine: raise RuntimeError("TTS engine not initialized")
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, sample in enumerate(samples):
                    if context.cancelled(): return
                    emotion = str(sample.get("emotion", "")).strip()
                    description = request.description if not emotion else f"{request.description}. {emotion}"
                    seed = request.seeds[index] if request.seeds and index < len(request.seeds) else int(sample.get("seed", -1) or -1)
                    wav_path, _ = engine.generate_voice_design(description, str(sample["text"]), seed=seed)
                    filename = f"sample_{index:03d}.wav"
                    output_sample = context.output_dir / filename
                    shutil.copyfile(wav_path, output_sample)
                    archive.write(output_sample, filename)
                    try: os.remove(wav_path)
                    except OSError: pass
                    metadata.append({"audio_filepath": filename, "text": str(sample["text"]), "emotion": emotion, "ref_audio": "ref.wav"})
                    artifacts.append({"id": str(index), "chunk_id": str(sample.get("client_id", index)), "path": str(output_sample.relative_to(context.workspace)), "filename": filename, "media_type": "audio/wav"})
                    context.set_progress(index + 1, len(samples), f"Generated {index + 1}/{len(samples)} samples")
                if artifacts:
                    first_sample = context.workspace / artifacts[0]["path"]
                    archive.write(first_sample, "ref.wav")
                    archive.writestr("ref_text.txt", metadata[0]["text"])
                archive.writestr("metadata.jsonl", "\n".join(json.dumps(item, ensure_ascii=False) for item in metadata) + "\n")
        context.set_artifacts(artifacts)
        context.set_result(archive_path, "application/zip", archive_path.name)

    return _job_response(job_manager.submit("dataset-builder", run))


@app.post("/api/jobs/lora-train")
async def create_lora_training_job(config_json: str = Form(...), dataset_zip: UploadFile = File(...)):
    try:
        config = LoraTrainingRequest(**json.loads(config_json))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config_json: {exc}")
    dataset_bytes = await dataset_zip.read()

    def run(context: JobContext):
        dataset_dir = context.input_dir / "dataset"
        dataset_dir.mkdir()
        source_zip = context.input_dir / "dataset.zip"
        source_zip.write_bytes(dataset_bytes)
        with zipfile.ZipFile(source_zip) as archive:
            for member in archive.infolist():
                target = (dataset_dir / member.filename).resolve()
                if dataset_dir.resolve() not in target.parents or member.is_dir():
                    if member.is_dir(): continue
                    raise RuntimeError("Unsafe dataset ZIP entry")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination: shutil.copyfileobj(source, destination)
        output_dir = context.work_dir / "adapter"
        command = [sys.executable, "-u", os.path.join(BASE_DIR, "train_lora.py"), "--data_dir", str(dataset_dir), "--output_dir", str(output_dir), "--epochs", str(config.epochs), "--lr", str(config.lr), "--batch_size", str(config.batch_size), "--lora_r", str(config.lora_r), "--lora_alpha", str(config.lora_alpha), "--gradient_accumulation_steps", str(config.gradient_accumulation_steps), "--language", config.language]
        returncode = context.run_subprocess(command, context.work_dir)
        if context.cancelled(): return
        if returncode != 0: raise RuntimeError(f"LoRA training exited with code {returncode}")
        # Adapters are server-side compute assets (not user audio).  Register
        # them so future TTS jobs can reuse them without re-uploading weights.
        adapter_id = f"{_sanitize_name(config.name) or 'adapter'}_{int(time.time())}"
        persistent_adapter = Path(LORA_MODELS_DIR) / adapter_id
        shutil.copytree(output_dir, persistent_adapter)
        manifest = _load_manifest(LORA_MODELS_MANIFEST)
        manifest.append({"id": adapter_id, "name": config.name, "created": time.time(), "job_id": context.job_id})
        _save_manifest(LORA_MODELS_MANIFEST, manifest)
        result = context.output_dir / "adapter.zip"
        with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in output_dir.rglob("*"):
                if file.is_file(): archive.write(file, str(file.relative_to(output_dir)))
        context.set_artifacts([{"id": "adapter", "chunk_id": adapter_id, "path": str(result.relative_to(context.workspace)), "filename": result.name, "media_type": "application/zip"}])
        context.set_result(result, "application/zip", result.name)

    return _job_response(job_manager.submit("lora-train", run))


@app.post("/api/jobs/preparer")
async def create_preparer_job(config_json: str = Form(...), audio_file: UploadFile = File(...)):
    if not os.path.exists(PREPARER_SCRIPT_PATH):
        raise HTTPException(status_code=503, detail="Preparer script is not installed")
    try: config = PreparerConfig(**json.loads(config_json))
    except Exception as exc: raise HTTPException(status_code=422, detail=f"Invalid config_json: {exc}")
    audio_bytes = await audio_file.read()
    audio_name = os.path.basename(audio_file.filename or config.audio_filename)

    def run(context: JobContext):
        audio_path = context.input_dir / audio_name
        audio_path.write_bytes(audio_bytes)
        result = context.output_dir / (config.output_filename or "alexandria_dataset.zip")
        command = [sys.executable, "-u", PREPARER_SCRIPT_PATH, "--audio", str(audio_path), "--output", str(result), "--lang", config.lang, "--min-confidence", str(config.min_confidence), "--min-snr", str(config.min_snr)]
        returncode = context.run_subprocess(command, context.work_dir)
        if context.cancelled(): return
        if returncode != 0 or not result.is_file(): raise RuntimeError("Preparer failed")
        context.set_result(result, "application/zip", result.name)

    return _job_response(job_manager.submit("preparer", run))

@app.get("/api/config")
async def get_config():
    default_config = {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "local",
            "model_name": "richardyoung/qwen3-14b-abliterated:Q8_0"
        },
        "tts": {
            "mode": "local",
            "url": "http://127.0.0.1:7860",
            "device": "auto"
        },
        "prompts": {
            "system_prompt": "",
            "user_prompt": ""
        }
    }

    if not os.path.exists(CONFIG_PATH):
        sys_prompt, usr_prompt = load_default_prompts()
        default_config["prompts"]["system_prompt"] = sys_prompt
        default_config["prompts"]["user_prompt"] = usr_prompt
        try:
            rev_sys, rev_usr = load_review_prompts()
            default_config["prompts"]["review_system_prompt"] = rev_sys
            default_config["prompts"]["review_user_prompt"] = rev_usr
        except RuntimeError:
            pass
        try:
            per_sys, per_usr, per_adv = load_persona_prompts()
            default_config["prompts"]["persona_system_prompt"] = per_sys
            default_config["prompts"]["persona_user_prompt"] = per_usr
            default_config["prompts"]["persona_advanced_prompt"] = per_adv
        except RuntimeError:
            pass
        config = default_config
    else:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Ensure prompts section exists with defaults from file
    if "prompts" not in config:
        sys_prompt, usr_prompt = load_default_prompts()
        prompts = {"system_prompt": sys_prompt, "user_prompt": usr_prompt}
        try:
            rev_sys, rev_usr = load_review_prompts()
            prompts["review_system_prompt"] = rev_sys
            prompts["review_user_prompt"] = rev_usr
        except RuntimeError:
            pass
        try:
            per_sys, per_usr, per_adv = load_persona_prompts()
            prompts["persona_system_prompt"] = per_sys
            prompts["persona_user_prompt"] = per_usr
            prompts["persona_advanced_prompt"] = per_adv
        except RuntimeError:
            pass
        config["prompts"] = prompts
    else:
        if not config["prompts"].get("system_prompt") or not config["prompts"].get("user_prompt"):
            sys_prompt, usr_prompt = load_default_prompts()
            if not config["prompts"].get("system_prompt"):
                config["prompts"]["system_prompt"] = sys_prompt
            if not config["prompts"].get("user_prompt"):
                config["prompts"]["user_prompt"] = usr_prompt
        if not config["prompts"].get("review_system_prompt") or not config["prompts"].get("review_user_prompt"):
            try:
                rev_sys, rev_usr = load_review_prompts()
                if not config["prompts"].get("review_system_prompt"):
                    config["prompts"]["review_system_prompt"] = rev_sys
                if not config["prompts"].get("review_user_prompt"):
                    config["prompts"]["review_user_prompt"] = rev_usr
            except RuntimeError:
                pass  # review_prompts.txt missing or malformed — leave fields empty
        if not config["prompts"].get("persona_system_prompt") or not config["prompts"].get("persona_user_prompt") or not config["prompts"].get("persona_advanced_prompt"):
            try:
                per_sys, per_usr, per_adv = load_persona_prompts()
                if not config["prompts"].get("persona_system_prompt"):
                    config["prompts"]["persona_system_prompt"] = per_sys
                if not config["prompts"].get("persona_user_prompt"):
                    config["prompts"]["persona_user_prompt"] = per_usr
                if not config["prompts"].get("persona_advanced_prompt"):
                    config["prompts"]["persona_advanced_prompt"] = per_adv
            except RuntimeError:
                pass

    return config

@app.get("/api/default_prompts")
async def get_default_prompts():
    system_prompt, user_prompt = load_default_prompts()
    result = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }
    try:
        review_sys, review_usr = load_review_prompts()
        result["review_system_prompt"] = review_sys
        result["review_user_prompt"] = review_usr
    except RuntimeError:
        pass
    try:
        persona_sys, persona_usr, persona_adv = load_persona_prompts()
        result["persona_system_prompt"] = persona_sys
        result["persona_user_prompt"] = persona_usr
        result["persona_advanced_prompt"] = persona_adv
    except RuntimeError:
        pass
    return result

@app.post("/api/config")
async def save_config(config: AppConfig):
    config_dir = os.path.dirname(CONFIG_PATH)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
    # Reset engine so it picks up new TTS settings on next use
    project_manager.engine = None
    return {"status": "saved"}


# --- Reusable server-side compute assets -----------------------------------

LORA_MODELS_MANIFEST = os.path.join(LORA_MODELS_DIR, "manifest.json")


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\- ]", "", name).strip()
    return re.sub(r"\s+", "_", name).lower()


def _load_manifest(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
                return value if isinstance(value, list) else []
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return []


def _save_manifest(path: str, manifest: list) -> None:
    atomic_json_write(manifest, path)


def _load_builtin_lora_manifest() -> list:
    result = []
    for source in fetch_builtin_manifest(BUILTIN_LORA_DIR):
        entry = dict(source)
        local_id = entry["id"] if entry["id"].startswith("builtin_") else f"builtin_{entry['id']}"
        downloaded = is_adapter_downloaded(local_id, BUILTIN_LORA_DIR)
        entry.update({
            "id": local_id,
            "builtin": True,
            "downloaded": downloaded,
            "adapter_path": f"builtin_lora/{local_id}" if downloaded else None,
            "preview_audio_url": None,
        })
        result.append(entry)
    return result


@app.get("/api/lora/models")
async def lora_list_models():
    trained = _load_manifest(LORA_MODELS_MANIFEST)
    for entry in trained:
        entry.update({
            "builtin": False,
            "downloaded": True,
            "adapter_path": f"lora_models/{entry['id']}",
            "preview_audio_url": None,
        })
    return _load_builtin_lora_manifest() + trained


@app.delete("/api/lora/models/{adapter_id}")
async def lora_delete_model(adapter_id: str):
    if any(entry["id"] == adapter_id for entry in _load_builtin_lora_manifest()):
        raise HTTPException(status_code=403, detail="Built-in adapters cannot be deleted")
    manifest = _load_manifest(LORA_MODELS_MANIFEST)
    if not any(entry.get("id") == adapter_id for entry in manifest):
        raise HTTPException(status_code=404, detail="Adapter not found")
    adapter_dir = (Path(LORA_MODELS_DIR) / adapter_id).resolve()
    root = Path(LORA_MODELS_DIR).resolve()
    if root not in adapter_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid adapter ID")
    if adapter_dir.is_dir():
        shutil.rmtree(adapter_dir)
    _save_manifest(LORA_MODELS_MANIFEST, [entry for entry in manifest if entry.get("id") != adapter_id])
    project_manager.engine = None
    return {"status": "deleted", "adapter_id": adapter_id}


@app.post("/api/lora/download/{adapter_id}")
async def lora_download_builtin(adapter_id: str):
    remote_manifest = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    hf_name = adapter_id.removeprefix("builtin_")
    if not any(entry.get("id") in {hf_name, adapter_id} for entry in remote_manifest):
        raise HTTPException(status_code=404, detail=f"Unknown built-in adapter: {adapter_id}")
    if is_adapter_downloaded(adapter_id, BUILTIN_LORA_DIR):
        return {"status": "already_downloaded", "adapter_id": adapter_id}
    try:
        download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
    except Exception as exc:
        logger.exception("Built-in adapter download failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "downloaded", "adapter_id": adapter_id}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("ALEXANDRIA_HOST", "127.0.0.1")
    port = int(os.environ.get("ALEXANDRIA_PORT", "4200"))
    uvicorn.run(app, host=host, port=port, access_log=False)
