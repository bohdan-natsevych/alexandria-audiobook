"""Build a LoRA-ready voice dataset from user-provided speech audio."""

import argparse
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-snr", type=float, default=25.0)
    return parser.parse_args()


def estimate_snr(audio: AudioSegment) -> float:
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float64)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    if not samples.size:
        return float("-inf")
    frame_size = max(1, int(audio.frame_rate * 0.05))
    frame_rms = []
    for offset in range(0, len(samples), frame_size):
        frame = samples[offset:offset + frame_size]
        if frame.size:
            frame_rms.append(math.sqrt(float(np.mean(frame * frame)) + 1e-12))
    signal = math.sqrt(float(np.mean(samples * samples)) + 1e-12)
    noise = max(float(np.percentile(frame_rms, 20)), 1e-6)
    return 20.0 * math.log10(max(signal, 1e-6) / noise)


def main():
    args = parse_args()
    audio_path = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("ALEXANDRIA_WHISPER_MODEL", "small")
    device = os.environ.get("ALEXANDRIA_WHISPER_DEVICE", "auto")
    print(f"[LOAD] faster-whisper model={model_name} device={device}", flush=True)
    model = WhisperModel(model_name, device=device, compute_type="default")
    language = None if args.lang.lower() in {"", "auto"} else args.lang
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    print(f"[ASR] language={info.language} probability={info.language_probability:.3f}", flush=True)
    source = AudioSegment.from_file(audio_path)

    with tempfile.TemporaryDirectory(prefix="alexandria-preparer-") as temporary:
        work = Path(temporary)
        metadata = []
        first_audio = None
        first_text = ""
        seen = 0
        for segment in segments:
            seen += 1
            text = segment.text.strip()
            confidence = math.exp(min(0.0, float(segment.avg_logprob)))
            if not text or confidence < args.min_confidence:
                print(f"[SKIP] segment={seen} confidence={confidence:.3f}", flush=True)
                continue
            clip = source[max(0, int(segment.start * 1000)):min(len(source), int(segment.end * 1000))]
            clip = clip.set_channels(1).set_frame_rate(24000)
            snr = estimate_snr(clip)
            if snr < args.min_snr:
                print(f"[SKIP] segment={seen} snr={snr:.1f}dB", flush=True)
                continue
            filename = f"sample_{len(metadata):04d}.wav"
            destination = work / filename
            clip.export(destination, format="wav")
            metadata.append({"audio_filepath": filename, "text": text, "ref_audio": "ref.wav"})
            if first_audio is None:
                first_audio = destination
                first_text = text
            print(f"[KEEP] {filename} confidence={confidence:.3f} snr={snr:.1f}dB", flush=True)

        if not metadata or first_audio is None:
            raise RuntimeError("No segments passed the confidence and SNR thresholds")
        shutil.copyfile(first_audio, work / "ref.wav")
        (work / "ref_text.txt").write_text(first_text, encoding="utf-8")
        (work / "metadata.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in metadata) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in work.iterdir():
                archive.write(file, file.name)
        print(f"[DONE] wrote {len(metadata)} samples to {output_path}", flush=True)


if __name__ == "__main__":
    main()
