"""Local ESpeech (F5-TTS Russian) backend for Alexandria clone voices."""

import gc
import os
import threading

import numpy as np
import soundfile as sf
import torch

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MODEL_REPO = "ESpeech/ESpeech-TTS-1_RL-V2"
MODEL_FILE = "espeech_tts_rlv2.pt"
VOCAB_FILE = "vocab.txt"


class ESpeechEngine:
    """Lazy-loaded ESpeech synthesizer (reference-audio cloning only)."""

    def __init__(self, device="auto"):
        self._device_pref = device
        # CLAUDE CODE: Reentrant because synthesize() holds it while calling
        # _ensure_loaded(). It also serializes inference: parallel TTS workers
        # share this instance, and RUAccent, the F5 model and the global torch
        # seed are not safe to use from several threads at once.
        self._lock = threading.RLock()
        self._model = None
        self._vocoder = None
        self._accentizer = None
        self._device = None

    def _resolve_device(self):
        if self._device_pref and self._device_pref != "auto":
            return torch.device(self._device_pref)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _is_ready(self):
        return (
            self._model is not None
            and self._vocoder is not None
            and self._accentizer is not None
        )

    def _ensure_loaded(self):
        with self._lock:
            if self._is_ready():
                return

            try:
                from huggingface_hub import hf_hub_download
                from ruaccent import RUAccent
                from f5_tts.infer.utils_infer import load_model, load_vocoder
                from f5_tts.model import DiT
            except ImportError as e:
                raise RuntimeError(
                    "ESpeech dependencies missing. Install with: "
                    "pip install f5-tts ruaccent huggingface_hub"
                ) from e

            # CURSOR: Build into locals first; only publish on full success so a
            # failed RUAccent load cannot leave model set and accentizer None.
            print(f"ESpeech: downloading {MODEL_REPO} weights...")
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
            vocab_path = hf_hub_download(repo_id=MODEL_REPO, filename=VOCAB_FILE)

            device = self._resolve_device()
            print(f"ESpeech: loading model on {device}...")
            model = load_model(DiT, MODEL_CFG, model_path, vocab_file=vocab_path)
            model.to(device)

            print("ESpeech: loading vocoder...")
            vocoder = load_vocoder()
            vocoder.to(device)

            print("ESpeech: loading RUAccent...")
            accentizer = RUAccent()
            accentizer.load(
                omograph_model_size="turbo3.1",
                use_dictionary=True,
                tiny_mode=False,
            )
            probe = accentizer.process_all("тест")
            if not probe:
                raise RuntimeError("RUAccent.load succeeded but process_all returned empty")

            self._device = device
            self._model = model
            self._vocoder = vocoder
            self._accentizer = accentizer
            print("ESpeech: ready.")

    def _accent(self, text):
        if not text or not text.strip():
            return text
        # CURSOR: Skip auto-accent when user already marked stress with '+'
        if "+" in text:
            return text
        if self._accentizer is None:
            raise RuntimeError(
                "RUAccent is not loaded. ESpeech init did not finish; check logs "
                "for 'ESpeech: loading RUAccent' / 'ESpeech: ready'."
            )
        return self._accentizer.process_all(text)

    def synthesize(
        self,
        text,
        ref_audio_path,
        ref_text,
        output_path,
        seed=-1,
        speed=1.0,
        nfe_step=32,
        cross_fade_duration=0.15,
    ):
        """Clone voice from ref audio/text and write WAV to output_path.

        Returns True on success.
        """
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"ESpeech reference audio not found: {ref_audio_path}")
        if not ref_text or not str(ref_text).strip():
            raise ValueError("ESpeech requires ref_text matching the reference audio")
        if not text or not str(text).strip():
            raise ValueError("ESpeech generation text is empty")

        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        with self._lock:
            self._ensure_loaded()

            if seed is None or int(seed) < 0:
                seed = int(np.random.randint(0, 2**31 - 1))
            torch.manual_seed(int(seed))

            processed_ref_text = self._accent(ref_text)
            processed_gen_text = self._accent(text)

            ref_audio_proc, processed_ref_text_final = preprocess_ref_audio_text(
                ref_audio_path,
                processed_ref_text,
                show_info=print,
            )

            inference = infer_process(
                ref_audio_proc,
                processed_ref_text_final,
                processed_gen_text,
                self._model,
                self._vocoder,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                show_info=print,
            )

            # F5-TTS releases differ here: older versions return the result tuple
            # directly, while current versions yield it from an iterator.
            if not isinstance(inference, tuple):
                inference = next(inference, None)
            if inference is None:
                return False

            wave, sample_rate, _ = inference

        if wave is None:
            return False

        audio = np.asarray(wave)
        if audio.ndim > 1:
            audio = audio.flatten()
        sf.write(output_path, audio, sample_rate)
        return True

    def synthesize_batch(
        self,
        texts,
        ref_audio_path,
        ref_text,
        output_paths,
        seed=-1,
        plan_sub_batches=None,
        speed=1.0,
        nfe_step=32,
        cross_fade_duration=0.15,
    ):
        """Generate several texts with one reference voice in batched model calls.

        Mirrors f5_tts.infer.utils_infer.infer_process per text (reference
        preparation, chunk_text splitting, duration estimate, cross-fade join),
        but sends the resulting pieces through CFM.sample as batches instead of
        one call per piece.

        Args:
            plan_sub_batches: callable(piece_texts) -> list of (start, end)
                splitting length-sorted pieces into model calls. None sends all
                pieces in one call.

        Returns a list of booleans, one per text: True when its WAV was written.
        """
        if len(texts) != len(output_paths):
            raise ValueError("texts and output_paths must have the same length")
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"ESpeech reference audio not found: {ref_audio_path}")
        if not ref_text or not str(ref_text).strip():
            raise ValueError("ESpeech requires ref_text matching the reference audio")

        import torchaudio
        from f5_tts.infer.utils_infer import (
            cfg_strength,
            chunk_text,
            hop_length,
            preprocess_ref_audio_text,
            sway_sampling_coef,
            target_rms,
            target_sample_rate,
        )
        from f5_tts.model.utils import convert_char_to_pinyin

        with self._lock:
            self._ensure_loaded()

            if seed is None or int(seed) < 0:
                seed = int(np.random.randint(0, 2**31 - 1))

            ref_audio_proc, ref_text_final = preprocess_ref_audio_text(
                ref_audio_path,
                self._accent(ref_text),
                show_info=print,
            )

            audio, sr = torchaudio.load(ref_audio_proc)
            ref_seconds = audio.shape[-1] / sr
            max_chars = int(len(ref_text_final.encode("utf-8")) / ref_seconds * (22 - ref_seconds) * speed)

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            if sr != target_sample_rate:
                audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
            audio = audio.to(self._device)

            if len(ref_text_final[-1].encode("utf-8")) == 1:
                ref_text_final = ref_text_final + " "
            ref_audio_len = audio.shape[-1] // hop_length
            ref_text_len = len(ref_text_final.encode("utf-8"))

            # (text index, piece index, piece text, duration in mel frames)
            pieces = []
            piece_counts = []
            for text_idx, text in enumerate(texts):
                text_pieces = chunk_text(self._accent(text), max_chars=max_chars) if text and text.strip() else []
                piece_counts.append(len(text_pieces))
                for piece_idx, piece in enumerate(text_pieces):
                    local_speed = 0.3 if len(piece.encode("utf-8")) < 10 else speed
                    duration = ref_audio_len + int(
                        ref_audio_len / ref_text_len * len(piece.encode("utf-8")) / local_speed
                    )
                    pieces.append((text_idx, piece_idx, piece, duration))

            # CLAUDE CODE: Similar lengths share a call so little compute is
            # spent on padding.
            pieces.sort(key=lambda p: p[3])
            piece_texts = [p[2] for p in pieces]
            sub_batches = plan_sub_batches(piece_texts) if plan_sub_batches else [(0, len(pieces))]

            waves = {}
            for start, end in sub_batches:
                batch = pieces[start:end]
                if not batch:
                    continue
                final_text_list = convert_char_to_pinyin([ref_text_final + p[2] for p in batch])
                # CLAUDE CODE: CFM.sample raises a duration to at least
                # max(text tokens, reference frames) + 1. Apply the same rule
                # here so each output is trimmed at the length actually generated.
                durations = [
                    max(max(len(tokens), ref_audio_len) + 1, p[3])
                    for tokens, p in zip(final_text_list, batch)
                ]
                with torch.inference_mode():
                    generated, _ = self._model.sample(
                        cond=audio.repeat(len(batch), 1),
                        text=final_text_list,
                        duration=torch.tensor(durations, dtype=torch.long, device=self._device),
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        seed=int(seed),
                    )
                    for row, (p, duration) in enumerate(zip(batch, durations)):
                        mel = generated[row, ref_audio_len:duration, :].unsqueeze(0)
                        mel = mel.permute(0, 2, 1).to(torch.float32)
                        wave = self._vocoder.decode(mel)
                        if rms < target_rms:
                            wave = wave * rms / target_rms
                        waves[(p[0], p[1])] = wave.squeeze().cpu().numpy()
                del generated

        results = []
        for text_idx, output_path in enumerate(output_paths):
            count = piece_counts[text_idx]
            if count == 0:
                results.append(False)
                continue
            audio_out = self._cross_fade_join(
                [waves[(text_idx, i)] for i in range(count)],
                target_sample_rate,
                cross_fade_duration,
            )
            sf.write(output_path, audio_out, target_sample_rate)
            results.append(True)
        return results

    @staticmethod
    def _cross_fade_join(waves, sample_rate, cross_fade_duration):
        """Join consecutive pieces with a linear cross-fade (same as F5's infer_batch_process)."""
        final_wave = waves[0]
        for next_wave in waves[1:]:
            fade_samples = min(int(cross_fade_duration * sample_rate), len(final_wave), len(next_wave))
            if fade_samples <= 0:
                final_wave = np.concatenate([final_wave, next_wave])
                continue
            overlap = (
                final_wave[-fade_samples:] * np.linspace(1, 0, fade_samples)
                + next_wave[:fade_samples] * np.linspace(0, 1, fade_samples)
            )
            final_wave = np.concatenate([final_wave[:-fade_samples], overlap, next_wave[fade_samples:]])
        return final_wave

    def unload(self):
        """Release model weights and free GPU memory."""
        with self._lock:
            self._model = None
            self._vocoder = None
            self._accentizer = None
            self._device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
