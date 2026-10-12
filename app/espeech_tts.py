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
