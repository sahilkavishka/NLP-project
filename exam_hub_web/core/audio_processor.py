"""
Advanced Audio Processor v2
============================
Transcription (Whisper) + sentiment + optional summarization.

New in this version:
  - CLI (run `python audio_processor.py --help`)
  - float16 / SDPA attention on GPU for faster inference
  - Context-manager support (`with AdvancedAudioProcessor() as p:`)
  - Retry-with-backoff around model calls (handles transient OOM/network hiccups)
  - Export to TXT / JSON / SRT / VTT
  - tqdm progress bar for batch jobs
  - Config validation (fails fast on bad values instead of failing mid-run)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import librosa
from transformers import pipeline

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op iterator
    def tqdm(iterable, **kwargs):
        return iterable

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AudioProcessor")

# Set your HuggingFace token via environment variable, not in code:
#   export HF_TOKEN="hf_xxx"
os.environ.setdefault("HF_TOKEN", "")


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class AudioProcessorConfig:
    asr_model: str = "openai/whisper-large-v3-turbo"
    sentiment_model: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    summarization_model: Optional[str] = "facebook/bart-large-cnn"  # None disables summaries
    chunk_length_s: int = 30
    stride_length_s: int = 5
    sample_rate: int = 16000
    sentiment_char_limit: int = 2000
    min_words_for_sentiment: int = 10
    min_words_for_summary: int = 40
    device: Optional[str] = None          # auto-detected if None
    use_fp16: bool = True                 # half precision on GPU (ignored on CPU)
    max_retries: int = 2                  # retries for transient model-call failures
    retry_backoff_s: float = 2.0

    def __post_init__(self):
        if self.chunk_length_s <= 0:
            raise ValueError("chunk_length_s must be positive")
        if self.stride_length_s < 0 or self.stride_length_s >= self.chunk_length_s:
            raise ValueError("stride_length_s must be >= 0 and < chunk_length_s")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")


# ------------------------------------------------------------------
# Small retry helper (avoids pulling in an extra dependency like tenacity)
# ------------------------------------------------------------------

def _with_retries(fn, max_retries: int, backoff_s: float, label: str):
    last_exc = None
    for attempt in range(1, max_retries + 2):  # first try + retries
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt <= max_retries:
                logger.warning(
                    f"{label} failed (attempt {attempt}/{max_retries + 1}): {e}. "
                    f"Retrying in {backoff_s:.1f}s..."
                )
                time.sleep(backoff_s)
            else:
                logger.error(f"{label} failed after {attempt} attempts: {e}")
    raise last_exc


# ------------------------------------------------------------------
# Core class
# ------------------------------------------------------------------

class AdvancedAudioProcessor:
    """
    Loads ASR, sentiment, and (optionally) summarization pipelines once,
    and reuses them across multiple audio files. Usable as a context manager:

        with AdvancedAudioProcessor() as proc:
            result = proc.process_audio("call.wav")
    """

    def __init__(self, config: Optional[AudioProcessorConfig] = None):
        self.config = config or AudioProcessorConfig()
        self.device = self.config.device or self._detect_device()
        self.device_index = 0 if self.device == "cuda" else -1
        self.torch_dtype = (
            torch.float16 if (self.device == "cuda" and self.config.use_fp16) else torch.float32
        )

        logger.info(f"Using device: {self.device} (dtype={self.torch_dtype})")

        self.transcriber = self._load_pipeline(
            "automatic-speech-recognition",
            self.config.asr_model,
            extra_kwargs=dict(
                chunk_length_s=self.config.chunk_length_s,
                stride_length_s=self.config.stride_length_s,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
            ),
            try_sdpa=True,
        )

        self.sentiment_analyzer = self._load_pipeline(
            "sentiment-analysis", self.config.sentiment_model
        )

        self.summarizer = None
        if self.config.summarization_model:
            try:
                self.summarizer = self._load_pipeline(
                    "summarization", self.config.summarization_model
                )
            except Exception as e:
                logger.warning(f"Summarizer failed to load, disabling summaries: {e}")
                self.summarizer = None

        logger.info("All models loaded successfully.")

    # ---------------- context manager ----------------

    def __enter__(self) -> "AdvancedAudioProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"AdvancedAudioProcessor(asr={self.config.asr_model}, "
            f"device={self.device}, summarizer={'on' if self.summarizer else 'off'})"
        )

    # ---------------- internal helpers ----------------

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_pipeline(
        self,
        task: str,
        model_name: str,
        extra_kwargs: Optional[dict] = None,
        try_sdpa: bool = False,
    ):
        logger.info(f"Loading {task} model: {model_name}")
        kwargs = dict(extra_kwargs or {})
        device_arg = self.device if self.device == "mps" else self.device_index

        # Try scaled-dot-product-attention for a speed boost on GPU; silently
        # fall back to default attention if the model/transformers version
        # doesn't support it.
        if try_sdpa and self.device == "cuda":
            try:
                return pipeline(
                    task,
                    model=model_name,
                    device=device_arg,
                    model_kwargs={"attn_implementation": "sdpa"},
                    **kwargs,
                )
            except Exception as e:
                logger.info(f"SDPA attention unavailable ({e}), using default attention.")

        try:
            return pipeline(task, model=model_name, device=device_arg, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load {task} pipeline ({model_name}): {e}")
            raise

    def _load_audio(self, file_path: str) -> tuple:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {file_path}")

        logger.info(f"Reading and normalizing audio: {file_path}")
        try:
            audio_data, sample_rate = librosa.load(
                file_path, sr=self.config.sample_rate, mono=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio file '{file_path}': {e}") from e

        if audio_data.size == 0:
            raise ValueError(f"Decoded audio is empty: {file_path}")

        duration_s = len(audio_data) / sample_rate
        logger.info(f"Audio duration: {duration_s:.1f}s")
        return audio_data, sample_rate

    def _transcribe(self, audio_data, sample_rate, language, task) -> Dict[str, Any]:
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task

        def _run():
            with torch.inference_mode():
                return self.transcriber(
                    {"array": audio_data, "sampling_rate": sample_rate},
                    generate_kwargs=generate_kwargs,
                )

        return _with_retries(
            _run, self.config.max_retries, self.config.retry_backoff_s, "Transcription"
        )

    def _analyze_sentiment(self, transcript: str) -> str:
        if len(transcript.split()) < self.config.min_words_for_sentiment:
            return "Not enough text for sentiment analysis"
        try:
            snippet = transcript[: self.config.sentiment_char_limit]
            with torch.inference_mode():
                result = self.sentiment_analyzer(snippet)[0]
            return f"{result['label']} (Confidence: {round(result['score'] * 100, 2)}%)"
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return "Analysis Failed"

    def _summarize(self, transcript: str) -> str:
        if not self.summarizer:
            return "Summarization is disabled."
        if len(transcript.split()) < self.config.min_words_for_summary:
            return "Transcript too short to summarize."
        try:
            with torch.inference_mode():
                result = self.summarizer(
                    transcript[:4000], max_length=130, min_length=30, do_sample=False
                )
            return result[0]["summary_text"].strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return "Summarization Failed"

    # ---------------- public API ----------------

    def process_audio(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        summarize: bool = True,
    ) -> Dict[str, Any]:
        audio_data, sample_rate = self._load_audio(file_path)

        logger.info(f"Transcribing (task={task}, lang={language or 'auto'})...")
        result = self._transcribe(audio_data, sample_rate, language, task)
        transcript = result.get("text", "").strip()

        if not transcript:
            logger.warning("Transcription produced empty text.")
            return {
                "file": file_path,
                "transcript": "",
                "summary": "N/A (empty transcript)",
                "sentiment": "N/A (empty transcript)",
                "timestamps": result.get("chunks", []),
            }

        logger.info("Transcription complete. Running sentiment analysis...")
        sentiment = self._analyze_sentiment(transcript)
        summary = self._summarize(transcript) if summarize else "Summarization skipped."

        return {
            "file": file_path,
            "transcript": transcript,
            "summary": summary,
            "sentiment": sentiment,
            "timestamps": result.get("chunks", []),
        }

    def process_batch(
        self,
        file_paths: List[str],
        language: Optional[str] = None,
        task: str = "transcribe",
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple files, continuing past individual failures."""
        results = []
        iterator = tqdm(file_paths, desc="Processing audio") if show_progress else file_paths
        for fp in iterator:
            try:
                results.append(self.process_audio(fp, language=language, task=task))
            except Exception as e:
                logger.error(f"Skipping '{fp}' due to error: {e}")
                results.append({"file": fp, "error": str(e)})
        return results

    # ---------------- export helpers ----------------

    def save_json(self, results: Dict[str, Any], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON saved to {output_path}")

    def save_txt(self, results: Dict[str, Any], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(results.get("transcript", ""))
        logger.info(f"Transcript saved to {output_path}")

    def save_srt(self, results: Dict[str, Any], output_path: str) -> None:
        chunks = results.get("timestamps", [])
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, start=1):
                start, end = chunk.get("timestamp", (None, None))
                if start is None or end is None:
                    continue
                f.write(f"{i}\n")
                f.write(f"{self._srt_time(start)} --> {self._srt_time(end)}\n")
                f.write(f"{chunk.get('text', '').strip()}\n\n")
        logger.info(f"SRT saved to {output_path}")

    def save_vtt(self, results: Dict[str, Any], output_path: str) -> None:
        chunks = results.get("timestamps", [])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for chunk in chunks:
                start, end = chunk.get("timestamp", (None, None))
                if start is None or end is None:
                    continue
                f.write(f"{self._vtt_time(start)} --> {self._vtt_time(end)}\n")
                f.write(f"{chunk.get('text', '').strip()}\n\n")
        logger.info(f"VTT saved to {output_path}")

    @staticmethod
    def _srt_time(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _vtt_time(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def cleanup(self) -> None:
        """Free GPU memory when done with this processor."""
        for attr in ("transcriber", "sentiment_analyzer", "summarizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Resources released.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe, analyze sentiment, and summarize audio files."
    )
    parser.add_argument("files", nargs="+", help="One or more audio file paths")
    parser.add_argument("--language", default=None, help="Force source language, e.g. 'en'")
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"],
        help="Whisper task (translate = translate to English)",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip summarization")
    parser.add_argument(
        "--format", default="json", choices=["json", "txt", "srt", "vtt"],
        help="Output format for saved results",
    )
    parser.add_argument("--outdir", default=".", help="Directory to write output files to")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 even on GPU")
    parser.add_argument("--quiet", action="store_true", help="Only log warnings and errors")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("ignore")
    args = _build_arg_parser().parse_args(argv)

    if args.quiet:
        logger.setLevel(logging.WARNING)

    config = AudioProcessorConfig(asr_model=args.asr_model, use_fp16=not args.no_fp16)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    with AdvancedAudioProcessor(config) as processor:
        for file_path in args.files:
            try:
                result = processor.process_audio(
                    file_path,
                    language=args.language,
                    task=args.task,
                    summarize=not args.no_summary,
                )
            except Exception as e:
                logger.error(f"Failed to process '{file_path}': {e}")
                exit_code = 1
                continue

            stem = Path(file_path).stem
            out_path = outdir / f"{stem}.{args.format}"
            save_fn = {
                "json": processor.save_json,
                "txt": processor.save_txt,
                "srt": processor.save_srt,
                "vtt": processor.save_vtt,
            }[args.format]
            save_fn(result, str(out_path))

            print(f"\n=== {file_path} ===")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Summary:   {result['summary']}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())