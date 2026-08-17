import logging
import os
import threading

from celery import shared_task

from .audio_processor import AdvancedAudioProcessor  # your AdvancedAudioProcessor module
from .rag import LectureRAG
from .seo import SEOGenerator

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy, thread-safe singleton.
#
# Loading Whisper + sentiment + summarization models is expensive
# (multi-GB downloads, seconds-to-minutes of load time, real GPU/CPU
# memory). We do NOT want that happening at Django import time (it
# would block `manage.py migrate`, `manage.py shell`, test collection,
# etc.) and we do NOT want a race where two concurrent requests each
# try to load their own copy of the models.
# ------------------------------------------------------------------

_processor = None
_processor_lock = threading.Lock()


def get_processor() -> AdvancedAudioProcessor:
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:  # re-check inside the lock
                logger.info("Lazily initializing AdvancedAudioProcessor...")
                _processor = AdvancedAudioProcessor()
    return _processor


@shared_task(bind=True, max_retries=1, default_retry_delay=30)
def process_lecture_task(self, temp_file_path: str, language: str | None, task_type: str):
    """
    Runs transcription -> RAG indexing -> SEO generation for one uploaded file.
    Always deletes the temp file, whether it succeeds or fails.
    """
    try:
        logger.info(f"[task {self.request.id}] Starting pipeline for {temp_file_path}")

        processor = get_processor()
        audio_data = processor.process_audio(
            file_path=temp_file_path, language=language, task=task_type
        )
        transcript = audio_data["transcript"]

        if not transcript:
            return {
                "success": False,
                "error": "No speech was detected in the uploaded file.",
            }

        # Give each task run its own RAG collection so concurrent uploads
        # never write into the same vector store.
        from django.conf import settings

        db_path = os.path.join(settings.RAG_DB_ROOT, self.request.id)
        rag = LectureRAG(db_dir=db_path)
        rag.build_knowledge_base(transcript)
        qa_answer = rag.ask_question("What is the main topic discussed here?")

        seo_data = SEOGenerator().generate_all(transcript)

        logger.info(f"[task {self.request.id}] Pipeline complete.")
        return {
            "success": True,
            "transcript": transcript,
            "summary": audio_data["summary"],
            "sentiment": audio_data["sentiment"],
            "qa_answer": qa_answer,
            "seo_data": seo_data,
        }

    except Exception as e:
        logger.exception(f"[task {self.request.id}] Pipeline failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")