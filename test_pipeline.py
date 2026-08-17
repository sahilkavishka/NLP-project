import os
import warnings
warnings.filterwarnings("ignore")

# Importing our custom modules from the 'core' package
from core.audio_processor import AudioTranscriber
from core.rag_engine import LectureRAG
from core.seo_generator import SEOGenerator

def run_exam_hub_pipeline(audio_file_path):
    print("==================================================")
    print("🚀 INITIALIZING EXAM HUB AI PIPELINE")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # STEP 1: Transcription
    # ---------------------------------------------------------
    print(">>> [STEP 1] Starting Audio Transcription...")
    try:
        transcriber = AudioTranscriber()
        transcript = transcriber.transcribe(audio_file_path)
        print("\n--- TRANSCRIPT ---")
        print(transcript)
        print("------------------\n")
    except Exception as e:
        print(f"[Error] Transcription failed: {e}")
        return

    # ---------------------------------------------------------
    # STEP 2: RAG Engine (Knowledge Base & QA)
    # ---------------------------------------------------------
    print(">>> [STEP 2] Processing Knowledge Base & RAG...")
    try:
        rag = LectureRAG(db_dir="data/chroma_db_final")
        rag.build_knowledge_base(transcript)
        
        # Simulating a student asking a question based on the lecture
        test_question = "What topics are discussed today for the 2026 syllabus?"
        print(f"\n[Student Question]: {test_question}")
        
        answer = rag.ask_question(test_question)
        print(f"[AI Answer]: {answer}\n")
    except Exception as e:
        print(f"[Error] RAG Engine failed: {e}")
        return

    # ---------------------------------------------------------
    # STEP 3: SEO & Social Media Generation
    # ---------------------------------------------------------
    print(">>> [STEP 3] Generating SEO & Social Media Content...")
    try:
        seo_gen = SEOGenerator()
        seo_results = seo_gen.generate_all(transcript)
    except Exception as e:
        print(f"[Error] SEO Generation failed: {e}")
        return

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n==================================================")
    print("✅ PIPELINE EXECUTION SUCCESSFUL! FINAL OUTPUTS:")
    print("==================================================")
    print(f"📌 YouTube Title : {seo_results['youtube_title']}")
    print(f"📌 SEO Tags      : {seo_results['seo_tags']}")
    print(f"📌 Facebook Post : {seo_results['facebook_post']}")
    print("==================================================\n")

if __name__ == "__main__":
    # Define the path to the test audio file we created earlier
    sample_audio = "data/lecture_sample.wav"
    
    # Check if the audio file exists before running
    if not os.path.exists(sample_audio):
        print(f"Error: '{sample_audio}' not found. Please run 'create_test_audio.py' first.")
    else:
        run_exam_hub_pipeline(sample_audio)