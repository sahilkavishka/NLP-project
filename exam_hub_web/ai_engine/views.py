import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Importing our custom AI modules from the core folder
from core.audio_processor import AudioTranscriber
from core.rag_engine import LectureRAG
from core.seo_generator import SEOGenerator

def index(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        
        # Save the uploaded file temporarily in the 'data' folder
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        fs = FileSystemStorage(location=data_dir)
        filename = fs.save('web_upload.wav', audio_file)
        file_path = os.path.join(data_dir, filename)
        
        try:
            print("[WEB] Starting AI Processing Pipeline...")
            
            # STEP 1: Transcription
            transcriber = AudioTranscriber()
            transcript = transcriber.transcribe(file_path)
            
            # STEP 2: RAG Engine
            db_path = os.path.join(data_dir, "chroma_db_web")
            rag = LectureRAG(db_dir=db_path)
            rag.build_knowledge_base(transcript)
            qa_answer = rag.ask_question("What is this lecture about?")
            
            # STEP 3: SEO Generation
            seo_gen = SEOGenerator()
            seo_data = seo_gen.generate_all(transcript)
            
            # Pass all results to the HTML template
            context['transcript'] = transcript
            context['qa_answer'] = qa_answer
            context['seo_data'] = seo_data
            
            print("[WEB] Processing Complete!")
            
        except Exception as e:
            context['error'] = str(e)
            print(f"[WEB Error] {str(e)}")
            
        finally:
            # Clean up the uploaded audio file to save space
            if os.path.exists(file_path):
                os.remove(file_path)
                
    return render(request, 'ai_engine/index.html', context)