import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Importing our custom AI modules from the core folder
from core.audio_processor import AudioTranscriber
from core.rag_engine import LectureRAG
from core.seo_generator import SEOGenerator
from core.essay_grader import EssayGrader
from core.mind_mapper import MindMapper
from core.socratic_tutor import SocraticTutor

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

def teacher_grader(request):
    context = {}
    if request.method == 'POST':
        student_ans = request.POST.get('student_answer')
        scheme_raw = request.POST.get('marking_scheme')
        
        # Split the marking scheme text area into a list of points
        scheme_points = [point.strip() for point in scheme_raw.split('\n') if point.strip()]
        
        try:
            grader = EssayGrader()
            result = grader.grade_answer(student_ans, scheme_points)
            context['result'] = result
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'ai_engine/essay_grader.html', context)

def student_mindmap(request):
    context = {}
    if request.method == 'POST':
        notes = request.POST.get('study_notes')
        try:
            mapper = MindMapper()
            # Generate the graph code
            mermaid_syntax = mapper.generate_mermaid_syntax(notes)
            context['mermaid_code'] = mermaid_syntax
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'ai_engine/student_mindmap.html', context)

def student_tutor(request):
    context = {}
    if request.method == 'POST':
        question = request.POST.get('student_question')
        context['student_question'] = question
        try:
            tutor = SocraticTutor()
            # Get the Socratic hint
            response = tutor.get_guidance(question)
            context['tutor_response'] = response
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'ai_engine/student_tutor.html', context)