import os
import tempfile
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import traceback
# Importing our custom AI modules from the core folder
from core.audio_processor import AdvancedAudioProcessor
from core.rag_engine import LectureRAG
from core.seo_generator import SEOGenerator
from core.essay_grader import EssayGrader
from core.mind_mapper import MindMapper
from core.socratic_tutor import SocraticTutor
from core.singlish_analyzer import SinglishAnalyzer
import json
from core.trend_predictor import TrendPredictor
from core.gap_analyzer import ContentGapAnalyzer
from core.predictive_paper_gen import PredictivePaperGenerator

def home(request):
    return render(request, 'ai_engine/home.html')

from celery.result import AsyncResult
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, Http404
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
 
from core.forms import AudioUploadForm
from ..core.tasks import process_lecture_task
 
logger = logging.getLogger(__name__)
 
 
@login_required
@require_http_methods(["GET", "POST"])
def index(request):
    """
    GET: render the upload form.
    POST: validate + save the upload to a secure temp file, enqueue the
          AI pipeline as a background Celery task, and return the task id
          so the page can poll for the result. The heavy work never runs
          inside this request/response cycle, so it can't hit a web-server
          timeout no matter how long the audio is.
    """
    if request.method == "GET":
        return render(request, "ai_engine/audio_upload.html", {})
 
    form = AudioUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        # Surface the first validation error; keep the rest for debugging/logs.
        first_error = next(iter(form.errors.values()))[0]
        logger.info(f"Upload rejected for user {request.user.id}: {form.errors.as_json()}")
        return render(request, "ai_engine/audio_upload.html", {"error": first_error})
 
    audio_file = form.cleaned_data["audio_file"]
    language = form.cleaned_data["language"]
    task_type = form.cleaned_data["task"]
 
    ext = os.path.splitext(audio_file.name)[1].lower()
    os.makedirs(settings.AUDIO_UPLOAD_TMP_DIR, exist_ok=True)
 
    fd, temp_path = tempfile.mkstemp(suffix=ext, dir=settings.AUDIO_UPLOAD_TMP_DIR)
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
    except Exception:
        os.remove(temp_path)
        raise
 
    task = process_lecture_task.delay(temp_path, language, task_type)
    logger.info(f"Enqueued task {task.id} for user {request.user.id} ({audio_file.name})")
 
    return JsonResponse({"task_id": task.id})
 
 
@login_required
@require_http_methods(["GET"])
def task_status(request, task_id):
    """Polled by the frontend while the AI pipeline runs in the background."""
    result = AsyncResult(task_id)
 
    if result.state == "PENDING":
        return JsonResponse({"state": "PENDING"})
    if result.state in ("STARTED", "RETRY"):
        return JsonResponse({"state": "RUNNING"})
    if result.state == "FAILURE":
        return JsonResponse({"state": "FAILURE", "error": "Processing failed unexpectedly."})
    if result.state == "SUCCESS":
        payload = result.result or {}
        if not payload.get("success"):
            return JsonResponse({"state": "FAILURE", "error": payload.get("error", "Unknown error")})
        return JsonResponse({"state": "SUCCESS", "result": payload})
 
    return JsonResponse({"state": result.state})
 

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

def singlish_analyzer(request):
    context = {}
    if request.method == 'POST':
        query = request.POST.get('singlish_query')
        try:
            analyzer = SinglishAnalyzer()
            # Pass the singlish query to our new NLP module
            result = analyzer.analyze_and_answer(query)
            context['result'] = result
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'ai_engine/singlish_analyzer.html', context)


def trend_dashboard(request):
    try:
        predictor = TrendPredictor()
        trends_data = predictor.predict_2026_trends()
        
        # We pass both the raw dictionary (for cards) and JSON string (for Chart.js)
        context = {
            'trends_data': trends_data,
            'trends_json': trends_data
        }
    except Exception as e:
        context = {'error': str(e)}
        
    return render(request, 'ai_engine/trend_dashboard.html', context)


def gap_analyzer(request):
    context = {}
    if request.method == 'POST':
        syllabus = request.POST.get('syllabus')
        existing = request.POST.get('existing')
        
        try:
            analyzer = ContentGapAnalyzer()
            result = analyzer.analyze_gap(syllabus, existing)
            context['result'] = result
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'ai_engine/gap_analyzer.html', context)

def predictive_paper(request):
    context = {}
    if request.method == 'POST':
        subject = request.POST.get('subject_name')
        context['subject_name'] = subject
        try:
            generator = PredictivePaperGenerator()
            # Generate the predicted paper for the upcoming exam
            paper = generator.analyze_and_generate(subject)
            context['mock_paper'] = paper
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'ai_engine/predictive_paper.html', context)