from django.contrib import admin
from django.urls import path
from ai_engine import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),                     # The New Dashboard Portal
    path('audio/', views.index, name='audio_upload'),      # Moved to /audio/
    path('teacher/', views.teacher_grader, name='teacher_grader'),
    path('student/', views.student_mindmap, name='student_mindmap'),
    path('tutor/', views.student_tutor, name='student_tutor'),
    path('singlish/', views.singlish_analyzer, name='singlish_analyzer'),
    path('dashboard/', views.trend_dashboard, name='trend_dashboard'),
    path('gap/', views.gap_analyzer, name='gap_analyzer'),
    path('predictive-paper/', views.predictive_paper, name='predictive_paper'),
    path("status/<str:task_id>/", views.task_status, name="task_status"),
]
