from django.contrib import admin
from django.urls import path
from ai_engine import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'), # This is the Creators/Audio page
    path('teacher/', views.teacher_grader, name='teacher_grader'), # This is the new Teacher page
    path('student/', views.student_mindmap, name='student_mindmap'),
    path('tutor/', views.student_tutor, name='student_tutor'),


]
