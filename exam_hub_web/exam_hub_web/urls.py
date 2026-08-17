from django.contrib import admin
from django.urls import path
from ai_engine import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'), # This connects the homepage to our AI view
]