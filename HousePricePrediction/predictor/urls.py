"""
predictor/urls.py
URL patterns for the predictor app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('',            views.home,      name='home'),
    path('predict/',    views.predict,   name='predict'),
    path('result/<int:pk>/', views.result, name='result'),
    path('dashboard/',  views.dashboard, name='dashboard'),
    path('history/',    views.history,   name='history'),
    path('about/',      views.about,     name='about'),
]
