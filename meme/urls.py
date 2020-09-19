from django.urls import path
from . import views

urlpatterns = [
    path('meme/api/',views.LeadListCreate.as_view()),
]
