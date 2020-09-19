from django.shortcuts import render
from .models import Meme
from .serializers import LeadSerializer
from rest_framework import generics

class LeadListCreate(generics.ListCreateAPIView):
    queryset = Meme.objects.all()
    serializer_class = LeadSerializer
