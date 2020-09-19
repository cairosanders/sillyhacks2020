from rest_framework import serializers
from .models import Meme

class LeadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Meme
        fields = ('description','img_url')
