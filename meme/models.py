from django.db import models

# Create your models here.
class Meme(models.Model):
    description = models.TextField()
    img_url = models.URLField()
