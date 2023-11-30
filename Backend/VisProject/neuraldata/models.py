from django.db import models

# Create your models here.
class Activation(models.Model):
    data = models.JSONField()

class Gradient(models.Model):
    data = models.JSONField()
