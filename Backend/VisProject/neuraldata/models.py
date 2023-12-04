from django.db import models

class TrainingData(models.Model):
    activations = models.JSONField()
    activations_shape = models.JSONField()
    gradients = models.JSONField()
    gradients_shape = models.JSONField()
    loss = models.FloatField()  # Loss is a single float value
    layer_order = models.JSONField(default=list)

