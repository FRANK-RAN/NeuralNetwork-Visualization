from django.db import models

class NeuralNetwork(models.Model):
    name = models.CharField(max_length=100)
    state_dict = models.BinaryField()
    activations = models.JSONField(default=list)
    activations_shape = models.JSONField(default=list)
    gradients = models.JSONField(default=list)
    gradients_shape = models.JSONField(default=list)
    loss = models.FloatField(null=True, blank=True, default=None)  # Loss is a single float value
    layer_order = models.JSONField(default=list)
