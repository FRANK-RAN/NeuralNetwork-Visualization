from rest_framework import serializers
from .models import Activation, Gradient

class ActivationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Activation
        fields = '__all__'

class GradientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Gradient
        fields = '__all__'