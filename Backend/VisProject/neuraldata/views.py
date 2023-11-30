from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import viewsets
from .models import Activation, Gradient
from .serializers import ActivationSerializer, GradientSerializer

class ActivationViewSet(viewsets.ModelViewSet):
    queryset = Activation.objects.all()
    serializer_class = ActivationSerializer

    def create(self, request, *args, **kwargs):
        Activation.objects.all().delete()  # Delete existing records
        serializer = self.get_serializer(data=request.data)  # Create a new record
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

class GradientViewSet(viewsets.ModelViewSet):
    queryset = Gradient.objects.all()
    serializer_class = GradientSerializer

    def create(self, request, *args, **kwargs):
        Gradient.objects.all().delete()  # Delete existing records
        serializer = self.get_serializer(data=request.data)  # Create a new record
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)