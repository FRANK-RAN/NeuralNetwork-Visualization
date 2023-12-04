from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import viewsets
from .models import TrainingData


from rest_framework import viewsets
from .models import TrainingData
from .serializers import TrainingDataSerializer

class TrainingDataViewSet(viewsets.ModelViewSet):
    queryset = TrainingData.objects.all()
    serializer_class = TrainingDataSerializer

    def get_queryset(self):
        # Order by 'id' in descending order and return only the first record
        return TrainingData.objects.all().order_by('-id')[:1]

    def create(self, request, *args, **kwargs):
        # Create a new record with the combined training data
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)
