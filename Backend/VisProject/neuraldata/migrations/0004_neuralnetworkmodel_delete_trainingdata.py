# Generated by Django 4.2.7 on 2023-12-05 20:19

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neuraldata", "0003_trainingdata_layer_order"),
    ]

    operations = [
        migrations.CreateModel(
            name="NeuralNetwork",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("state_dict", models.BinaryField()),
                ("activations", models.JSONField(default=list)),
                ("activations_shape", models.JSONField(default=list)),
                ("gradients", models.JSONField(default=list)),
                ("gradients_shape", models.JSONField(default=list)),
                ("loss", models.FloatField(blank=True, default=None, null=True)),
                ("layer_order", models.JSONField(default=list)),
            ],
        ),
        migrations.DeleteModel(
            name="TrainingData",
        ),
    ]