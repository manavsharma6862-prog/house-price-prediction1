"""
predictor/models.py
Database models for storing prediction history.
"""
from django.db import models
from django.utils import timezone


class PredictionRecord(models.Model):
    median_income       = models.FloatField()
    house_age           = models.FloatField()
    avg_rooms           = models.FloatField()
    avg_bedrooms        = models.FloatField()
    population          = models.FloatField()
    avg_occupancy       = models.FloatField()
    latitude            = models.FloatField()
    longitude           = models.FloatField()

    predicted_price     = models.FloatField()
    predicted_price_usd = models.FloatField()
    confidence_level    = models.CharField(max_length=50)
    model_used          = models.CharField(max_length=100, default='')

    created_at          = models.DateTimeField(default=timezone.now)
    ip_address          = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction Record'
        verbose_name_plural = 'Prediction Records'

    def __str__(self):
        return (f"Prediction #{self.pk} — "
                f"${self.predicted_price_usd:,.0f} "
                f"({self.created_at.strftime('%Y-%m-%d %H:%M')})")

    @property
    def confidence_badge_class(self):
        m = {'High Confidence': 'success', 'Medium Confidence': 'warning',
             'Low Confidence': 'danger'}
        return m.get(self.confidence_level, 'secondary')
