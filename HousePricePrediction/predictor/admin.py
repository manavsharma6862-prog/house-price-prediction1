"""
predictor/admin.py
Django admin configuration for PredictionRecord model.
"""
from django.contrib import admin
from .models import PredictionRecord


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'predicted_price_usd_display', 'median_income',
        'house_age', 'avg_rooms', 'confidence_level',
        'model_used', 'created_at',
    )
    list_filter  = ('confidence_level', 'model_used', 'created_at')
    search_fields = ('confidence_level', 'model_used')
    readonly_fields = (
        'predicted_price', 'predicted_price_usd',
        'confidence_level', 'model_used', 'created_at',
    )
    ordering = ('-created_at',)

    fieldsets = (
        ('Prediction Result', {
            'fields': ('predicted_price', 'predicted_price_usd',
                       'confidence_level', 'model_used'),
        }),
        ('Input Features', {
            'fields': ('median_income', 'house_age', 'avg_rooms',
                       'avg_bedrooms', 'population', 'avg_occupancy',
                       'latitude', 'longitude'),
        }),
        ('Metadata', {
            'fields': ('created_at', 'ip_address'),
        }),
    )

    def predicted_price_usd_display(self, obj):
        return f"${obj.predicted_price_usd:,.0f}"
    predicted_price_usd_display.short_description = 'Predicted Price'
    predicted_price_usd_display.admin_order_field = 'predicted_price_usd'
