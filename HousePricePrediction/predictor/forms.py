"""
predictor/forms.py
Prediction input form with validation.
"""
from django import forms


class PredictionForm(forms.Form):
    median_income = forms.FloatField(
        label='Median Income (in $10k units)',
        min_value=0.5, max_value=15.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 3.5  (= $35,000 median income)',
            'step': '0.01',
        }),
        help_text='Block median income in tens of thousands of USD (0.5 – 15)'
    )

    house_age = forms.FloatField(
        label='House Age (years)',
        min_value=1.0, max_value=52.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 25',
            'step': '1',
        }),
        help_text='Median house age in the block (1 – 52 years)'
    )

    avg_rooms = forms.FloatField(
        label='Average Rooms per House',
        min_value=1.0, max_value=15.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 5.2',
            'step': '0.1',
        }),
        help_text='Average number of rooms per dwelling (1 – 15)'
    )

    avg_bedrooms = forms.FloatField(
        label='Average Bedrooms per House',
        min_value=1.0, max_value=5.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 1.8',
            'step': '0.1',
        }),
        help_text='Average number of bedrooms per dwelling (1 – 5)'
    )

    population = forms.FloatField(
        label='Block Population',
        min_value=3.0, max_value=35682.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 1200',
            'step': '1',
        }),
        help_text='Total population of the block group'
    )

    avg_occupancy = forms.FloatField(
        label='Average Occupancy (persons per house)',
        min_value=0.5, max_value=20.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 3.1',
            'step': '0.1',
        }),
        help_text='Average number of occupants per house (0.5 – 20)'
    )

    latitude = forms.FloatField(
        label='Latitude',
        min_value=32.5, max_value=42.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 37.88',
            'step': '0.0001',
        }),
        help_text='Block group latitude (California: 32.5 – 42.0)'
    )

    longitude = forms.FloatField(
        label='Longitude',
        min_value=-124.5, max_value=-114.5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. -122.23',
            'step': '0.0001',
        }),
        help_text='Block group longitude (California: -124.5 – -114.5)'
    )

    def clean(self):
        cleaned = super().clean()
        rooms = cleaned.get('avg_rooms')
        beds  = cleaned.get('avg_bedrooms')
        if rooms and beds and beds >= rooms:
            raise forms.ValidationError(
                'Average bedrooms must be less than average rooms.')
        return cleaned
