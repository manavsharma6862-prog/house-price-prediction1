"""
predictor/views.py
All views: Home, Predict, Result, Dashboard, History, About
"""
import os
import joblib
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.utils import timezone
from django.db.models import Avg, Count
from django.core.paginator import Paginator

from .forms import PredictionForm
from .models import PredictionRecord

MODEL_DIR = settings.MODEL_DIR

_model       = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
_scaler      = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
_model_info  = joblib.load(os.path.join(MODEL_DIR, 'best_model_info.pkl'))
_all_results = joblib.load(os.path.join(MODEL_DIR, 'model_results.pkl'))
_feat_names  = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))


def _engineer(data):
    mi  = data['median_income']
    ar  = data['avg_rooms']
    ab  = data['avg_bedrooms']
    ao  = data['avg_occupancy']
    row = [mi, data['house_age'], ar, ab, data['population'], ao,
           data['latitude'], data['longitude'],
           ar / (ao + 1e-9), ab / (ar + 1e-9), mi / (ar + 1e-9)]
    return np.array(row).reshape(1, -1)


def _confidence(p):
    if 0.5 <= p <= 4.0:   return 'High Confidence'
    elif p < 0.5:          return 'Low Confidence'
    else:                  return 'Medium Confidence'


def _get_ip(request):
    x = request.META.get('HTTP_X_FORWARDED_FOR')
    return x.split(',')[0] if x else request.META.get('REMOTE_ADDR')


def home(request):
    total = PredictionRecord.objects.count()
    avg_price = PredictionRecord.objects.aggregate(avg=Avg('predicted_price_usd'))['avg'] or 0
    recent = PredictionRecord.objects.all()[:5]
    model_results = [{'name': n, 'r2': m['R2 Score'], 'rmse': m['RMSE'],
                       'mae': m['MAE'], 'best': n == _model_info['name']}
                     for n, m in _all_results.items()]
    return render(request, 'predictor/home.html', {
        'total_predictions': total, 'avg_price': avg_price,
        'recent': recent, 'model_results': model_results,
        'best_model': _model_info['name'],
        'best_r2': _model_info['metrics']['R2 Score'],
    })


def predict(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            pred_100k  = float(_model.predict(_scaler.transform(_engineer(data)))[0])
            pred_usd   = pred_100k * 100_000
            confidence = _confidence(pred_100k)
            record = PredictionRecord.objects.create(
                **{k: data[k] for k in data},
                predicted_price=round(pred_100k, 4),
                predicted_price_usd=round(pred_usd, 2),
                confidence_level=confidence,
                model_used=_model_info['name'],
                ip_address=_get_ip(request),
            )
            return redirect('result', pk=record.pk)
    else:
        form = PredictionForm(initial={
            'median_income': 3.5, 'house_age': 25, 'avg_rooms': 5.2,
            'avg_bedrooms': 1.8, 'population': 1200, 'avg_occupancy': 3.1,
            'latitude': 37.88, 'longitude': -122.23,
        })
    return render(request, 'predictor/predict.html', {'form': form})


def result(request, pk):
    try:
        record = PredictionRecord.objects.get(pk=pk)
    except PredictionRecord.DoesNotExist:
        return redirect('predict')
    input_data = [
        ('Median Income',  f"${record.median_income * 10000:,.0f} / year"),
        ('House Age',      f"{record.house_age:.0f} years"),
        ('Avg Rooms',      f"{record.avg_rooms:.1f}"),
        ('Avg Bedrooms',   f"{record.avg_bedrooms:.1f}"),
        ('Population',     f"{record.population:,.0f}"),
        ('Avg Occupancy',  f"{record.avg_occupancy:.1f} persons"),
        ('Latitude',       f"{record.latitude:.4f}"),
        ('Longitude',      f"{record.longitude:.4f}"),
    ]
    return render(request, 'predictor/result.html', {
        'record': record, 'input_data': input_data,
        'badge_class': record.confidence_badge_class,
    })


def dashboard(request):
    qs    = PredictionRecord.objects.all()
    total = qs.count()
    avg_usd = qs.aggregate(avg=Avg('predicted_price_usd'))['avg'] or 0
    high_conf = qs.filter(confidence_level='High Confidence').count()
    med_conf  = qs.filter(confidence_level='Medium Confidence').count()
    low_conf  = qs.filter(confidence_level='Low Confidence').count()
    recent10  = qs[:10]
    model_results = [{'name': n, 'r2': m['R2 Score'], 'rmse': m['RMSE'],
                       'mae': m['MAE'], 'best': n == _model_info['name']}
                     for n, m in _all_results.items()]
    buckets = {'<$100k': 0, '$100–200k': 0, '$200–300k': 0,
               '$300–400k': 0, '>$400k': 0}
    for r in qs:
        p = r.predicted_price_usd
        if p < 100000:   buckets['<$100k'] += 1
        elif p < 200000: buckets['$100–200k'] += 1
        elif p < 300000: buckets['$200–300k'] += 1
        elif p < 400000: buckets['$300–400k'] += 1
        else:            buckets['>$400k'] += 1
    return render(request, 'predictor/dashboard.html', {
        'total': total, 'avg_usd': avg_usd,
        'high_conf': high_conf, 'med_conf': med_conf, 'low_conf': low_conf,
        'recent10': recent10, 'model_results': model_results,
        'best_model': _model_info['name'],
        'buckets_labels': list(buckets.keys()),
        'buckets_values': list(buckets.values()),
    })


def history(request):
    qs = PredictionRecord.objects.all()
    paginator = Paginator(qs, 15)
    page_obj  = paginator.get_page(request.GET.get('page', 1))
    return render(request, 'predictor/history.html', {'page_obj': page_obj})


def about(request):
    model_results = [{'name': n, 'r2': m['R2 Score'], 'rmse': m['RMSE'],
                       'mae': m['MAE'], 'best': n == _model_info['name']}
                     for n, m in _all_results.items()]
    tech_stack = [
        {'name': 'Python 3.11',        'icon': '🐍', 'role': 'Core Language'},
        {'name': 'Django 4.2',         'icon': '🌐', 'role': 'Web Framework'},
        {'name': 'Scikit-Learn',       'icon': '🤖', 'role': 'Machine Learning'},
        {'name': 'Pandas & NumPy',     'icon': '📊', 'role': 'Data Processing'},
        {'name': 'Matplotlib/Seaborn', 'icon': '📈', 'role': 'Visualization'},
        {'name': 'Bootstrap 5',        'icon': '🎨', 'role': 'UI Framework'},
        {'name': 'SQLite',             'icon': '🗄️',  'role': 'Database'},
        {'name': 'Joblib',             'icon': '💾', 'role': 'Model Persistence'},
    ]
    return render(request, 'predictor/about.html', {
        'model_results': model_results,
        'best_model': _model_info['name'],
        'tech_stack': tech_stack,
    })
