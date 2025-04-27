"""Prometheus monitoring metrics for the application."""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Define metrics
PREDICTION_COUNTER = Counter(
    'image_predictions_total',
    'Total number of image predictions made',
    ['class_name', 'user_email'] # Added user_email label
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    ['user_email'], # Added user_email label
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

UPLOAD_COUNTER = Counter(
    'image_uploads_total',
    'Total number of images uploaded',
    ['user_email'] # Added user_email label
)

HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests received',
    ['method', 'path'] # Labels for method and path
)

def get_metrics():
    """Generate latest metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)