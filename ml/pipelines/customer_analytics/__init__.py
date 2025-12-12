"""
Customer Analytics Module.

Provides ML pipelines for customer-centric analytics:
- RFM (Recency, Frequency, Monetary) Analysis
- Customer Segmentation using clustering
- Churn Prediction
- Customer Lifetime Value (CLV) Prediction
"""

from ml.pipelines.customer_analytics.churn_prediction import (
    ChurnPredictor,
    predict_churn,
    train_churn_model,
)
from ml.pipelines.customer_analytics.customer_ltv import CLVPredictor, predict_clv, train_clv_model
from ml.pipelines.customer_analytics.customer_segmentation import (
    CustomerSegmenter,
    get_segment_profiles,
    train_segmentation_model,
)
from ml.pipelines.customer_analytics.rfm_segmentation import (
    RFMAnalyzer,
    compute_rfm_features,
    segment_customers_rfm,
)

__all__ = [
    # RFM Analysis
    "RFMAnalyzer",
    "compute_rfm_features",
    "segment_customers_rfm",
    # Churn Prediction
    "ChurnPredictor",
    "train_churn_model",
    "predict_churn",
    # Customer Segmentation
    "CustomerSegmenter",
    "train_segmentation_model",
    "get_segment_profiles",
    # Customer LTV
    "CLVPredictor",
    "train_clv_model",
    "predict_clv",
]
