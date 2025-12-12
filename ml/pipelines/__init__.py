"""ML pipelines package initialization."""

from ml.pipelines.demand_forecasting import (
    DemandForecaster,
    train_aggregate_forecaster,
    train_item_level_forecaster,
)
from ml.pipelines.nlp_assistant import RestaurantAssistant, create_assistant
from ml.pipelines.recommender import (
    ContentBasedRecommender,
    CooccurrenceRecommender,
    HybridRecommender,
    train_hybrid_recommender,
)

__all__ = [
    # Demand Forecasting
    "DemandForecaster",
    "train_aggregate_forecaster",
    "train_item_level_forecaster",
    # Recommender
    "CooccurrenceRecommender",
    "ContentBasedRecommender",
    "HybridRecommender",
    "train_hybrid_recommender",
    # NLP Assistant
    "RestaurantAssistant",
    "create_assistant",
]
