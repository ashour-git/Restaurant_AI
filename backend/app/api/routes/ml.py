"""
ML API routes for AI features.

Provides endpoints for:
- Demand forecasting predictions
- Menu item recommendations
- NLP assistant chat
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Global model instances (initialized on startup)
_demand_forecaster = None
_recommender = None
_assistant = None


# Request/Response Models


class ForecastRequest(BaseModel):
    """Request for demand forecast."""

    item_id: int | None = Field(None, description="Specific item ID to forecast")
    days_ahead: int = Field(7, ge=1, le=30, description="Days to forecast")

    model_config = {"json_schema_extra": {"example": {"item_id": 1, "days_ahead": 7}}}


class ForecastResponse(BaseModel):
    """Response with demand forecast."""

    item_id: int | None
    forecast_type: str
    predictions: list[dict[str, Any]]


class RecommendationRequest(BaseModel):
    """Request for item recommendations."""

    item_ids: list[int] = Field(..., description="List of item IDs for context")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations")
    strategy: str = Field(
        "hybrid", description="Recommendation strategy: hybrid, cooccurrence, content"
    )

    model_config = {
        "json_schema_extra": {"example": {"item_ids": [1, 2, 3], "top_k": 5, "strategy": "hybrid"}}
    }


class RecommendationResponse(BaseModel):
    """Response with recommendations."""

    recommendations: list[dict[str, Any]]
    strategy_used: str


class ChatRequest(BaseModel):
    """Request for chat with assistant."""

    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    reset_conversation: bool = Field(False, description="Reset conversation history")
    sync_data: bool = Field(True, description="Whether to sync live DB data before responding")

    model_config = {
        "json_schema_extra": {
            "example": {"message": "What vegetarian options do you have?", "use_rag": True}
        }
    }


class SyncKnowledgeRequest(BaseModel):
    """Request to sync knowledge base with live data."""
    
    menu_items: list[dict[str, Any]] = Field(default_factory=list)
    orders: list[dict[str, Any]] | None = Field(None)
    customers: list[dict[str, Any]] | None = Field(None)
    inventory: list[dict[str, Any]] | None = Field(None)


class ChatResponse(BaseModel):
    """Response from chat assistant."""

    response: str
    context_used: bool


class MenuRecommendationRequest(BaseModel):
    """Request for menu recommendations."""

    dietary_preference: str | None = Field(None, description="Dietary preference filter")
    category: str | None = Field(None, description="Menu category filter")
    max_price: float | None = Field(None, gt=0, description="Maximum price filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations")


class MenuRecommendationResponse(BaseModel):
    """Response with menu recommendations."""

    recommendations: list[dict[str, Any]]
    filters_applied: dict[str, Any]


# Endpoints


@router.get("/health")
async def ml_health_check() -> dict[str, Any]:
    """Check ML service health and model availability."""
    status = {
        "demand_forecaster": "loaded" if _demand_forecaster else "not_loaded",
        "recommender": "loaded" if _recommender else "not_loaded",
        "assistant": "loaded" if _assistant else "not_loaded",
    }
    return {"status": "healthy", "models": status}


@router.post("/forecast", response_model=ForecastResponse)
async def get_demand_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Get demand forecast predictions.

    Returns predicted demand for the specified period.
    """
    if _demand_forecaster is None:
        raise HTTPException(status_code=503, detail="Demand forecaster not initialized")

    try:
        # Generate predictions
        predictions = _demand_forecaster.predict(
            days_ahead=request.days_ahead, item_id=request.item_id
        )

        forecast_type = "item_level" if request.item_id else "aggregate"

        return ForecastResponse(
            item_id=request.item_id, forecast_type=forecast_type, predictions=predictions
        )

    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate forecast. Please try again.")


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get item recommendations based on context items.

    Uses hybrid recommendation combining co-occurrence and content similarity.
    """
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    try:
        # Convert integer item IDs to string format used by recommender
        item_ids = [f"ITEM_{str(i).zfill(4)}" for i in request.item_ids]

        # Use the recommend method with strategy parameter
        recs = _recommender.recommend(
            item_ids=item_ids, top_n=request.top_k, method=request.strategy
        )

        return RecommendationResponse(recommendations=recs, strategy_used=request.strategy)

    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate recommendations. Please try again.")


@router.post("/sync-knowledge")
async def sync_knowledge_base(request: SyncKnowledgeRequest) -> dict[str, Any]:
    """
    Sync live database data into the AI assistant's knowledge base.
    
    This enables the AI to answer questions about current menu items,
    recent orders, customers, and inventory from the live database.
    """
    if _assistant is None:
        raise HTTPException(
            status_code=503, detail="Assistant not initialized. Please set GROQ_API_KEY."
        )

    try:
        count = _assistant.sync_knowledge(
            menu_items=request.menu_items,
            orders=request.orders,
            customers=request.customers,
            inventory=request.inventory,
        )
        return {"status": "success", "documents_indexed": count}
    except Exception as e:
        logger.error(f"Knowledge sync error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to sync knowledge base.")


# Helper to fetch live data from database
async def _fetch_live_data_for_rag() -> dict[str, Any]:
    """Fetch live menu items, orders, customers for RAG context."""
    from app.core.database import get_db
    from app.models import MenuItem, Order, Subcategory, Category
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    
    data = {"menu_items": [], "orders": [], "customers": []}
    
    try:
        async for db in get_db():
            # Fetch menu items with subcategory->category relationships
            result = await db.execute(
                select(MenuItem)
                .options(selectinload(MenuItem.subcategory).selectinload(Subcategory.category))
                .where(MenuItem.is_active == True)
                .limit(100)
            )
            items = result.scalars().all()
            
            menu_list = []
            for item in items:
                cat_name = "General"
                if item.subcategory and item.subcategory.category:
                    cat_name = item.subcategory.category.name
                menu_list.append({
                    "id": item.id,
                    "name": item.name,
                    "description": item.description or "",
                    "price": float(item.price) if item.price else 0,
                    "category": cat_name,
                    "is_vegetarian": getattr(item, "is_vegetarian", False),
                    "is_vegan": getattr(item, "is_vegan", False),
                    "is_gluten_free": getattr(item, "is_gluten_free", False),
                    "is_active": item.is_active,
                })
            data["menu_items"] = menu_list
            
            # Fetch recent orders
            result = await db.execute(select(Order).order_by(Order.created_at.desc()).limit(20))
            orders = result.scalars().all()
            data["orders"] = [
                {
                    "id": o.id,
                    "order_number": o.order_number,
                    "total": float(o.total) if o.total else 0,
                    "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
                }
                for o in orders
            ]
            break
    except Exception as e:
        logger.warning(f"Could not fetch live data for RAG: {e}")
    
    return data


@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest) -> ChatResponse:
    """
    Chat with the AI restaurant assistant.

    Uses RAG for context-aware responses about menu, operations, etc.
    Automatically syncs live database data for accurate responses.
    """
    if _assistant is None:
        raise HTTPException(
            status_code=503, detail="Assistant not initialized. Please set GROQ_API_KEY."
        )

    try:
        if request.reset_conversation:
            _assistant.clear_history()
        
        # Sync live database data before responding (if enabled)
        if request.sync_data:
            try:
                live_data = await _fetch_live_data_for_rag()
                if live_data.get("menu_items"):
                    _assistant.sync_knowledge(
                        menu_items=live_data["menu_items"],
                        orders=live_data.get("orders"),
                    )
            except Exception as sync_error:
                logger.warning(f"RAG sync failed, using cached data: {sync_error}")

        response = _assistant.chat(message=request.message, include_context=request.use_rag)

        return ChatResponse(response=response, context_used=request.use_rag)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat message. Please try again.")


@router.post("/menu-recommendations", response_model=MenuRecommendationResponse)
async def get_menu_recommendations(
    request: MenuRecommendationRequest,
) -> MenuRecommendationResponse:
    """
    Get menu item recommendations with filters.

    Filter by dietary preference, category, and price.
    """
    if _assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")

    try:
        recs = _assistant.get_menu_recommendations(
            dietary_preference=request.dietary_preference,
            category=request.category,
            max_price=request.max_price,
            top_k=request.top_k,
        )

        return MenuRecommendationResponse(
            recommendations=recs,
            filters_applied={
                "dietary_preference": request.dietary_preference,
                "category": request.category,
                "max_price": request.max_price,
            },
        )

    except Exception as e:
        logger.error(f"Menu recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get menu recommendations. Please try again.")


@router.get("/menu-search")
async def search_menu(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
) -> dict[str, Any]:
    """
    Search menu items using semantic search.
    """
    if _assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")

    try:
        results = _assistant.document_store.search(query, top_k)

        # Filter to menu items only
        menu_results = [r for r in results if r.get("type") == "menu_item"]

        return {"query": query, "results": menu_results}

    except Exception as e:
        logger.error(f"Menu search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to search menu. Please try again.")


# Model initialization functions


def initialize_demand_forecaster(model_path: str | None = None) -> None:
    """Initialize the demand forecaster model."""
    global _demand_forecaster

    try:
        from ml.pipelines.demand_forecasting import DemandForecaster
        from ml.utils.data_utils import load_model

        if model_path:
            _demand_forecaster = load_model(model_path)
        else:
            # Create new forecaster (would need training in production)
            _demand_forecaster = DemandForecaster()

        logger.info("Demand forecaster initialized")

    except Exception as e:
        logger.warning(f"Could not initialize demand forecaster: {e}")


def initialize_recommender(model_path: str | None = None) -> None:
    """Initialize the recommender model."""
    global _recommender

    try:
        from ml.pipelines.recommender import train_hybrid_recommender
        from ml.utils.data_utils import load_model

        if model_path:
            _recommender = load_model(model_path)
        else:
            # Train a new recommender - returns tuple (recommender, samples)
            result = train_hybrid_recommender()
            # Handle both tuple return and direct return
            _recommender = result[0] if isinstance(result, tuple) else result

        logger.info("Recommender initialized")

    except Exception as e:
        logger.warning(f"Could not initialize recommender: {e}")


def initialize_assistant(api_key: str | None = None) -> None:
    """Initialize the NLP assistant."""
    global _assistant

    try:
        from ml.pipelines.nlp_assistant import create_assistant

        _assistant = create_assistant(api_key=api_key)
        logger.info("NLP assistant initialized")

    except Exception as e:
        logger.warning(f"Could not initialize assistant: {e}")


def initialize_all_models(
    demand_model_path: str | None = None,
    recommender_model_path: str | None = None,
    groq_api_key: str | None = None,
) -> None:
    """Initialize all ML models."""
    initialize_demand_forecaster(demand_model_path)
    initialize_recommender(recommender_model_path)
    initialize_assistant(groq_api_key)
