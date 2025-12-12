"""
Hybrid Item Recommender System.

Combines collaborative filtering (co-occurrence) with content-based
embeddings for personalized menu item recommendations.
"""

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.utils.data_utils import (
    load_cooccurrence,
    load_menu_items,
    load_model,
    load_transactions,
    save_embeddings,
    save_model,
)


class CooccurrenceRecommender:
    """
    Collaborative filtering recommender based on item co-occurrence.

    Uses market basket analysis to recommend items frequently bought together.

    Attributes:
        cooccurrence_matrix: Dictionary of item pairs and their scores.
        item_popularities: Dictionary of item popularity scores.

    Example:
        >>> recommender = CooccurrenceRecommender()
        >>> recommender.fit(cooccurrence_df)
        >>> recommendations = recommender.recommend(['ITEM_0001'], top_n=5)
    """

    def __init__(self, min_support: float = 0.001, min_confidence: float = 0.1):
        """
        Initialize the co-occurrence recommender.

        Args:
            min_support: Minimum support threshold for associations.
            min_confidence: Minimum confidence threshold.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.cooccurrence_matrix: dict[str, dict[str, float]] = defaultdict(dict)
        self.item_popularities: dict[str, int] = {}
        self.item_names: dict[str, str] = {}
        self.trained_at: datetime | None = None

    def fit(
        self,
        cooccurrence_df: pd.DataFrame,
        menu_df: pd.DataFrame | None = None,
    ) -> "CooccurrenceRecommender":
        """
        Fit the recommender on co-occurrence data.

        Args:
            cooccurrence_df: DataFrame with co-occurrence statistics.
            menu_df: Optional menu items DataFrame for names.

        Returns:
            Self for method chaining.

        Example:
            >>> recommender = CooccurrenceRecommender()
            >>> recommender.fit(cooccurrence_df)
        """
        print("Building co-occurrence matrix...")

        # Filter by thresholds
        filtered = cooccurrence_df[
            (cooccurrence_df["support"] >= self.min_support)
            & (cooccurrence_df["confidence_1_to_2"] >= self.min_confidence)
        ]

        # Build symmetric co-occurrence matrix using lift as score
        for _, row in filtered.iterrows():
            item1 = row["item_id_1"]
            item2 = row["item_id_2"]
            score = row["lift"]

            self.cooccurrence_matrix[item1][item2] = score
            self.cooccurrence_matrix[item2][item1] = score

            # Store item names
            self.item_names[item1] = row["item_name_1"]
            self.item_names[item2] = row["item_name_2"]

            # Update popularity
            self.item_popularities[item1] = (
                self.item_popularities.get(item1, 0) + row["cooccurrence_count"]
            )
            self.item_popularities[item2] = (
                self.item_popularities.get(item2, 0) + row["cooccurrence_count"]
            )

        # Add menu item names if provided
        if menu_df is not None:
            for _, row in menu_df.iterrows():
                # Support both 'name' and 'item_name' columns
                name_col = "item_name" if "item_name" in menu_df.columns else "name"
                self.item_names[row["item_id"]] = row[name_col]

        self.trained_at = datetime.now()
        print(f"Built matrix with {len(self.cooccurrence_matrix)} items")

        return self

    def recommend(
        self,
        item_ids: list[str],
        top_n: int = 5,
        exclude_items: set[str] | None = None,
    ) -> list[dict]:
        """
        Get recommendations based on input items.

        Args:
            item_ids: List of item IDs in the current basket.
            top_n: Number of recommendations to return.
            exclude_items: Set of item IDs to exclude.

        Returns:
            List of recommended items with scores.

        Example:
            >>> recs = recommender.recommend(['ITEM_0001', 'ITEM_0002'])
            >>> for rec in recs:
            ...     print(f"{rec['item_name']}: {rec['score']:.2f}")
        """
        if exclude_items is None:
            exclude_items = set()

        # Add current items to exclusion
        exclude_items = exclude_items | set(item_ids)

        # Aggregate scores for candidate items
        candidate_scores: dict[str, float] = defaultdict(float)

        for item_id in item_ids:
            if item_id in self.cooccurrence_matrix:
                for related_item, score in self.cooccurrence_matrix[item_id].items():
                    if related_item not in exclude_items:
                        candidate_scores[related_item] += score

        # Sort by score
        sorted_items = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        recommendations = []
        for item_id, score in sorted_items:
            recommendations.append(
                {
                    "item_id": item_id,
                    "item_name": self.item_names.get(item_id, "Unknown"),
                    "score": round(score, 4),
                    "method": "cooccurrence",
                }
            )

        return recommendations

    def get_popular_items(self, top_n: int = 10) -> list[dict]:
        """
        Get most popular items.

        Args:
            top_n: Number of items to return.

        Returns:
            List of popular items.

        Example:
            >>> popular = recommender.get_popular_items(5)
        """
        sorted_items = sorted(
            self.item_popularities.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [
            {
                "item_id": item_id,
                "item_name": self.item_names.get(item_id, "Unknown"),
                "popularity_score": score,
            }
            for item_id, score in sorted_items
        ]


class ContentBasedRecommender:
    """
    Content-based recommender using item embeddings.

    Creates TF-IDF embeddings from item descriptions and uses
    cosine similarity for recommendations.

    Attributes:
        embeddings: Item embedding matrix.
        item_ids: List of item IDs.
        vectorizer: TF-IDF vectorizer.

    Example:
        >>> recommender = ContentBasedRecommender()
        >>> recommender.fit(menu_df)
        >>> recommendations = recommender.recommend('ITEM_0001', top_n=5)
    """

    def __init__(self, embedding_dim: int = 100):
        """
        Initialize the content-based recommender.

        Args:
            embedding_dim: Dimension of TF-IDF embeddings.
        """
        self.embedding_dim = embedding_dim
        self.embeddings: np.ndarray | None = None
        self.item_ids: list[str] = []
        self.item_data: dict[str, dict] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.similarity_matrix: np.ndarray | None = None
        self.trained_at: datetime | None = None

    def _create_item_text(self, row: pd.Series) -> str:
        """Create text representation of an item for TF-IDF."""
        # Support both 'name' and 'item_name' columns
        name = row.get("item_name") or row.get("name", "")
        parts = [
            name,
            row.get("description", "") or "",
            row.get("category", "") or "",
            row.get("subcategory", "") or "",
        ]

        # Add dietary tags
        if row.get("is_vegetarian"):
            parts.append("vegetarian")
        if row.get("is_vegan"):
            parts.append("vegan")
        if row.get("is_gluten_free"):
            parts.append("gluten free")

        return " ".join(parts)

    def fit(self, menu_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fit the recommender on menu item data.

        Args:
            menu_df: DataFrame with menu items.

        Returns:
            Self for method chaining.

        Example:
            >>> recommender = ContentBasedRecommender()
            >>> recommender.fit(menu_df)
        """
        print("Creating item embeddings...")

        # Support both 'name' and 'item_name' columns
        name_col = "item_name" if "item_name" in menu_df.columns else "name"

        # Store item data
        for _, row in menu_df.iterrows():
            self.item_ids.append(row["item_id"])
            self.item_data[row["item_id"]] = {
                "name": row[name_col],
                "category": row.get("category", ""),
                "price": row.get("price", 0),
            }

        # Create text representations
        texts = [self._create_item_text(row) for _, row in menu_df.iterrows()]

        # Create TF-IDF embeddings
        self.vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.embeddings = self.vectorizer.fit_transform(texts).toarray()

        # Compute similarity matrix
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)

        self.trained_at = datetime.now()
        print(f"Created embeddings for {len(self.item_ids)} items")

        return self

    def recommend(
        self,
        item_id: str,
        top_n: int = 5,
        exclude_items: set[str] | None = None,
    ) -> list[dict]:
        """
        Get similar items based on content.

        Args:
            item_id: Source item ID.
            top_n: Number of recommendations.
            exclude_items: Items to exclude.

        Returns:
            List of similar items with scores.

        Example:
            >>> recs = recommender.recommend('ITEM_0001', top_n=5)
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")

        if exclude_items is None:
            exclude_items = set()

        exclude_items.add(item_id)

        if item_id not in self.item_ids:
            return []

        item_idx = self.item_ids.index(item_id)
        similarities = self.similarity_matrix[item_idx]

        # Get top similar items
        recommendations = []
        sorted_indices = np.argsort(similarities)[::-1]

        for idx in sorted_indices:
            candidate_id = self.item_ids[idx]
            if candidate_id not in exclude_items:
                recommendations.append(
                    {
                        "item_id": candidate_id,
                        "item_name": self.item_data[candidate_id]["name"],
                        "score": round(float(similarities[idx]), 4),
                        "method": "content",
                    }
                )

                if len(recommendations) >= top_n:
                    break

        return recommendations

    def get_embedding(self, item_id: str) -> np.ndarray | None:
        """
        Get embedding for a specific item.

        Args:
            item_id: Item ID.

        Returns:
            Embedding vector or None if not found.
        """
        if item_id not in self.item_ids:
            return None

        idx = self.item_ids.index(item_id)
        return self.embeddings[idx]

    def save_embeddings(self, name: str = "item_embeddings") -> Path:
        """Save embeddings to disk."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Call fit() first.")

        return save_embeddings(
            self.embeddings,
            self.item_ids,
            name,
            {
                "embedding_dim": self.embedding_dim,
                "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            },
        )


class HybridRecommender:
    """
    Hybrid recommender combining collaborative and content-based approaches.

    Merges recommendations from co-occurrence and content-based methods
    with configurable weights.

    Attributes:
        cooccurrence_recommender: Co-occurrence based recommender.
        content_recommender: Content-based recommender.
        cooccurrence_weight: Weight for co-occurrence recommendations.

    Example:
        >>> hybrid = HybridRecommender()
        >>> hybrid.fit(cooccurrence_df, menu_df)
        >>> recs = hybrid.recommend(['ITEM_0001'], top_n=5)
    """

    def __init__(
        self,
        cooccurrence_weight: float = 0.6,
        content_weight: float = 0.4,
    ):
        """
        Initialize the hybrid recommender.

        Args:
            cooccurrence_weight: Weight for co-occurrence scores.
            content_weight: Weight for content-based scores.
        """
        self.cooccurrence_weight = cooccurrence_weight
        self.content_weight = content_weight
        self.cooccurrence_recommender = CooccurrenceRecommender()
        self.content_recommender = ContentBasedRecommender()
        self.trained_at: datetime | None = None

    def fit(
        self,
        cooccurrence_df: pd.DataFrame,
        menu_df: pd.DataFrame,
    ) -> "HybridRecommender":
        """
        Fit both recommenders.

        Args:
            cooccurrence_df: Co-occurrence statistics DataFrame.
            menu_df: Menu items DataFrame.

        Returns:
            Self for method chaining.
        """
        print("=" * 60)
        print("TRAINING HYBRID RECOMMENDER")
        print("=" * 60)

        # Fit co-occurrence recommender
        print("\n[1/2] Training co-occurrence recommender...")
        self.cooccurrence_recommender.fit(cooccurrence_df, menu_df)

        # Fit content-based recommender
        print("\n[2/2] Training content-based recommender...")
        self.content_recommender.fit(menu_df)

        self.trained_at = datetime.now()
        print("\nHybrid recommender training complete!")

        return self

    def recommend(
        self,
        item_ids: list[str],
        top_n: int = 5,
        exclude_items: set[str] | None = None,
        method: str = "hybrid",
    ) -> list[dict]:
        """
        Get recommendations using specified method.

        Args:
            item_ids: List of item IDs in current basket.
            top_n: Number of recommendations.
            exclude_items: Items to exclude.
            method: 'hybrid', 'cooccurrence', or 'content'.

        Returns:
            List of recommended items with scores.

        Example:
            >>> recs = hybrid.recommend(['ITEM_0001'], method='hybrid')
        """
        if exclude_items is None:
            exclude_items = set()

        exclude_items = exclude_items | set(item_ids)

        if method == "cooccurrence":
            return self.cooccurrence_recommender.recommend(item_ids, top_n, exclude_items)

        if method == "content":
            # For content-based, use the first item as reference
            if item_ids:
                return self.content_recommender.recommend(item_ids[0], top_n, exclude_items)
            return []

        # Hybrid method
        cooc_recs = self.cooccurrence_recommender.recommend(item_ids, top_n * 2, exclude_items)

        content_recs = []
        for item_id in item_ids:
            content_recs.extend(self.content_recommender.recommend(item_id, top_n, exclude_items))

        # Merge and re-score
        merged_scores: dict[str, dict] = {}

        for rec in cooc_recs:
            item_id = rec["item_id"]
            if item_id not in merged_scores:
                merged_scores[item_id] = {
                    "item_id": item_id,
                    "item_name": rec["item_name"],
                    "cooc_score": 0,
                    "content_score": 0,
                }
            merged_scores[item_id]["cooc_score"] = rec["score"]

        for rec in content_recs:
            item_id = rec["item_id"]
            if item_id not in merged_scores:
                merged_scores[item_id] = {
                    "item_id": item_id,
                    "item_name": rec["item_name"],
                    "cooc_score": 0,
                    "content_score": 0,
                }
            # Take max content score if multiple
            merged_scores[item_id]["content_score"] = max(
                merged_scores[item_id]["content_score"],
                rec["score"],
            )

        # Normalize and combine scores
        max_cooc = max((m["cooc_score"] for m in merged_scores.values()), default=1)
        max_content = max((m["content_score"] for m in merged_scores.values()), default=1)

        recommendations = []
        for item_id, scores in merged_scores.items():
            norm_cooc = scores["cooc_score"] / max_cooc if max_cooc > 0 else 0
            norm_content = scores["content_score"] / max_content if max_content > 0 else 0

            final_score = self.cooccurrence_weight * norm_cooc + self.content_weight * norm_content

            recommendations.append(
                {
                    "item_id": item_id,
                    "item_name": scores["item_name"],
                    "score": round(final_score, 4),
                    "cooccurrence_score": round(scores["cooc_score"], 4),
                    "content_score": round(scores["content_score"], 4),
                    "method": "hybrid",
                }
            )

        # Sort by final score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return recommendations[:top_n]

    def recommend_for_customer(
        self,
        customer_id: str,
        transactions_df: pd.DataFrame,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Get personalized recommendations based on customer history.

        Args:
            customer_id: Customer ID.
            transactions_df: Historical transactions.
            top_n: Number of recommendations.

        Returns:
            List of personalized recommendations.
        """
        # Get customer's recently ordered items
        customer_txns = transactions_df[transactions_df["customer_id"] == customer_id].sort_values(
            "timestamp", ascending=False
        )

        if len(customer_txns) == 0:
            # Cold start - return popular items
            return self.cooccurrence_recommender.get_popular_items(top_n)

        # Get last 10 unique items ordered
        recent_items = customer_txns["item_id"].drop_duplicates().head(10).tolist()

        # Exclude all previously ordered items for novelty
        all_ordered = set(customer_txns["item_id"].unique())

        return self.recommend(
            recent_items,
            top_n=top_n,
            exclude_items=all_ordered,
            method="hybrid",
        )

    def save(self, name: str = "hybrid_recommender") -> Path:
        """Save the hybrid recommender."""
        metadata = {
            "cooccurrence_weight": self.cooccurrence_weight,
            "content_weight": self.content_weight,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }

        # Save embeddings
        self.content_recommender.save_embeddings("item_embeddings")

        # Save full model
        model_data = {
            "cooccurrence_recommender": self.cooccurrence_recommender,
            "content_recommender": self.content_recommender,
        }

        return save_model(model_data, name, metadata)

    @classmethod
    def load(cls, name: str = "hybrid_recommender") -> "HybridRecommender":
        """Load a trained hybrid recommender."""
        model_data, metadata = load_model(name)

        hybrid = cls(
            cooccurrence_weight=metadata.get("cooccurrence_weight", 0.6),
            content_weight=metadata.get("content_weight", 0.4),
        )

        hybrid.cooccurrence_recommender = model_data["cooccurrence_recommender"]
        hybrid.content_recommender = model_data["content_recommender"]
        hybrid.trained_at = (
            datetime.fromisoformat(metadata["trained_at"]) if metadata.get("trained_at") else None
        )

        return hybrid


def train_hybrid_recommender() -> tuple[HybridRecommender, dict]:
    """
    Train the hybrid recommender on available data.

    Returns:
        Tuple[HybridRecommender, Dict]: Trained recommender and sample recommendations.

    Example:
        >>> recommender, samples = train_hybrid_recommender()
    """
    # Load data
    cooccurrence_df = load_cooccurrence()
    menu_df = load_menu_items()
    transactions_df = load_transactions()

    # Train recommender
    hybrid = HybridRecommender(cooccurrence_weight=0.6, content_weight=0.4)
    hybrid.fit(cooccurrence_df, menu_df)

    # Save model
    hybrid.save("hybrid_recommender")

    # Generate sample recommendations
    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 60)

    sample_items = menu_df["item_id"].sample(3).tolist()

    # Support both 'name' and 'item_name' columns
    name_col = "item_name" if "item_name" in menu_df.columns else "name"

    for item_id in sample_items:
        item_name = menu_df[menu_df["item_id"] == item_id][name_col].values[0]
        print(f"\nRecommendations for '{item_name}':")

        recs = hybrid.recommend([item_id], top_n=5)
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['item_name']} (score: {rec['score']:.3f})")

    return hybrid, {"sample_items": sample_items}


if __name__ == "__main__":
    hybrid, samples = train_hybrid_recommender()
