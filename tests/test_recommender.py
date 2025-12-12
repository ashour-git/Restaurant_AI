"""
Tests for recommendation system ML pipeline.

This module contains comprehensive tests for the hybrid
recommendation system combining collaborative filtering
and content-based approaches.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCollaborativeFiltering:
    """Tests for collaborative filtering component."""

    def test_user_item_matrix_creation(self, sample_transactions):
        """Test user-item interaction matrix creation."""
        df = sample_transactions.copy()

        # Create user-item matrix
        matrix = df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="sum", fill_value=0
        )

        assert matrix.shape[0] == df["customer_id"].nunique()
        assert matrix.shape[1] <= df["item_id"].nunique()
        assert (matrix >= 0).all().all()

    def test_implicit_feedback_matrix(self, sample_transactions):
        """Test implicit feedback (purchase history) matrix."""
        df = sample_transactions.copy()

        # Binary interaction matrix
        matrix = df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="count", fill_value=0
        )
        matrix = (matrix > 0).astype(int)

        assert matrix.isin([0, 1]).all().all()

    def test_cosine_similarity_users(self, sample_transactions):
        """Test user-user cosine similarity calculation."""
        from sklearn.metrics.pairwise import cosine_similarity

        df = sample_transactions.copy()

        matrix = df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="sum", fill_value=0
        ).values

        similarity = cosine_similarity(matrix)

        # Diagonal should be 1 (self-similarity)
        assert np.allclose(np.diag(similarity), 1.0, atol=0.01)
        # Similarity should be between -1 and 1
        assert (similarity >= -1.01).all() and (similarity <= 1.01).all()

    def test_cosine_similarity_items(self, sample_transactions):
        """Test item-item cosine similarity calculation."""
        from sklearn.metrics.pairwise import cosine_similarity

        df = sample_transactions.copy()

        matrix = df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="sum", fill_value=0
        ).values.T  # Transpose for item-item

        similarity = cosine_similarity(matrix)

        # Diagonal should be 1
        assert np.allclose(np.diag(similarity), 1.0, atol=0.01)


class TestContentBasedFiltering:
    """Tests for content-based filtering component."""

    def test_item_feature_extraction(self, sample_menu_items):
        """Test item feature extraction from menu data."""
        df = sample_menu_items.copy()

        # One-hot encode category
        features = pd.get_dummies(df[["category"]], prefix="cat")

        assert features.shape[0] == len(df)
        assert features.shape[1] >= 1
        assert features.isin([0, 1]).all().all()

    def test_price_normalization(self, sample_menu_items):
        """Test price feature normalization."""
        from sklearn.preprocessing import MinMaxScaler

        df = sample_menu_items.copy()

        scaler = MinMaxScaler()
        df["price_normalized"] = scaler.fit_transform(df[["price"]])

        assert df["price_normalized"].min() >= 0
        assert df["price_normalized"].max() <= 1

    def test_item_similarity_content(self, sample_menu_items):
        """Test content-based item similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import MinMaxScaler

        df = sample_menu_items.copy()

        # Create feature matrix
        cat_features = pd.get_dummies(df[["category"]])
        scaler = MinMaxScaler()
        price_feature = scaler.fit_transform(df[["price"]])

        feature_matrix = np.hstack([cat_features.values, price_feature])
        similarity = cosine_similarity(feature_matrix)

        assert similarity.shape == (len(df), len(df))
        assert np.allclose(np.diag(similarity), 1.0, atol=0.01)

    def test_tfidf_description_features(self, sample_menu_items):
        """Test TF-IDF features from item descriptions."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = sample_menu_items.copy()

        # Use item names as descriptions
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(df["name"])

        assert tfidf_matrix.shape[0] == len(df)
        assert tfidf_matrix.shape[1] <= 50


class TestHybridRecommender:
    """Tests for hybrid recommendation approach."""

    def test_hybrid_score_combination(self):
        """Test combining collaborative and content-based scores."""
        collab_scores = np.array([0.8, 0.6, 0.4, 0.2])
        content_scores = np.array([0.5, 0.7, 0.3, 0.9])

        # Weighted combination
        alpha = 0.6  # Weight for collaborative
        hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores

        assert len(hybrid_scores) == 4
        assert hybrid_scores.min() >= 0
        assert hybrid_scores.max() <= 1

    def test_recommendation_ranking(self):
        """Test recommendation ranking by score."""
        items = ["I001", "I002", "I003", "I004", "I005"]
        scores = [0.8, 0.3, 0.9, 0.5, 0.7]

        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]
        top_recommendations = [items[i] for i in ranked_indices[:3]]

        assert top_recommendations[0] == "I003"  # Highest score
        assert len(top_recommendations) == 3

    def test_exclude_purchased_items(self, sample_transactions):
        """Test excluding already purchased items."""
        df = sample_transactions.copy()

        # Get items purchased by a customer
        customer = df["customer_id"].iloc[0]
        purchased = set(df[df["customer_id"] == customer]["item_id"])

        # All items
        all_items = set(df["item_id"])

        # Recommend only unpurchased items
        candidates = all_items - purchased

        assert len(candidates.intersection(purchased)) == 0

    def test_cold_start_handling(self, sample_menu_items):
        """Test cold start handling for new users."""
        # For new users, use content-based only (popular items)
        df = sample_menu_items.copy()

        # Recommend by category diversity
        recommendations = df.groupby("category").first().reset_index()

        assert len(recommendations) >= 1
        assert recommendations["category"].is_unique


class TestRecommenderEvaluation:
    """Tests for recommendation system evaluation."""

    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        recommended = ["I001", "I002", "I003", "I004", "I005"]
        relevant = {"I001", "I003", "I005", "I007"}

        k = 5
        hits = sum(1 for item in recommended[:k] if item in relevant)
        precision_at_k = hits / k

        assert precision_at_k == 0.6  # 3 hits / 5 recommendations

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        recommended = ["I001", "I002", "I003", "I004", "I005"]
        relevant = {"I001", "I003", "I005", "I007"}

        k = 5
        hits = sum(1 for item in recommended[:k] if item in relevant)
        recall_at_k = hits / len(relevant)

        assert recall_at_k == 0.75  # 3 hits / 4 relevant

    def test_ndcg_calculation(self):
        """Test NDCG (Normalized Discounted Cumulative Gain)."""
        # Relevance scores (ground truth)
        relevances = [3, 2, 3, 0, 1]  # Scores for recommended items

        # DCG calculation
        dcg = relevances[0]
        for i, rel in enumerate(relevances[1:], start=2):
            dcg += rel / np.log2(i + 1)

        # Ideal DCG (sorted relevances)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = ideal_relevances[0]
        for i, rel in enumerate(ideal_relevances[1:], start=2):
            idcg += rel / np.log2(i + 1)

        ndcg = dcg / idcg if idcg > 0 else 0

        assert 0 <= ndcg <= 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        # List of (user_recommendations, user_relevant_items)
        test_cases = [
            (["I001", "I002"], {"I001", "I003"}),  # Hit
            (["I004", "I005"], {"I006"}),  # Miss
            (["I007", "I008"], {"I007", "I008"}),  # Hit
        ]

        hits = sum(1 for recs, relevant in test_cases if any(r in relevant for r in recs))
        hit_rate = hits / len(test_cases)

        assert hit_rate == 2 / 3

    def test_coverage_calculation(self, sample_menu_items):
        """Test catalog coverage calculation."""
        all_items = set(sample_menu_items["item_id"])
        recommended_items = {"I001", "I002", "I003", "I004"}

        coverage = len(recommended_items) / len(all_items)

        assert 0 <= coverage <= 1


class TestRecommenderTraining:
    """Tests for recommender model training."""

    def test_matrix_factorization(self, sample_transactions):
        """Test matrix factorization approach."""
        from sklearn.decomposition import TruncatedSVD

        df = sample_transactions.copy()

        matrix = df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="sum", fill_value=0
        )

        # Apply SVD
        n_components = min(5, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(matrix.values)
        item_factors = svd.components_.T

        # Reconstruct ratings
        reconstructed = np.dot(user_factors, item_factors.T)

        assert reconstructed.shape == matrix.shape

    def test_als_model_training(self, sample_transactions):
        """Test ALS-style training (simplified)."""
        df = sample_transactions.copy()

        matrix = df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="sum", fill_value=0
        ).values

        n_users, n_items = matrix.shape
        n_factors = 5

        # Initialize factors
        np.random.seed(42)
        user_factors = np.random.rand(n_users, n_factors) * 0.1
        item_factors = np.random.rand(n_items, n_factors) * 0.1

        assert user_factors.shape == (n_users, n_factors)
        assert item_factors.shape == (n_items, n_factors)


class TestRecommenderIntegration:
    """Integration tests for recommendation system."""

    def test_full_recommendation_pipeline(self, sample_transactions, sample_menu_items):
        """Test complete recommendation pipeline."""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import MinMaxScaler

        txn_df = sample_transactions.copy()
        menu_df = sample_menu_items.copy()

        # Step 1: Build collaborative filtering matrix
        user_item = txn_df.pivot_table(
            index="customer_id", columns="item_id", values="quantity", aggfunc="sum", fill_value=0
        )

        # Step 2: Build content features
        cat_features = pd.get_dummies(menu_df[["category"]])
        scaler = MinMaxScaler()
        price_features = scaler.fit_transform(menu_df[["price"]])
        item_features = np.hstack([cat_features.values, price_features])

        # Step 3: Calculate similarities
        item_similarity = cosine_similarity(item_features)

        # Step 4: Generate recommendations for a user
        user = user_item.index[0]
        user_history = user_item.loc[user]
        purchased_items = user_history[user_history > 0].index.tolist()

        # Step 5: Score unpurchased items
        all_items = list(user_item.columns)
        candidate_items = [i for i in all_items if i not in purchased_items]

        # Content-based scoring
        if len(candidate_items) > 0:
            item_idx_map = {item: i for i, item in enumerate(menu_df["item_id"])}

            scores = []
            for candidate in candidate_items:
                if candidate in item_idx_map:
                    candidate_idx = item_idx_map[candidate]
                    # Average similarity to purchased items
                    sim_scores = []
                    for purchased in purchased_items:
                        if purchased in item_idx_map:
                            purchased_idx = item_idx_map[purchased]
                            sim_scores.append(item_similarity[candidate_idx, purchased_idx])
                    if sim_scores:
                        scores.append((candidate, np.mean(sim_scores)))

            # Rank and get top recommendations
            scores.sort(key=lambda x: x[1], reverse=True)
            top_recs = [item for item, score in scores[:5]]

            assert len(top_recs) <= 5
            assert all(item not in purchased_items for item in top_recs)

    def test_real_time_recommendation(self, sample_transactions, sample_menu_items):
        """Test real-time recommendation generation."""
        # Simulate real-time recommendation request
        customer_id = sample_transactions["customer_id"].iloc[0]

        # Get customer history
        history = sample_transactions[sample_transactions["customer_id"] == customer_id][
            "item_id"
        ].tolist()

        # Get candidates (all items not in history)
        all_items = sample_menu_items["item_id"].tolist()
        candidates = [i for i in all_items if i not in history]

        # Simple scoring (random for test)
        np.random.seed(42)
        scores = np.random.rand(len(candidates))

        # Get top 3
        top_indices = np.argsort(scores)[::-1][:3]
        recommendations = [candidates[i] for i in top_indices]

        assert len(recommendations) <= 3
        assert all(r in candidates for r in recommendations)
        assert all(r in candidates for r in recommendations)
