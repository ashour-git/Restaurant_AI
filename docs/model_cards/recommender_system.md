# Model Card: Hybrid Recommendation System

## Model Details

| Property          | Value                                                 |
| ----------------- | ----------------------------------------------------- |
| **Model Name**    | HybridRecommender                                     |
| **Model Version** | 1.0.0                                                 |
| **Model Type**    | Hybrid Recommendation (Collaborative + Content-Based) |
| **Algorithms**    | SVD, TF-IDF, Cosine Similarity                        |
| **Framework**     | scikit-learn, NumPy                                   |
| **Owner**         | Smart Restaurant ML Team                              |
| **License**       | MIT                                                   |
| **Last Updated**  | 2024                                                  |

## Model Description

### Purpose

The Hybrid Recommendation System provides personalized menu item recommendations to customers, enabling:

- **Personalized Ordering**: Suggest items based on past preferences
- **Discovery**: Help customers explore new items they may enjoy
- **Cross-Selling**: Recommend complementary items
- **Increased Revenue**: Higher average order value through relevant suggestions

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID RECOMMENDER                           │
├────────────────────────┬────────────────────────────────────────┤
│  Collaborative Filter  │        Content-Based Filter            │
│                        │                                        │
│  User-Item Matrix      │   TF-IDF on Item Descriptions          │
│         ↓              │              ↓                         │
│   SVD Decomposition    │     Cosine Similarity Matrix           │
│         ↓              │              ↓                         │
│  Similar Users/Items   │     Similar Items by Content           │
└────────────┬───────────┴──────────────────┬─────────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ↓
                  Weighted Combination
                            ↓
                  Final Recommendations
```

### Components

| Component                   | Algorithm                  | Purpose                              |
| --------------------------- | -------------------------- | ------------------------------------ |
| **Collaborative Filtering** | SVD (TruncatedSVD)         | Learn user preferences from behavior |
| **Content-Based**           | TF-IDF + Cosine Similarity | Find similar items by description    |
| **Popularity**              | Transaction frequency      | Fallback for cold start              |
| **Hybrid Weighting**        | Configurable blend         | Combine multiple signals             |

### Input Features

**Collaborative Filtering:**

- User-Item interaction matrix (purchases)
- Implicit feedback (quantity, frequency)

**Content-Based:**

- Item name
- Item description
- Category
- Tags/ingredients

### Output

- **Primary Output**: Ranked list of recommended items
- **Confidence Scores**: 0.0 - 1.0 for each recommendation
- **Explanation**: Reason for recommendation

## Intended Use

### Primary Use Cases

1. **Homepage Recommendations**: "Recommended for You"
2. **Item Page**: "Customers Also Ordered"
3. **Cart Page**: "Complete Your Order"
4. **New User**: Popular items until preferences learned

### Users

- Restaurant mobile app
- Online ordering website
- POS system for upselling suggestions
- Marketing for personalized campaigns

### Out of Scope

- Real-time contextual recommendations (time, weather)
- Group recommendations
- Dietary restriction filtering (handled separately)
- Price-based recommendations

## Training Data

### Dataset Description

| Metric                | Value                                 |
| --------------------- | ------------------------------------- |
| **Users (Customers)** | ~5,000                                |
| **Items (Menu)**      | 78                                    |
| **Transactions**      | ~114,000                              |
| **Sparsity**          | 85% (typical for recommender systems) |
| **Time Range**        | 12 months                             |

### Interaction Matrix

- Rows: Customers
- Columns: Menu items
- Values: Purchase frequency (normalized)

### Content Features

- Item descriptions tokenized and vectorized
- TF-IDF with n-grams (1-2)
- Stop words removed
- Category encoded

## Evaluation

### Metrics

| Metric          | Value | Description                           |
| --------------- | ----- | ------------------------------------- |
| **Precision@5** | 0.42  | Relevant items in top 5               |
| **Recall@5**    | 0.28  | Coverage of relevant items            |
| **NDCG@5**      | 0.51  | Normalized discounted cumulative gain |
| **MAP**         | 0.38  | Mean average precision                |
| **Coverage**    | 85%   | Items that can be recommended         |

### Evaluation Methodology

- **Train/Test Split**: Temporal (last 20% of transactions)
- **Leave-One-Out**: Holdout last purchase per user
- **Cross-Validation**: 5-fold user-based split

### Performance by User Type

| User Type               | Precision@5 | Notes             |
| ----------------------- | ----------- | ----------------- |
| Frequent (10+ orders)   | 0.52        | Best accuracy     |
| Regular (5-10 orders)   | 0.41        | Good accuracy     |
| Occasional (2-5 orders) | 0.35        | Moderate accuracy |
| New (1 order)           | 0.25        | Limited data      |

### A/B Testing Results (Simulated)

- **Click-Through Rate**: +18% vs random
- **Conversion Rate**: +12% vs popularity-based
- **Average Order Value**: +8% with recommendations

## Algorithm Details

### Collaborative Filtering (SVD)

```python
# Matrix factorization
U, Σ, V^T = SVD(User-Item Matrix)

# Reduced rank approximation (k=50)
User_factors = U[:, :k] @ diag(Σ[:k])
Item_factors = V[:k, :]

# Prediction
score(user, item) = User_factors[user] · Item_factors[:, item]
```

### Content-Based Filtering

```python
# TF-IDF vectorization
item_vectors = TF-IDF(item_descriptions)

# Similarity matrix
similarity = cosine_similarity(item_vectors)

# Recommendations based on liked items
recommendations = weighted_average(similarity[liked_items])
```

### Hybrid Combination

```python
# Weighted blend
final_score = (
    collaborative_weight * collab_score +
    content_weight * content_score +
    popularity_weight * popularity_score
)

# Default weights
collaborative_weight = 0.6
content_weight = 0.3
popularity_weight = 0.1
```

## Explainability

### Recommendation Reasons

| Reason Type       | Example                              |
| ----------------- | ------------------------------------ |
| **Similar Users** | "Customers like you also ordered..." |
| **Similar Items** | "Because you liked [item]..."        |
| **Popular**       | "Trending this week..."              |
| **Category**      | "More from [category]..."            |

### Similarity Explanation

For content-based recommendations, top contributing features:

- Shared category
- Common ingredients
- Similar description terms
- Price range alignment

## Ethical Considerations

### Fairness

- All items have equal opportunity to be recommended
- No demographic-based filtering
- Popularity bias mitigated with diversity

### Privacy

- User preferences aggregated and anonymized
- No PII used in recommendations
- Opt-out available for personalization

### Potential Issues

- Filter bubbles (addressed with exploration)
- Popularity bias (addressed with diversity)
- Cold start for new items (addressed with content features)

## Limitations

### Known Limitations

1. **Cold Start - New Users**: Limited recommendations until behavior observed
2. **Cold Start - New Items**: Relies on content similarity only
3. **Sparsity**: Most users haven't tried most items
4. **Implicit Feedback**: Only purchase data, no explicit ratings

### Failure Modes

- Very new users receive generic recommendations
- Seasonal items may not be recommended off-season
- Similar items may dominate (lack of diversity)

### Mitigation Strategies

- Exploration-exploitation trade-off (10% random)
- Diversity constraints (category spread)
- Recency weighting for temporal relevance

## Maintenance

### Monitoring

- Track recommendation acceptance rate
- Monitor item coverage metrics
- Alert on diversity degradation

### Retraining Triggers

- New menu items added
- Precision@5 drops below 0.35
- Monthly scheduled update

### Recommended Update Frequency

- **Full Retrain**: Weekly with new transaction data
- **Model Refresh**: Real-time user factor updates (optional)
- **Item Embeddings**: On menu changes

## Usage

### Python API

```python
from ml.pipelines.recommender import HybridRecommender

# Initialize and train
recommender = HybridRecommender(
    collaborative_weight=0.6,
    content_weight=0.3,
    n_factors=50
)
recommender.train(transactions_df, menu_items_df)

# Get recommendations for a user
recommendations = recommender.recommend(
    customer_id="C001",
    n_recommendations=5,
    exclude_purchased=True
)

# Get similar items
similar = recommender.similar_items(
    item_id="I001",
    n_similar=5
)
```

### REST API

```bash
GET /api/v1/recommendations/customer/C001?n=5

Response:
{
    "customer_id": "C001",
    "recommendations": [
        {
            "item_id": "I045",
            "name": "Grilled Salmon",
            "score": 0.89,
            "reason": "Based on your preferences"
        },
        {
            "item_id": "I023",
            "name": "Caesar Salad",
            "score": 0.76,
            "reason": "Pairs well with your order"
        }
    ]
}
```

## Performance Optimization

### Scalability

- Matrix factorization: O(n·m·k) where k << min(n,m)
- Precomputed similarity matrices for real-time serving
- Cached user factors updated incrementally

### Latency

- Recommendation generation: < 50ms
- Precomputation: Batch (offline)
- Memory: ~100MB for full model

## Version History

| Version | Date    | Changes                        |
| ------- | ------- | ------------------------------ |
| 1.0.0   | 2024-01 | Initial SVD + TF-IDF hybrid    |
| 1.1.0   | 2024-02 | Added popularity fallback      |
| 1.2.0   | 2024-03 | Improved diversity constraints |

## References

- [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [TF-IDF for Content-Based Filtering](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Hybrid Recommender Systems](https://dl.acm.org/doi/10.1145/371920.372071)
- [Recommender Systems Handbook](https://www.springer.com/gp/book/9780387858203)

---

_This model card follows the format proposed by Mitchell et al. (2019) "Model Cards for Model Reporting"_
