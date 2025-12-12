# Model Card: Customer Lifetime Value Prediction

## Model Details

| Property          | Value                       |
| ----------------- | --------------------------- |
| **Model Name**    | CustomerLTV                 |
| **Model Version** | 1.0.0                       |
| **Model Type**    | Regression                  |
| **Algorithm**     | Gradient Boosting Regressor |
| **Framework**     | scikit-learn                |
| **Owner**         | Smart Restaurant ML Team    |
| **License**       | MIT                         |
| **Last Updated**  | 2024                        |

## Model Description

### Purpose

The Customer Lifetime Value (LTV) model predicts the future revenue a customer will generate, enabling:

- **Customer Prioritization**: Focus resources on high-value customers
- **Marketing Budget Allocation**: Optimize CAC vs LTV ratio
- **Segmentation**: Group customers by predicted value
- **Retention ROI**: Justify retention spend based on customer value

### Architecture

```
Customer Features → Feature Scaling → Gradient Boosting → LTV Prediction
       ↓                  ↓                  ↓
  RFM + Behavioral   StandardScaler    Ensemble of
    Features                         Decision Trees
```

### LTV Definition

**Predicted Lifetime Value** = Expected revenue from a customer over the next **365 days** (configurable prediction horizon).

### Input Features

| Feature Category | Features                                                      | Description              |
| ---------------- | ------------------------------------------------------------- | ------------------------ |
| **Recency**      | recency_days, recency_ratio                                   | Time since last purchase |
| **Frequency**    | total_transactions, orders_per_month, avg_days_between_orders | Purchase patterns        |
| **Monetary**     | total_revenue, avg_order_value, revenue_per_month             | Spending patterns        |
| **Tenure**       | tenure_days, purchase_span                                    | Customer lifetime        |
| **Variability**  | std_order_value, order_value_cv, order_value_range            | Consistency              |
| **Diversity**    | unique_items, item_diversity_ratio                            | Product exploration      |

### Output

- **Primary Output**: Predicted LTV in dollars (continuous)
- **LTV Segment**: Low, Below Average, Average, Above Average, High
- **Confidence**: Based on feature coverage and historical accuracy

## Intended Use

### Primary Use Cases

1. **Customer Valuation**: Assess worth of customer relationships
2. **Acquisition Targeting**: Set maximum CAC based on predicted LTV
3. **Retention Prioritization**: Focus on high-LTV customers at risk
4. **Segment Marketing**: Tailor strategies by value tier

### Users

- Marketing team for budget allocation
- Customer success for VIP identification
- Finance for customer asset valuation
- Product team for feature prioritization

### Out of Scope

- Real-time LTV updates
- LTV prediction for non-customers (prospects)
- Causal analysis of LTV drivers
- LTV guarantees or contractual obligations

## Training Data

### Dataset Description

| Metric                  | Value                          |
| ----------------------- | ------------------------------ |
| **Customers**           | ~5,000 unique customers        |
| **Transaction Records** | ~114,000                       |
| **Time Range**          | 12 months                      |
| **Holdout Period**      | 90 days for target calculation |
| **Data Split**          | 80% train, 20% test            |

### Target Variable Creation

```
Training Period: Days 1-275
Holdout Period: Days 276-365 (90 days)

Target = Sum of revenue in holdout period per customer
```

### Target Distribution

| Statistic  | Value               |
| ---------- | ------------------- |
| Mean LTV   | $245.32             |
| Median LTV | $178.50             |
| Std Dev    | $312.45             |
| Min        | $0.00               |
| Max        | $3,245.00           |
| Skewness   | 2.34 (right-skewed) |

## Evaluation

### Metrics

| Metric   | Value  | Description                    |
| -------- | ------ | ------------------------------ |
| **MAE**  | $42.15 | Mean Absolute Error            |
| **RMSE** | $68.92 | Root Mean Squared Error        |
| **R²**   | 0.72   | Coefficient of Determination   |
| **MAPE** | 28.3%  | Mean Absolute Percentage Error |

### Performance by LTV Segment

| Segment          | MAE    | MAPE  | Notes                  |
| ---------------- | ------ | ----- | ---------------------- |
| High (Top 20%)   | $85.40 | 15.2% | Best relative accuracy |
| Above Average    | $52.30 | 22.1% | Good accuracy          |
| Average          | $38.20 | 31.5% | Moderate accuracy      |
| Below Average    | $28.10 | 42.8% | Higher relative error  |
| Low (Bottom 20%) | $18.50 | 68.5% | Hardest to predict     |

### Cross-Validation

- **Method**: 5-Fold Cross-Validation
- **CV MAE**: $45.20 ± $8.35
- **CV R²**: 0.70 ± 0.05

## Explainability

### Feature Importance (Top 10)

| Rank | Feature                 | Importance |
| ---- | ----------------------- | ---------- |
| 1    | total_revenue           | 0.215      |
| 2    | orders_per_month        | 0.168      |
| 3    | avg_order_value         | 0.142      |
| 4    | tenure_days             | 0.095      |
| 5    | total_transactions      | 0.088      |
| 6    | recency_days            | 0.075      |
| 7    | revenue_per_month       | 0.062      |
| 8    | unique_items            | 0.048      |
| 9    | avg_days_between_orders | 0.038      |
| 10   | order_value_cv          | 0.028      |

### Interpretation

- **Historical spend** is the strongest predictor of future spend
- **Purchase frequency** indicates engagement level
- **Recency** affects likelihood of continued purchasing
- **Order consistency** (low CV) suggests reliable revenue

### Simple LTV Formula (Alternative)

For comparison, a simple formula-based LTV:

```
Simple LTV = Avg Order Value × Purchase Frequency × Expected Lifespan

Where:
- Purchase Frequency = Orders per month
- Expected Lifespan = 24 months (configurable)
```

## Ethical Considerations

### Fairness

- Model predicts based on behavior, not demographics
- All customers have equal opportunity for high LTV
- No discriminatory features used
- Regular audits for segment disparities

### Privacy

- Uses aggregated behavioral data only
- No PII in model features
- Customer IDs anonymized
- Compliant with privacy regulations

### Responsible Use

- Predictions inform, not determine, treatment
- Low-LTV customers still receive quality service
- Avoid self-fulfilling prophecies (reduced service → lower LTV)

## Limitations

### Known Limitations

1. **Historical Dependency**: Requires purchase history
2. **Assumption of Continuity**: Assumes patterns persist
3. **External Factors**: Doesn't account for competition, economy
4. **New Customers**: Limited accuracy with < 3 transactions

### Failure Modes

- Major life changes (relocation) not captured
- Business changes (menu, pricing) affect predictions
- Promotional periods may inflate predictions

### Uncertainty

- Higher uncertainty for customers with high variability
- Prediction intervals available for confidence assessment
- Recommend presenting ranges, not point estimates

## Maintenance

### Monitoring

- Track predicted vs actual LTV (after holdout period)
- Monitor MAE trends over time
- Alert on prediction distribution shifts

### Retraining Triggers

- MAE increases by > 25%
- R² drops below 0.60
- Significant business changes

### Recommended Frequency

- **Full Retrain**: Monthly with new holdout data
- **Feature Updates**: Quarterly review
- **Validation**: Continuous actual vs predicted tracking

## Usage

### Python API

```python
from ml.pipelines.customer_analytics.customer_ltv import CustomerLTV

# Initialize and train
ltv_model = CustomerLTV(prediction_horizon_days=365)
metrics = ltv_model.train(transactions_df, customers_df)

# Predict LTV
predictions = ltv_model.predict_for_transactions(transactions_df)

# Get high-value customers
high_value = ltv_model.get_high_value_customers(
    transactions_df,
    top_n=100
)

# Get LTV distribution
dist = ltv_model.get_ltv_distribution(transactions_df)

# Simple LTV calculation (no ML)
simple_ltv = ltv_model.calculate_simple_ltv(
    transactions_df,
    avg_lifespan_months=24
)
```

### REST API

```bash
GET /api/v1/analytics/ltv/customer/C001

Response:
{
    "customer_id": "C001",
    "predicted_ltv": 425.50,
    "ltv_segment": "High",
    "confidence": 0.85,
    "percentile": 92
}
```

### Batch Prediction

```bash
POST /api/v1/analytics/ltv/batch
Content-Type: application/json

{
    "customer_ids": ["C001", "C002", "C003"]
}
```

## Business Applications

### CAC:LTV Ratio

- **Target Ratio**: 1:3 or better
- **Maximum CAC**: LTV / 3
- **Example**: If LTV = $300, max CAC = $100

### Segmentation Strategy

| LTV Segment   | % Customers | % Revenue | Strategy                        |
| ------------- | ----------- | --------- | ------------------------------- |
| High          | 20%         | 55%       | VIP treatment, exclusive offers |
| Above Average | 20%         | 25%       | Loyalty programs, upselling     |
| Average       | 20%         | 12%       | Engagement campaigns            |
| Below Average | 20%         | 6%        | Reactivation offers             |
| Low           | 20%         | 2%        | Cost-effective touchpoints      |

## Version History

| Version | Date    | Changes                      |
| ------- | ------- | ---------------------------- |
| 1.0.0   | 2024-01 | Initial GBR model            |
| 1.1.0   | 2024-02 | Added simple LTV formula     |
| 1.2.0   | 2024-03 | Improved feature engineering |

## References

- [Customer Lifetime Value Models](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)
- [Gradient Boosting Documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [CLV in Practice](https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-three-building-blocks-of-customer-lifetime-value)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

---

_This model card follows the format proposed by Mitchell et al. (2019) "Model Cards for Model Reporting"_
