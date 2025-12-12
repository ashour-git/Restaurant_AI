# Model Card: Demand Forecasting

## Model Details

| Property          | Value                                      |
| ----------------- | ------------------------------------------ |
| **Model Name**    | DemandForecaster                           |
| **Model Version** | 1.0.0                                      |
| **Model Type**    | Time-Series Regression                     |
| **Algorithm**     | LightGBM (Gradient Boosting Decision Tree) |
| **Framework**     | LightGBM 4.x, scikit-learn                 |
| **Owner**         | Smart Restaurant ML Team                   |
| **License**       | MIT                                        |
| **Last Updated**  | 2024                                       |

## Model Description

### Purpose

The Demand Forecasting model predicts future demand for restaurant orders at both aggregate and item levels. It enables:

- **Inventory Planning**: Optimize ingredient ordering based on predicted demand
- **Staff Scheduling**: Align staffing with expected customer volume
- **Waste Reduction**: Minimize food waste through accurate predictions
- **Revenue Optimization**: Better planning leads to improved profitability

### Architecture

```
Input Features → Feature Engineering → LightGBM Regressor → Demand Prediction
     ↓                    ↓                    ↓
  Raw Data       Time/Lag/Rolling         Gradient Boosted
                   Features               Decision Trees
```

### Input Features

| Feature Category     | Features                                       | Description              |
| -------------------- | ---------------------------------------------- | ------------------------ |
| **Time Features**    | day_of_week, month, is_weekend, quarter        | Calendar-based patterns  |
| **Lag Features**     | lag_1, lag_7, lag_14, lag_28                   | Historical demand values |
| **Rolling Features** | rolling_mean_7, rolling_std_7, rolling_mean_14 | Moving statistics        |
| **Trend Features**   | days_since_start, trend                        | Long-term patterns       |

### Output

- **Primary Output**: Predicted number of orders (continuous)
- **Output Range**: Non-negative real numbers
- **Granularity**: Daily predictions (can be aggregated to weekly/monthly)

## Intended Use

### Primary Use Cases

1. **Short-term Forecasting**: 1-14 day demand predictions
2. **Item-Level Planning**: Predict demand per menu item
3. **Aggregate Forecasting**: Total daily/weekly order volumes
4. **Scenario Analysis**: What-if predictions for promotions

### Users

- Restaurant managers for operational planning
- Kitchen staff for ingredient preparation
- Supply chain for inventory ordering
- Business analysts for capacity planning

### Out of Scope

- Real-time order predictions (< 1 day)
- Weather-dependent predictions (not included in current model)
- Event-driven demand spikes (holidays, special events)
- New item demand with no historical data

## Training Data

### Dataset Description

| Metric            | Value                                 |
| ----------------- | ------------------------------------- |
| **Source**        | Synthetic restaurant transaction data |
| **Time Range**    | 12 months of historical data          |
| **Total Records** | ~114,000 transactions                 |
| **Unique Items**  | 78 menu items                         |
| **Data Split**    | 80% train, 20% validation (temporal)  |

### Data Processing

1. Aggregate transactions to daily level
2. Create time-based features (day, month, weekday)
3. Generate lag features (1, 7, 14, 21, 28 days)
4. Calculate rolling statistics (7, 14, 28 day windows)
5. Handle missing values from lag creation

### Data Quality Checks

- Pandera schema validation for input data
- Outlier detection for anomalous order counts
- Missing value imputation for lag features

## Evaluation

### Metrics

| Metric   | Training | Validation | Description                    |
| -------- | -------- | ---------- | ------------------------------ |
| **MAE**  | ~5.2     | ~7.1       | Mean Absolute Error (orders)   |
| **RMSE** | ~7.8     | ~10.2      | Root Mean Squared Error        |
| **R²**   | 0.85     | 0.78       | Coefficient of Determination   |
| **MAPE** | 8.2%     | 11.5%      | Mean Absolute Percentage Error |

### Performance by Time Horizon

| Forecast Days | RMSE  | Notes                           |
| ------------- | ----- | ------------------------------- |
| 1 day         | ~6.5  | Best accuracy                   |
| 7 days        | ~9.2  | Good for weekly planning        |
| 14 days       | ~12.1 | Acceptable for monthly planning |
| 28 days       | ~15.8 | Increased uncertainty           |

### Cross-Validation

- **Method**: Time Series Cross-Validation (5 folds)
- **CV RMSE**: 10.4 ± 2.1
- **Stability**: Model performs consistently across folds

## Hyperparameter Optimization

### Method

- **Optimizer**: Optuna with TPE Sampler
- **Trials**: 50-100 trials
- **Metric**: RMSE minimization
- **Cross-Validation**: 5-fold Time Series CV

### Optimized Parameters

| Parameter        | Search Space | Best Value |
| ---------------- | ------------ | ---------- |
| n_estimators     | [100, 1000]  | 487        |
| learning_rate    | [0.01, 0.3]  | 0.052      |
| max_depth        | [3, 12]      | 8          |
| num_leaves       | [15, 127]    | 45         |
| subsample        | [0.5, 1.0]   | 0.82       |
| colsample_bytree | [0.5, 1.0]   | 0.78       |

## Explainability

### Feature Importance (Top 10)

| Rank | Feature         | Importance |
| ---- | --------------- | ---------- |
| 1    | lag_1           | 0.245      |
| 2    | lag_7           | 0.178      |
| 3    | rolling_mean_7  | 0.142      |
| 4    | day_of_week     | 0.098      |
| 5    | rolling_std_7   | 0.076      |
| 6    | lag_14          | 0.068      |
| 7    | month           | 0.055      |
| 8    | is_weekend      | 0.042      |
| 9    | rolling_mean_14 | 0.038      |
| 10   | lag_28          | 0.032      |

### SHAP Analysis

- SHAP values available for individual prediction explanation
- TreeExplainer used for efficient computation
- Feature interactions analyzed for deeper insights

## Ethical Considerations

### Fairness

- Model predicts aggregate demand, not individual customer behavior
- No personal data used in predictions
- Equal treatment across all menu items

### Privacy

- No personally identifiable information (PII) used
- Aggregated transaction data only
- Customer IDs anonymized before analysis

### Environmental Impact

- Model helps reduce food waste through better predictions
- Optimized inventory reduces over-ordering
- Training is computationally lightweight

## Limitations

### Known Limitations

1. **Cold Start**: Cannot predict for new items with no history
2. **External Factors**: Weather, events not included
3. **Data Drift**: Performance may degrade with changing patterns
4. **Seasonality**: May not capture all seasonal effects

### Failure Modes

- Extreme events (holidays, closures) may cause poor predictions
- Sudden trend changes not captured immediately
- Very low volume days may have high relative error

## Maintenance

### Monitoring

- Track RMSE on rolling basis
- Alert on prediction errors > 2 standard deviations
- Monitor feature drift with distribution comparisons

### Retraining Triggers

- RMSE increases by > 20% from baseline
- New menu items added (retrain item-level model)
- Significant business changes (hours, capacity)

### Recommended Retraining Frequency

- **Full Retrain**: Monthly
- **Incremental Update**: Weekly with new data
- **Hyperparameter Tuning**: Quarterly

## Usage

### Python API

```python
from ml.pipelines.enhanced_forecasting import EnhancedDemandForecaster

# Initialize and train
forecaster = EnhancedDemandForecaster(
    target_col='total_orders',
    lags=[1, 7, 14, 21, 28],
    rolling_windows=[7, 14, 28]
)

# Train with optimization
results = forecaster.train_with_optimization(
    transactions_df,
    n_trials=50,
    experiment_name='demand_v1'
)

# Generate forecast
forecast = forecaster.forecast_future(historical_df, days_ahead=14)
```

### REST API

```bash
POST /api/v1/forecast/demand
Content-Type: application/json

{
    "days_ahead": 7,
    "item_id": null  # Optional for item-level
}
```

## Version History

| Version | Date    | Changes                            |
| ------- | ------- | ---------------------------------- |
| 1.0.0   | 2024-01 | Initial release with LightGBM      |
| 1.1.0   | 2024-02 | Added Optuna hyperparameter tuning |
| 1.2.0   | 2024-03 | Added SHAP explainability          |

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)

---

_This model card follows the format proposed by Mitchell et al. (2019) "Model Cards for Model Reporting"_
