"""
RFM (Recency, Frequency, Monetary) Segmentation.

RFM analysis is a customer segmentation technique that uses three metrics:
- Recency: How recently a customer made a purchase
- Frequency: How often they purchase
- Monetary: How much they spend

This module provides tools to compute RFM scores and segment customers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ml.utils.data_utils import load_transactions

logger = logging.getLogger(__name__)


class RFMAnalyzer:
    """
    RFM Analysis for customer segmentation.

    Computes RFM metrics and assigns customers to segments based on
    their purchasing behavior.

    Attributes:
        analysis_date: Reference date for recency calculation.
        rfm_df: DataFrame with RFM metrics.
        segment_labels: Mapping of RFM scores to segment names.

    Example:
        >>> analyzer = RFMAnalyzer()
        >>> rfm_df = analyzer.fit(transactions_df)
        >>> segments = analyzer.get_segments()
    """

    # Segment labels based on RFM scores
    SEGMENT_MAP = {
        "Champions": {"R": [4, 5], "F": [4, 5], "M": [4, 5]},
        "Loyal Customers": {"R": [3, 4, 5], "F": [3, 4, 5], "M": [3, 4, 5]},
        "Potential Loyalists": {"R": [4, 5], "F": [2, 3], "M": [2, 3]},
        "New Customers": {"R": [4, 5], "F": [1], "M": [1, 2]},
        "Promising": {"R": [3, 4], "F": [1, 2], "M": [1, 2]},
        "Need Attention": {"R": [2, 3], "F": [2, 3], "M": [2, 3]},
        "About to Sleep": {"R": [2, 3], "F": [1, 2], "M": [1, 2]},
        "At Risk": {"R": [1, 2], "F": [3, 4, 5], "M": [3, 4, 5]},
        "Can't Lose Them": {"R": [1], "F": [4, 5], "M": [4, 5]},
        "Hibernating": {"R": [1, 2], "F": [1, 2], "M": [1, 2]},
        "Lost": {"R": [1], "F": [1], "M": [1]},
    }

    def __init__(
        self,
        analysis_date: datetime | None = None,
        recency_bins: int = 5,
        frequency_bins: int = 5,
        monetary_bins: int = 5,
    ) -> None:
        """
        Initialize the RFM Analyzer.

        Args:
            analysis_date: Reference date for recency. Defaults to today.
            recency_bins: Number of bins for recency scores (default 5).
            frequency_bins: Number of bins for frequency scores.
            monetary_bins: Number of bins for monetary scores.
        """
        self.analysis_date = analysis_date or datetime.now()
        self.recency_bins = recency_bins
        self.frequency_bins = frequency_bins
        self.monetary_bins = monetary_bins

        self.rfm_df: pd.DataFrame | None = None
        self.quantiles: dict[str, pd.Series] = {}

    def fit(
        self,
        transactions_df: pd.DataFrame,
        customer_id_col: str = "customer_id",
        date_col: str = "timestamp",
        amount_col: str = "total_price",
    ) -> pd.DataFrame:
        """
        Compute RFM metrics from transaction data.

        Args:
            transactions_df: Transaction DataFrame.
            customer_id_col: Column name for customer ID.
            date_col: Column name for transaction date.
            amount_col: Column name for transaction amount.

        Returns:
            DataFrame with RFM metrics and scores.
        """
        df = transactions_df.copy()

        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter to customers with IDs
        df = df[df[customer_id_col].notna()]

        # Calculate RFM metrics
        rfm = (
            df.groupby(customer_id_col)
            .agg(
                {
                    date_col: lambda x: (self.analysis_date - x.max()).days,  # Recency
                    customer_id_col: "count",  # Frequency (using customer_id as proxy)
                    amount_col: "sum",  # Monetary
                }
            )
            .reset_index()
        )

        # Rename columns
        rfm.columns = [customer_id_col, "recency", "frequency", "monetary"]

        # Fix frequency (use transaction count)
        freq = df.groupby(customer_id_col).size().reset_index(name="frequency")
        rfm["frequency"] = freq["frequency"]

        # Calculate RFM scores (1-5)
        # Recency: lower is better, so reverse the scoring
        rfm["R"] = pd.qcut(
            rfm["recency"],
            q=self.recency_bins,
            labels=range(self.recency_bins, 0, -1),
            duplicates="drop",
        ).astype(int)

        # Frequency: higher is better
        rfm["F"] = pd.qcut(
            rfm["frequency"].rank(method="first"),
            q=self.frequency_bins,
            labels=range(1, self.frequency_bins + 1),
            duplicates="drop",
        ).astype(int)

        # Monetary: higher is better
        rfm["M"] = pd.qcut(
            rfm["monetary"].rank(method="first"),
            q=self.monetary_bins,
            labels=range(1, self.monetary_bins + 1),
            duplicates="drop",
        ).astype(int)

        # Calculate RFM Score (concatenated)
        rfm["RFM_Score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)

        # Calculate total score
        rfm["RFM_Total"] = rfm["R"] + rfm["F"] + rfm["M"]

        # Assign segments
        rfm["Segment"] = rfm.apply(self._assign_segment, axis=1)

        self.rfm_df = rfm

        logger.info(f"Computed RFM metrics for {len(rfm)} customers")

        return rfm

    def _assign_segment(self, row: pd.Series) -> str:
        """Assign a segment based on RFM scores."""
        r, f, m = row["R"], row["F"], row["M"]

        # Check each segment in priority order
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f in [2, 3] and m in [2, 3]:
            return "Potential Loyalists"
        elif r >= 4 and f == 1:
            return "New Customers"
        elif r in [3, 4] and f <= 2 and m <= 2:
            return "Promising"
        elif r in [2, 3] and f in [2, 3] and m in [2, 3]:
            return "Need Attention"
        elif r in [2, 3] and f <= 2:
            return "About to Sleep"
        elif r <= 2 and f >= 3:
            if m >= 4:
                return "Can't Lose Them"
            return "At Risk"
        elif r <= 2 and f <= 2:
            if r == 1 and f == 1:
                return "Lost"
            return "Hibernating"
        else:
            return "Other"

    def get_segments(self) -> pd.DataFrame:
        """
        Get segment summary statistics.

        Returns:
            DataFrame with segment profiles.
        """
        if self.rfm_df is None:
            raise ValueError("Must call fit() first")

        segments = (
            self.rfm_df.groupby("Segment")
            .agg(
                {
                    "customer_id": "count",
                    "recency": "mean",
                    "frequency": "mean",
                    "monetary": ["mean", "sum"],
                    "RFM_Total": "mean",
                }
            )
            .round(2)
        )

        segments.columns = [
            "customer_count",
            "avg_recency_days",
            "avg_frequency",
            "avg_monetary",
            "total_monetary",
            "avg_rfm_score",
        ]

        # Calculate percentages
        total_customers = segments["customer_count"].sum()
        total_revenue = segments["total_monetary"].sum()

        segments["customer_pct"] = (segments["customer_count"] / total_customers * 100).round(1)
        segments["revenue_pct"] = (segments["total_monetary"] / total_revenue * 100).round(1)

        return segments.sort_values("avg_rfm_score", ascending=False)

    def get_segment_recommendations(self) -> dict[str, dict[str, str]]:
        """
        Get marketing recommendations for each segment.

        Returns:
            Dictionary with recommendations per segment.
        """
        recommendations = {
            "Champions": {
                "strategy": "Reward and retain",
                "actions": [
                    "Offer exclusive loyalty rewards",
                    "Early access to new menu items",
                    "Personalized VIP experiences",
                    "Ask for referrals and reviews",
                ],
                "priority": "High - Maintain relationship",
            },
            "Loyal Customers": {
                "strategy": "Upsell and engage",
                "actions": [
                    "Recommend premium items",
                    "Loyalty program upgrades",
                    "Birthday/anniversary specials",
                    "Feedback surveys",
                ],
                "priority": "High - Increase share of wallet",
            },
            "Potential Loyalists": {
                "strategy": "Convert to loyal",
                "actions": [
                    "Membership offers",
                    "Cross-sell related items",
                    "Personalized recommendations",
                    "Engagement campaigns",
                ],
                "priority": "High - Build loyalty",
            },
            "New Customers": {
                "strategy": "Onboard and nurture",
                "actions": [
                    "Welcome offers",
                    "Introduction to menu variety",
                    "First-time buyer discounts",
                    "Sign up for loyalty program",
                ],
                "priority": "Medium - Establish relationship",
            },
            "Promising": {
                "strategy": "Encourage more visits",
                "actions": [
                    "Time-limited promotions",
                    "Bundle deals",
                    "Occasion-based marketing",
                ],
                "priority": "Medium - Increase frequency",
            },
            "Need Attention": {
                "strategy": "Reactivate",
                "actions": [
                    "Win-back campaigns",
                    "Special comeback offers",
                    "Ask for feedback",
                ],
                "priority": "Medium - Prevent churn",
            },
            "About to Sleep": {
                "strategy": "Reactivate urgently",
                "actions": [
                    "Limited-time offers",
                    "Reminder campaigns",
                    "We miss you messaging",
                ],
                "priority": "High - Immediate action",
            },
            "At Risk": {
                "strategy": "Win back",
                "actions": [
                    "Personal outreach",
                    "Significant discounts",
                    "Survey for feedback",
                ],
                "priority": "High - Urgent retention",
            },
            "Can't Lose Them": {
                "strategy": "Win back urgently",
                "actions": [
                    "Immediate personal contact",
                    "Special recovery offers",
                    "Understand issues",
                ],
                "priority": "Critical - High value at risk",
            },
            "Hibernating": {
                "strategy": "Reawaken",
                "actions": [
                    "Reactivation campaign",
                    "New menu highlights",
                    "Special comeback offer",
                ],
                "priority": "Low - Test reactivation",
            },
            "Lost": {
                "strategy": "Attempt win-back",
                "actions": [
                    "Last resort offers",
                    "Brand awareness campaigns",
                    "Low-cost reactivation attempts",
                ],
                "priority": "Low - Accept potential loss",
            },
        }

        return recommendations

    def get_customer_segment(self, customer_id: str) -> dict | None:
        """
        Get segment info for a specific customer.

        Args:
            customer_id: Customer identifier.

        Returns:
            Customer segment information or None.
        """
        if self.rfm_df is None:
            raise ValueError("Must call fit() first")

        customer = self.rfm_df[self.rfm_df["customer_id"] == customer_id]

        if len(customer) == 0:
            return None

        row = customer.iloc[0]
        recommendations = self.get_segment_recommendations()

        return {
            "customer_id": customer_id,
            "segment": row["Segment"],
            "rfm_score": row["RFM_Score"],
            "recency_days": row["recency"],
            "frequency": row["frequency"],
            "monetary": row["monetary"],
            "recommendations": recommendations.get(row["Segment"], {}),
        }


def compute_rfm_features(
    transactions_df: pd.DataFrame | None = None,
    analysis_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Compute RFM features for all customers.

    Args:
        transactions_df: Transaction data. If None, loads from default path.
        analysis_date: Reference date for recency calculation.

    Returns:
        DataFrame with RFM metrics and segments.
    """
    if transactions_df is None:
        transactions_df = load_transactions()

    analyzer = RFMAnalyzer(analysis_date=analysis_date)
    return analyzer.fit(transactions_df)


def segment_customers_rfm(
    transactions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform complete RFM segmentation.

    Args:
        transactions_df: Transaction data.

    Returns:
        Tuple of (customer RFM DataFrame, segment summary DataFrame).
    """
    if transactions_df is None:
        transactions_df = load_transactions()

    analyzer = RFMAnalyzer()
    rfm_df = analyzer.fit(transactions_df)
    segment_summary = analyzer.get_segments()

    return rfm_df, segment_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    transactions = load_transactions()

    analyzer = RFMAnalyzer()
    rfm = analyzer.fit(transactions)

    print("\nRFM Segment Summary:")
    print(analyzer.get_segments())

    print("\nSample customers by segment:")
    for segment in rfm["Segment"].unique():
        sample = rfm[rfm["Segment"] == segment].head(2)
        print(f"\n{segment}:")
        print(sample[["customer_id", "recency", "frequency", "monetary", "RFM_Score"]])
