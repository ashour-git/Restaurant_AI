"""
Data Validation Module using Pandera.

Provides schema definitions and validation utilities for ensuring
data quality in ML pipelines.
"""

import logging
from typing import Any

import pandas as pd
import pandera as pa
from pandera.typing import Series

logger = logging.getLogger(__name__)


# ============================================
# Schema Definitions
# ============================================


class TransactionSchema(pa.DataFrameModel):
    """Schema for transaction data."""

    transaction_id: Series[str] = pa.Field(
        nullable=False, unique=True, description="Unique transaction identifier"
    )
    timestamp: Series[pd.Timestamp] = pa.Field(nullable=False, description="Transaction timestamp")
    customer_id: Series[str] = pa.Field(nullable=True, description="Customer identifier")
    item_id: Series[str] = pa.Field(nullable=False, description="Menu item identifier")
    item_name: Series[str] = pa.Field(nullable=False, description="Menu item name")
    quantity: Series[int] = pa.Field(ge=1, le=100, nullable=False, description="Quantity ordered")
    unit_price: Series[float] = pa.Field(
        ge=0, le=1000, nullable=False, description="Price per unit"
    )
    total_price: Series[float] = pa.Field(
        ge=0, nullable=False, description="Total transaction price"
    )

    class Config:
        strict = True
        coerce = True


class MenuItemSchema(pa.DataFrameModel):
    """Schema for menu item data."""

    item_id: Series[str] = pa.Field(
        nullable=False, unique=True, description="Unique item identifier"
    )
    item_name: Series[str] = pa.Field(
        nullable=False, str_length={"min_value": 1, "max_value": 200}, description="Item name"
    )
    category: Series[str] = pa.Field(nullable=False, description="Item category")
    subcategory: Series[str] = pa.Field(nullable=True, description="Item subcategory")
    price: Series[float] = pa.Field(ge=0, le=500, nullable=False, description="Item price")
    cost: Series[float] = pa.Field(ge=0, nullable=True, description="Item cost")
    prep_time_minutes: Series[int] = pa.Field(
        ge=0, le=180, nullable=True, description="Preparation time in minutes"
    )

    class Config:
        strict = False
        coerce = True


class CustomerSchema(pa.DataFrameModel):
    """Schema for customer data."""

    customer_id: Series[str] = pa.Field(
        nullable=False, unique=True, description="Unique customer identifier"
    )
    first_name: Series[str] = pa.Field(nullable=False, description="Customer first name")
    last_name: Series[str] = pa.Field(nullable=False, description="Customer last name")
    email: Series[str] = pa.Field(nullable=True, description="Customer email")
    phone: Series[str] = pa.Field(nullable=True, description="Customer phone")
    registration_date: Series[pd.Timestamp] = pa.Field(
        nullable=False, description="Registration date"
    )
    total_orders: Series[int] = pa.Field(ge=0, nullable=True, description="Total number of orders")
    total_spent: Series[float] = pa.Field(ge=0, nullable=True, description="Total amount spent")

    class Config:
        strict = False
        coerce = True


class DailySalesSchema(pa.DataFrameModel):
    """Schema for daily sales aggregates."""

    date: Series[pd.Timestamp] = pa.Field(nullable=False, description="Date")
    total_revenue: Series[float] = pa.Field(ge=0, nullable=False, description="Total daily revenue")
    total_orders: Series[int] = pa.Field(ge=0, nullable=False, description="Total number of orders")
    total_items: Series[int] = pa.Field(ge=0, nullable=False, description="Total items sold")

    class Config:
        strict = False
        coerce = True


class CustomerFeaturesSchema(pa.DataFrameModel):
    """Schema for customer features used in ML models."""

    customer_id: Series[str] = pa.Field(
        nullable=False, unique=True, description="Customer identifier"
    )
    recency_days: Series[int] = pa.Field(
        ge=0, nullable=False, description="Days since last purchase"
    )
    frequency: Series[int] = pa.Field(ge=1, nullable=False, description="Total number of purchases")
    monetary: Series[float] = pa.Field(ge=0, nullable=False, description="Total monetary value")
    avg_order_value: Series[float] = pa.Field(
        ge=0, nullable=False, description="Average order value"
    )

    class Config:
        strict = False
        coerce = True


# ============================================
# Data Validator Class
# ============================================


class DataValidator:
    """
    Data validation utility for ML pipelines.

    Provides methods to validate DataFrames against predefined schemas
    and log validation results.

    Example:
        >>> validator = DataValidator()
        >>> is_valid, errors = validator.validate(df, "transactions")
        >>> if not is_valid:
        ...     print(f"Validation errors: {errors}")
    """

    SCHEMAS = {
        "transactions": TransactionSchema,
        "menu_items": MenuItemSchema,
        "customers": CustomerSchema,
        "daily_sales": DailySalesSchema,
        "customer_features": CustomerFeaturesSchema,
    }

    def __init__(self, raise_on_error: bool = False) -> None:
        """
        Initialize the data validator.

        Args:
            raise_on_error: If True, raise exceptions on validation failure.
        """
        self.raise_on_error = raise_on_error
        self.validation_results: list[dict[str, Any]] = []

    def validate(
        self,
        df: pd.DataFrame,
        schema_name: str,
        sample_size: int | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate a DataFrame against a schema.

        Args:
            df: DataFrame to validate.
            schema_name: Name of the schema to use.
            sample_size: If provided, validate a sample of the data.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if schema_name not in self.SCHEMAS:
            return False, f"Unknown schema: {schema_name}"

        schema = self.SCHEMAS[schema_name]

        # Sample if requested
        if sample_size and len(df) > sample_size:
            df_to_validate = df.sample(sample_size, random_state=42)
        else:
            df_to_validate = df

        try:
            schema.validate(df_to_validate)

            result = {
                "schema": schema_name,
                "is_valid": True,
                "rows_validated": len(df_to_validate),
                "timestamp": pd.Timestamp.now(),
            }
            self.validation_results.append(result)

            logger.info(f"Validation passed for schema '{schema_name}'")
            return True, None

        except pa.errors.SchemaError as e:
            error_msg = str(e)

            result = {
                "schema": schema_name,
                "is_valid": False,
                "error": error_msg,
                "rows_validated": len(df_to_validate),
                "timestamp": pd.Timestamp.now(),
            }
            self.validation_results.append(result)

            logger.warning(f"Validation failed for schema '{schema_name}': {error_msg}")

            if self.raise_on_error:
                raise

            return False, error_msg

    def validate_custom(
        self,
        df: pd.DataFrame,
        checks: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """
        Run custom validation checks on a DataFrame.

        Args:
            df: DataFrame to validate.
            checks: List of check definitions.
                Each check should have:
                - column: Column name
                - check: Check type ("not_null", "unique", "range", "regex")
                - params: Check parameters

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        for check in checks:
            column = check.get("column")
            check_type = check.get("check")
            params = check.get("params", {})

            if column not in df.columns:
                errors.append(f"Column '{column}' not found")
                continue

            if check_type == "not_null":
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{column}' has {null_count} null values")

            elif check_type == "unique":
                dup_count = df[column].duplicated().sum()
                if dup_count > 0:
                    errors.append(f"Column '{column}' has {dup_count} duplicates")

            elif check_type == "range":
                min_val = params.get("min")
                max_val = params.get("max")
                if min_val is not None and df[column].min() < min_val:
                    errors.append(f"Column '{column}' has values below {min_val}")
                if max_val is not None and df[column].max() > max_val:
                    errors.append(f"Column '{column}' has values above {max_val}")

            elif check_type == "in_set":
                valid_values = set(params.get("values", []))
                invalid = set(df[column].unique()) - valid_values
                if invalid:
                    errors.append(f"Column '{column}' has invalid values: {invalid}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_data_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate a data quality profile for a DataFrame.

        Args:
            df: DataFrame to profile.

        Returns:
            Dictionary with data quality metrics.
        """
        profile = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "columns": {},
        }

        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": float(df[col].isnull().mean() * 100),
                "unique_count": int(df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update(
                    {
                        "min": float(df[col].min()) if not df[col].isnull().all() else None,
                        "max": float(df[col].max()) if not df[col].isnull().all() else None,
                        "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                        "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    }
                )

            profile["columns"][col] = col_profile

        return profile

    def get_validation_report(self) -> pd.DataFrame:
        """
        Get a report of all validation runs.

        Returns:
            DataFrame with validation history.
        """
        if not self.validation_results:
            return pd.DataFrame()

        return pd.DataFrame(self.validation_results)


# Convenience function


def validate_dataframe(
    df: pd.DataFrame,
    schema_name: str,
    raise_on_error: bool = False,
) -> tuple[bool, str | None]:
    """
    Validate a DataFrame against a named schema.

    Args:
        df: DataFrame to validate.
        schema_name: Name of the schema.
        raise_on_error: If True, raise on validation failure.

    Returns:
        Tuple of (is_valid, error_message).
    """
    validator = DataValidator(raise_on_error=raise_on_error)
    return validator.validate(df, schema_name)
