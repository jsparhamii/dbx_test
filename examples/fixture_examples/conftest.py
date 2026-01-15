"""
Example conftest.py demonstrating fixture usage.

This shows how teams can set up their own test projects using
the dbx_test fixture architecture.
"""

import pytest
import os
from pathlib import Path

# Import dbx_test fixtures
from dbx_test.fixtures import (
    spark_session,
    spark_config,
    dbutils,
    notebook_context,
    notebook_runner,
    sample_dataframe,
    temp_dbfs_path,
)
from dbx_test.fixtures.spark import SparkConfig
from dbx_test.fixtures.databricks import DatabricksConfig


# ============================================================================
# Project-specific fixture customization
# ============================================================================

@pytest.fixture(scope="session")
def spark_config() -> SparkConfig:
    """Customize Spark configuration for this project."""
    return SparkConfig(
        app_name="my_project_tests",
        master="local[4]",  # Use 4 cores locally
        config={
            "spark.sql.shuffle.partitions": "4",
            "spark.default.parallelism": "4",
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        },
        # Enable Databricks Connect when configured
        use_databricks_connect=os.environ.get("USE_DATABRICKS_CONNECT", "").lower() == "true",
        databricks_profile=os.environ.get("DATABRICKS_PROFILE", "default"),
    )


@pytest.fixture(scope="session")
def project_config():
    """Project-specific configuration."""
    return {
        "project_name": "my_data_project",
        "default_database": "test_db",
        "input_path": "/data/input",
        "output_path": "/data/output",
    }


# ============================================================================
# Project-specific fixtures
# ============================================================================

@pytest.fixture(scope="function")
def customer_data(spark_session):
    """Create sample customer data for tests."""
    return spark_session.createDataFrame([
        (1, "Alice", "alice@example.com", "premium"),
        (2, "Bob", "bob@example.com", "standard"),
        (3, "Charlie", "charlie@example.com", "premium"),
        (4, "Diana", "diana@example.com", "standard"),
    ], ["customer_id", "name", "email", "tier"])


@pytest.fixture(scope="function")
def order_data(spark_session):
    """Create sample order data for tests."""
    return spark_session.createDataFrame([
        (101, 1, 99.99, "2024-01-15"),
        (102, 2, 149.99, "2024-01-16"),
        (103, 1, 29.99, "2024-01-17"),
        (104, 3, 199.99, "2024-01-18"),
    ], ["order_id", "customer_id", "amount", "order_date"])


@pytest.fixture(scope="function")
def customer_orders_view(spark_session, customer_data, order_data):
    """Create a joined customer-orders view for testing."""
    customer_data.createOrReplaceTempView("customers")
    order_data.createOrReplaceTempView("orders")
    
    return spark_session.sql("""
        SELECT 
            c.customer_id,
            c.name,
            c.tier,
            o.order_id,
            o.amount,
            o.order_date
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
    """)


# ============================================================================
# Custom markers for this project
# ============================================================================

def pytest_configure(config):
    """Add project-specific markers."""
    config.addinivalue_line(
        "markers", "etl: mark test as ETL pipeline test"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as ML model test"
    )
    config.addinivalue_line(
        "markers", "dashboard: mark test as dashboard/reporting test"
    )

