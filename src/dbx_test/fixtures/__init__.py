"""
dbx_test pytest fixtures module.

This module provides production-ready pytest fixtures for testing Databricks notebooks
and applications. Fixtures are designed to work seamlessly in local development,
CI pipelines, and Databricks environments.

Usage:
    # In your conftest.py or test file:
    from dbx_test.fixtures import *

    # Or import specific fixtures:
    from dbx_test.fixtures import spark_session, databricks_client

Architecture:
    - spark.py: SparkSession lifecycle management
    - databricks.py: Databricks Connect and workspace client fixtures
    - paths.py: DBFS and workspace temporary path management
    - data.py: Test data generation and management
    - notebook.py: Notebook execution and result collection
"""

from dbx_test.fixtures.spark import (
    spark_session,
    spark_context,
    spark_config,
    local_spark,
)

from dbx_test.fixtures.databricks import (
    databricks_client,
    databricks_config,
    workspace_client,
    dbutils,
    is_databricks_runtime,
)

from dbx_test.fixtures.paths import (
    temp_dbfs_path,
    temp_workspace_path,
    temp_volume_path,
    temp_local_path,
    cleanup_paths,
    path_config,
    path_manager,
)

from dbx_test.fixtures.data import (
    sample_dataframe,
    test_table,
    test_catalog_schema,
    parquet_file,
    delta_table,
    data_config,
    data_manager,
    create_df,
)

from dbx_test.fixtures.notebook import (
    notebook_context,
    notebook_runner,
    notebook_test_result,
)

from dbx_test.fixtures.injection import (
    notebook_fixture,
    register_fixture,
    get_global_registry,
    FixtureRegistry,
    FixtureScope,
)

__all__ = [
    # Spark fixtures
    "spark_session",
    "spark_context",
    "spark_config",
    "local_spark",
    # Databricks fixtures
    "databricks_client",
    "databricks_config",
    "workspace_client",
    "dbutils",
    "is_databricks_runtime",
    # Path fixtures
    "temp_dbfs_path",
    "temp_workspace_path",
    "temp_volume_path",
    "temp_local_path",
    "cleanup_paths",
    "path_config",
    "path_manager",
    # Data fixtures
    "sample_dataframe",
    "test_table",
    "test_catalog_schema",
    "parquet_file",
    "delta_table",
    "data_config",
    "data_manager",
    "create_df",
    # Notebook fixtures
    "notebook_context",
    "notebook_runner",
    "notebook_test_result",
    # Fixture injection
    "notebook_fixture",
    "register_fixture",
    "get_global_registry",
    "FixtureRegistry",
    "FixtureScope",
]

