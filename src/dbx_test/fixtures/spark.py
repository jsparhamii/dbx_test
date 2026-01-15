"""
SparkSession lifecycle fixtures for pytest.

Provides session-scoped SparkSession management with support for:
- Local Spark (for unit tests)
- Databricks Connect (for integration tests)
- Databricks Runtime (when running in notebooks)

The fixtures automatically detect the execution environment and configure
Spark appropriately.
"""

import os
import pytest
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass, field


@dataclass
class SparkConfig:
    """Configuration for SparkSession creation.
    
    Attributes:
        app_name: Name for the Spark application
        master: Spark master URL (local, yarn, databricks)
        config: Additional Spark configuration options
        use_databricks_connect: Whether to use Databricks Connect
        databricks_cluster_id: Cluster ID for Databricks Connect
        databricks_profile: Databricks CLI profile name
    """
    app_name: str = "dbx_test"
    master: str = "local[*]"
    config: Dict[str, str] = field(default_factory=dict)
    use_databricks_connect: bool = False
    databricks_cluster_id: Optional[str] = None
    databricks_profile: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "SparkConfig":
        """Create SparkConfig from environment variables.
        
        Environment Variables:
            DBX_TEST_SPARK_MASTER: Spark master URL
            DBX_TEST_APP_NAME: Application name
            DBX_TEST_USE_CONNECT: Set to 'true' for Databricks Connect
            DBX_TEST_CLUSTER_ID: Databricks cluster ID
            DBX_TEST_PROFILE: Databricks CLI profile
            SPARK_*: Standard Spark configuration
        """
        config = {}
        
        # Collect SPARK_* environment variables
        for key, value in os.environ.items():
            if key.startswith("SPARK_CONF_"):
                spark_key = key[11:].replace("_", ".")
                config[spark_key] = value
        
        return cls(
            app_name=os.environ.get("DBX_TEST_APP_NAME", "dbx_test"),
            master=os.environ.get("DBX_TEST_SPARK_MASTER", "local[*]"),
            config=config,
            use_databricks_connect=os.environ.get("DBX_TEST_USE_CONNECT", "").lower() == "true",
            databricks_cluster_id=os.environ.get("DBX_TEST_CLUSTER_ID"),
            databricks_profile=os.environ.get("DBX_TEST_PROFILE"),
        )


def _is_databricks_runtime() -> bool:
    """Check if code is running in Databricks Runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _get_databricks_spark():
    """Get the existing SparkSession from Databricks Runtime."""
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    except Exception:
        return None


def _create_local_spark(config: SparkConfig):
    """Create a local SparkSession for testing."""
    from pyspark.sql import SparkSession
    
    builder = SparkSession.builder.appName(config.app_name)
    builder = builder.master(config.master)
    
    # Apply default test configurations
    default_config = {
        "spark.sql.shuffle.partitions": "2",
        "spark.default.parallelism": "2",
        "spark.sql.warehouse.dir": "/tmp/spark-warehouse",
        "spark.driver.extraJavaOptions": "-Dderby.system.home=/tmp/derby",
        "spark.ui.enabled": "false",
        "spark.sql.catalogImplementation": "in-memory",
    }
    
    # Merge with user config (user config takes precedence)
    merged_config = {**default_config, **config.config}
    
    for key, value in merged_config.items():
        builder = builder.config(key, value)
    
    return builder.getOrCreate()


def _create_databricks_connect_spark(config: SparkConfig):
    """Create a SparkSession using Databricks Connect."""
    try:
        from databricks.connect import DatabricksSession
        
        builder = DatabricksSession.builder
        
        if config.databricks_cluster_id:
            builder = builder.clusterId(config.databricks_cluster_id)
        
        if config.databricks_profile:
            builder = builder.profile(config.databricks_profile)
        
        # Apply additional configuration
        for key, value in config.config.items():
            builder = builder.config(key, value)
        
        return builder.getOrCreate()
    
    except ImportError:
        raise ImportError(
            "databricks-connect package not installed. "
            "Install with: pip install databricks-connect"
        )


@pytest.fixture(scope="session")
def spark_config() -> SparkConfig:
    """Session-scoped fixture providing Spark configuration.
    
    Override this fixture in your conftest.py to customize Spark settings:
    
    Example:
        @pytest.fixture(scope="session")
        def spark_config():
            return SparkConfig(
                app_name="my_tests",
                config={"spark.sql.shuffle.partitions": "4"}
            )
    """
    return SparkConfig.from_env()


@pytest.fixture(scope="session")
def spark_session(spark_config: SparkConfig) -> Generator:
    """Session-scoped SparkSession fixture.
    
    Automatically selects the appropriate Spark backend:
    1. Databricks Runtime: Uses existing SparkSession
    2. Databricks Connect: Creates remote session
    3. Local: Creates local SparkSession
    
    The SparkSession is shared across all tests in the session for
    efficiency. It is automatically stopped after all tests complete.
    
    Example:
        def test_dataframe_operations(spark_session):
            df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
            assert df.count() == 2
    """
    spark = None
    created_locally = False
    
    try:
        if _is_databricks_runtime():
            # Running in Databricks - use existing session
            spark = _get_databricks_spark()
            if spark is None:
                raise RuntimeError("Could not get SparkSession in Databricks Runtime")
        
        elif spark_config.use_databricks_connect:
            # Use Databricks Connect
            spark = _create_databricks_connect_spark(spark_config)
            created_locally = True
        
        else:
            # Create local SparkSession
            spark = _create_local_spark(spark_config)
            created_locally = True
        
        yield spark
    
    finally:
        # Only stop if we created the session locally
        if spark is not None and created_locally:
            try:
                spark.stop()
            except Exception:
                pass  # Ignore errors during cleanup


@pytest.fixture(scope="session")
def spark_context(spark_session):
    """Session-scoped SparkContext fixture.
    
    Provides access to the underlying SparkContext for low-level operations.
    
    Example:
        def test_rdd_operations(spark_context):
            rdd = spark_context.parallelize([1, 2, 3, 4, 5])
            assert rdd.count() == 5
    """
    return spark_session.sparkContext


@pytest.fixture(scope="function")
def local_spark() -> Generator:
    """Function-scoped local SparkSession for isolated tests.
    
    Creates a fresh local SparkSession for each test function.
    Use this when tests need complete isolation or might corrupt state.
    
    Note: This is slower than session-scoped spark_session. Use sparingly.
    
    Example:
        def test_isolated_operation(local_spark):
            # This test gets a fresh SparkSession
            df = local_spark.createDataFrame([(1,)], ["id"])
            df.createOrReplaceTempView("isolated_view")
            # View won't affect other tests
    """
    config = SparkConfig(
        app_name="dbx_test_isolated",
        master="local[2]",
        config={
            "spark.sql.shuffle.partitions": "2",
            "spark.ui.enabled": "false",
        }
    )
    
    spark = _create_local_spark(config)
    
    try:
        yield spark
    finally:
        spark.stop()

