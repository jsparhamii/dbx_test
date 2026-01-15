"""
Test data fixtures for pytest.

Provides fixtures for creating and managing test data:
- Sample DataFrames with various schemas
- Temporary tables (Delta, Parquet)
- Unity Catalog test schemas
- File-based test data
"""

import os
import uuid
import pytest
from typing import Optional, List, Dict, Any, Generator, Callable
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for test data generation.
    
    Attributes:
        catalog: Unity Catalog name for test tables
        schema_prefix: Prefix for test schema names
        table_prefix: Prefix for test table names
        cleanup_tables: Whether to drop tables after tests
    """
    catalog: Optional[str] = None
    schema_prefix: str = "dbx_test"
    table_prefix: str = "test"
    cleanup_tables: bool = True
    
    @classmethod
    def from_env(cls) -> "DataConfig":
        """Create DataConfig from environment variables."""
        return cls(
            catalog=os.environ.get("DBX_TEST_CATALOG"),
            schema_prefix=os.environ.get("DBX_TEST_SCHEMA_PREFIX", "dbx_test"),
            table_prefix=os.environ.get("DBX_TEST_TABLE_PREFIX", "test"),
            cleanup_tables=os.environ.get("DBX_TEST_CLEANUP_TABLES", "true").lower() == "true",
        )


class DataManager:
    """Manages test data lifecycle.
    
    Tracks created tables and provides cleanup functionality.
    """
    
    def __init__(self, spark, config: DataConfig):
        self.spark = spark
        self.config = config
        self._tables: List[str] = []
        self._temp_views: List[str] = []
        self._schemas: List[str] = []
    
    def generate_unique_name(self, prefix: str) -> str:
        """Generate a unique name with prefix."""
        suffix = uuid.uuid4().hex[:8]
        return f"{prefix}_{suffix}"
    
    def create_temp_view(self, df, name: Optional[str] = None) -> str:
        """Create a temporary view from a DataFrame.
        
        Args:
            df: DataFrame to create view from
            name: Optional view name (auto-generated if not provided)
            
        Returns:
            View name
        """
        view_name = name or self.generate_unique_name(self.config.table_prefix)
        df.createOrReplaceTempView(view_name)
        self._temp_views.append(view_name)
        return view_name
    
    def create_table(
        self,
        df,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
        format: str = "delta",
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
    ) -> str:
        """Create a table from a DataFrame.
        
        Args:
            df: DataFrame to create table from
            table_name: Optional table name (auto-generated if not provided)
            schema: Schema name (uses default if not provided)
            format: Table format (delta, parquet, etc.)
            mode: Write mode (overwrite, append, etc.)
            partition_by: Columns to partition by
            
        Returns:
            Full table name (schema.table or catalog.schema.table)
        """
        name = table_name or self.generate_unique_name(self.config.table_prefix)
        
        # Build full table name
        if self.config.catalog and schema:
            full_name = f"{self.config.catalog}.{schema}.{name}"
        elif schema:
            full_name = f"{schema}.{name}"
        else:
            full_name = name
        
        # Write the table
        writer = df.write.format(format).mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(full_name)
        
        self._tables.append(full_name)
        return full_name
    
    def create_schema(self, schema_name: Optional[str] = None) -> str:
        """Create a test schema.
        
        Args:
            schema_name: Optional schema name (auto-generated if not provided)
            
        Returns:
            Full schema name (catalog.schema or just schema)
        """
        name = schema_name or self.generate_unique_name(self.config.schema_prefix)
        
        if self.config.catalog:
            full_name = f"{self.config.catalog}.{name}"
        else:
            full_name = name
        
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {full_name}")
        self._schemas.append(full_name)
        return full_name
    
    def cleanup(self):
        """Clean up all created data artifacts."""
        if not self.config.cleanup_tables:
            return
        
        # Drop temp views
        for view in self._temp_views:
            try:
                self.spark.catalog.dropTempView(view)
            except Exception:
                pass
        
        # Drop tables
        for table in self._tables:
            try:
                self.spark.sql(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass
        
        # Drop schemas (only if empty)
        for schema in self._schemas:
            try:
                self.spark.sql(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            except Exception:
                pass
        
        # Clear tracking lists
        self._temp_views.clear()
        self._tables.clear()
        self._schemas.clear()


# Sample data generators
def _generate_users_data(spark, num_rows: int = 100):
    """Generate sample users data."""
    from pyspark.sql.functions import expr
    
    return spark.range(num_rows).select(
        expr("id").alias("user_id"),
        expr("concat('user_', id)").alias("username"),
        expr("concat('user_', id, '@example.com')").alias("email"),
        expr("CASE WHEN id % 3 = 0 THEN 'admin' WHEN id % 3 = 1 THEN 'user' ELSE 'guest' END").alias("role"),
        expr("date_add(current_date(), -cast(rand() * 365 as int))").alias("created_date"),
        expr("rand() > 0.2").alias("is_active"),
    )


def _generate_transactions_data(spark, num_rows: int = 1000):
    """Generate sample transactions data."""
    from pyspark.sql.functions import expr
    
    return spark.range(num_rows).select(
        expr("uuid()").alias("transaction_id"),
        expr("cast(rand() * 100 as int)").alias("user_id"),
        expr("round(rand() * 1000, 2)").alias("amount"),
        expr("CASE WHEN rand() > 0.5 THEN 'USD' ELSE 'EUR' END").alias("currency"),
        expr("CASE WHEN rand() > 0.8 THEN 'pending' WHEN rand() > 0.1 THEN 'completed' ELSE 'failed' END").alias("status"),
        expr("timestamp_add(current_timestamp(), -cast(rand() * 86400 * 30 as int), 'SECOND')").alias("created_at"),
    )


def _generate_products_data(spark, num_rows: int = 50):
    """Generate sample products data."""
    from pyspark.sql.functions import expr
    
    return spark.range(num_rows).select(
        expr("id").alias("product_id"),
        expr("concat('Product ', id)").alias("name"),
        expr("CASE WHEN id % 4 = 0 THEN 'Electronics' WHEN id % 4 = 1 THEN 'Clothing' WHEN id % 4 = 2 THEN 'Books' ELSE 'Home' END").alias("category"),
        expr("round(rand() * 500 + 10, 2)").alias("price"),
        expr("cast(rand() * 100 as int)").alias("stock_quantity"),
        expr("rand() > 0.1").alias("is_available"),
    )


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    """Session-scoped data configuration fixture."""
    return DataConfig.from_env()


@pytest.fixture(scope="session")
def data_manager(spark_session, data_config: DataConfig) -> Generator[DataManager, None, None]:
    """Session-scoped data manager fixture."""
    manager = DataManager(spark_session, data_config)
    yield manager
    manager.cleanup()


@pytest.fixture(scope="function")
def sample_dataframe(spark_session) -> Callable:
    """Factory fixture to create sample DataFrames.
    
    Returns a factory function that creates DataFrames on demand.
    
    Example:
        def test_aggregation(sample_dataframe):
            df = sample_dataframe("users", num_rows=50)
            assert df.count() == 50
            
            df = sample_dataframe([
                (1, "Alice", 100),
                (2, "Bob", 200),
            ], ["id", "name", "amount"])
            assert df.count() == 2
    """
    def _factory(
        data_type: str = "custom",
        data: Optional[List] = None,
        schema: Optional[List[str]] = None,
        num_rows: int = 100,
    ):
        if data is not None:
            # Custom data provided
            return spark_session.createDataFrame(data, schema or [])
        
        if data_type == "users":
            return _generate_users_data(spark_session, num_rows)
        elif data_type == "transactions":
            return _generate_transactions_data(spark_session, num_rows)
        elif data_type == "products":
            return _generate_products_data(spark_session, num_rows)
        else:
            # Default simple DataFrame
            return spark_session.range(num_rows).toDF("id")
    
    return _factory


@pytest.fixture(scope="function")
def test_table(data_manager: DataManager, sample_dataframe) -> Callable:
    """Factory fixture to create test tables.
    
    Creates tables that are automatically cleaned up after tests.
    
    Example:
        def test_table_operations(spark_session, test_table):
            table_name = test_table("users", num_rows=100)
            result = spark_session.sql(f"SELECT COUNT(*) FROM {table_name}")
            assert result.collect()[0][0] == 100
    """
    def _factory(
        data_type: str = "users",
        data: Optional[List] = None,
        schema: Optional[List[str]] = None,
        num_rows: int = 100,
        table_name: Optional[str] = None,
        db_schema: Optional[str] = None,
        format: str = "delta",
    ) -> str:
        df = sample_dataframe(data_type, data, schema, num_rows)
        return data_manager.create_table(
            df,
            table_name=table_name,
            schema=db_schema,
            format=format,
        )
    
    return _factory


@pytest.fixture(scope="function")
def test_catalog_schema(data_manager: DataManager) -> Callable:
    """Factory fixture to create test schemas.
    
    Creates schemas that are automatically cleaned up after tests.
    Requires Unity Catalog to be configured.
    
    Example:
        def test_schema_operations(spark_session, test_catalog_schema, test_table):
            schema_name = test_catalog_schema()
            table_name = test_table("users", db_schema=schema_name)
            # Schema and table cleaned up automatically
    """
    def _factory(schema_name: Optional[str] = None) -> str:
        return data_manager.create_schema(schema_name)
    
    return _factory


@pytest.fixture(scope="function")
def parquet_file(spark_session, temp_local_path, sample_dataframe) -> Callable:
    """Factory fixture to create parquet test files.
    
    Example:
        def test_parquet_read(spark_session, parquet_file):
            path = parquet_file("users", num_rows=50)
            df = spark_session.read.parquet(str(path))
            assert df.count() == 50
    """
    def _factory(
        data_type: str = "users",
        data: Optional[List] = None,
        schema: Optional[List[str]] = None,
        num_rows: int = 100,
        filename: str = "data.parquet",
    ):
        df = sample_dataframe(data_type, data, schema, num_rows)
        path = temp_local_path / filename
        df.write.parquet(str(path))
        return path
    
    return _factory


@pytest.fixture(scope="function")
def delta_table(spark_session, temp_local_path, sample_dataframe) -> Callable:
    """Factory fixture to create Delta Lake test tables.
    
    Example:
        def test_delta_operations(spark_session, delta_table):
            path = delta_table("transactions", num_rows=1000)
            df = spark_session.read.format("delta").load(str(path))
            assert df.count() == 1000
    """
    def _factory(
        data_type: str = "users",
        data: Optional[List] = None,
        schema: Optional[List[str]] = None,
        num_rows: int = 100,
        dirname: str = "delta_data",
    ):
        df = sample_dataframe(data_type, data, schema, num_rows)
        path = temp_local_path / dirname
        df.write.format("delta").save(str(path))
        return path
    
    return _factory


# Convenience fixture for quick DataFrame creation
@pytest.fixture(scope="function")
def create_df(spark_session):
    """Simple factory for creating DataFrames from lists.
    
    Example:
        def test_simple(create_df):
            df = create_df([(1, "a"), (2, "b")], ["id", "val"])
            assert df.count() == 2
    """
    def _factory(data: List, schema: List[str]):
        return spark_session.createDataFrame(data, schema)
    
    return _factory

