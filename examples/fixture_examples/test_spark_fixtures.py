"""
Example tests demonstrating Spark fixture usage.

These tests show how to use the dbx_test SparkSession fixtures
for local and Databricks Connect testing.
"""

import pytest


class TestSparkFixtures:
    """Tests demonstrating SparkSession fixture usage."""
    
    def test_spark_session_available(self, spark_session):
        """Verify SparkSession is available and working."""
        # Create a simple DataFrame
        df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
        
        # Verify it works
        assert df.count() == 2
        assert df.columns == ["id", "value"]
    
    def test_spark_sql_operations(self, spark_session):
        """Test SQL operations using SparkSession."""
        # Create temp view
        data = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        df = spark_session.createDataFrame(data, ["id", "name"])
        df.createOrReplaceTempView("test_users")
        
        # Query using SQL
        result = spark_session.sql("SELECT COUNT(*) as cnt FROM test_users")
        count = result.collect()[0]["cnt"]
        
        assert count == 3
    
    def test_spark_aggregations(self, spark_session):
        """Test DataFrame aggregations."""
        from pyspark.sql.functions import sum, avg, count
        
        data = [
            ("Electronics", 100.0),
            ("Electronics", 200.0),
            ("Books", 25.0),
            ("Books", 15.0),
            ("Books", 30.0),
        ]
        df = spark_session.createDataFrame(data, ["category", "amount"])
        
        # Aggregate by category
        result = df.groupBy("category").agg(
            sum("amount").alias("total"),
            count("*").alias("count"),
            avg("amount").alias("avg_amount")
        ).collect()
        
        # Convert to dict for easier assertions
        result_dict = {row["category"]: row.asDict() for row in result}
        
        assert result_dict["Electronics"]["total"] == 300.0
        assert result_dict["Electronics"]["count"] == 2
        assert result_dict["Books"]["count"] == 3
    
    def test_session_scoped_spark(self, spark_session, spark_context):
        """Verify session-scoped fixtures share the same Spark instance."""
        # Both fixtures should reference the same SparkSession
        assert spark_session.sparkContext == spark_context
        
        # Application name should match configuration
        assert "dbx_test" in spark_context.appName or "my_project" in spark_context.appName


class TestLocalSparkIsolation:
    """Tests demonstrating function-scoped local Spark for isolation."""
    
    @pytest.mark.slow
    def test_isolated_spark_session(self, local_spark):
        """Test with isolated SparkSession (function-scoped).
        
        This test gets its own SparkSession, useful when tests might
        corrupt shared state.
        """
        # Create a view that shouldn't affect other tests
        data = [(1, "isolated_data")]
        df = local_spark.createDataFrame(data, ["id", "value"])
        df.createOrReplaceTempView("isolated_view")
        
        # Verify the view exists
        tables = local_spark.catalog.listTables()
        table_names = [t.name for t in tables]
        assert "isolated_view" in table_names
    
    @pytest.mark.slow
    def test_another_isolated_session(self, local_spark):
        """Another test with isolated session.
        
        This should NOT see the view created in the previous test.
        """
        tables = local_spark.catalog.listTables()
        table_names = [t.name for t in tables]
        
        # isolated_view from previous test should not exist
        assert "isolated_view" not in table_names


class TestSparkWithSampleData:
    """Tests using the sample_dataframe factory fixture."""
    
    def test_users_dataframe(self, sample_dataframe):
        """Test with generated users data."""
        df = sample_dataframe("users", num_rows=50)
        
        assert df.count() == 50
        assert "user_id" in df.columns
        assert "username" in df.columns
        assert "email" in df.columns
    
    def test_transactions_dataframe(self, sample_dataframe):
        """Test with generated transactions data."""
        df = sample_dataframe("transactions", num_rows=100)
        
        assert df.count() == 100
        assert "transaction_id" in df.columns
        assert "amount" in df.columns
        assert "currency" in df.columns
    
    def test_custom_dataframe(self, sample_dataframe):
        """Test with custom data."""
        custom_data = [
            (1, "Product A", 10.99),
            (2, "Product B", 25.99),
            (3, "Product C", 5.99),
        ]
        df = sample_dataframe(
            data=custom_data,
            schema=["product_id", "name", "price"]
        )
        
        assert df.count() == 3
        assert df.columns == ["product_id", "name", "price"]
        
        # Verify data
        total_price = df.agg({"price": "sum"}).collect()[0][0]
        assert abs(total_price - 42.97) < 0.01

