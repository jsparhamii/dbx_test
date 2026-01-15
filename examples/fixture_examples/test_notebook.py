# Databricks notebook source
# MAGIC %md
# MAGIC # Example Notebook Test Using Fixtures
# MAGIC 
# MAGIC This notebook demonstrates how to use the dbx_test fixture architecture
# MAGIC within a Databricks notebook. It shows the integration between pytest fixtures
# MAGIC and Nutter-style NotebookTestFixture classes.

# COMMAND ----------

# Install the framework if not already installed
# %pip install dbx_test

# COMMAND ----------

from dbx_test import NotebookTestFixture, run_notebook_tests
from dbx_test.fixtures.notebook import (
    ContextAwareTestFixture,
    NotebookContext,
    notebook_test_session,
)
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Traditional NotebookTestFixture (Nutter-style)
# MAGIC 
# MAGIC This approach works exactly like Nutter - define test methods in a class
# MAGIC that inherits from NotebookTestFixture.

# COMMAND ----------

class TestTraditionalStyle(NotebookTestFixture):
    """Traditional Nutter-style test fixture.
    
    Uses the global `spark` session available in Databricks notebooks.
    """
    
    def run_setup(self):
        """Set up test data."""
        self.df = spark.createDataFrame([
            (1, "Alice", 100),
            (2, "Bob", 200),
            (3, "Charlie", 300),
        ], ["id", "name", "amount"])
        self.df.createOrReplaceTempView("test_users")
    
    def test_row_count(self):
        """Test that we have 3 rows."""
        count = spark.sql("SELECT COUNT(*) FROM test_users").collect()[0][0]
        assert count == 3, f"Expected 3 rows, got {count}"
    
    def test_total_amount(self):
        """Test the total amount."""
        total = spark.sql("SELECT SUM(amount) FROM test_users").collect()[0][0]
        assert total == 600, f"Expected 600, got {total}"
    
    def test_distinct_names(self):
        """Test distinct name count."""
        distinct = spark.sql("SELECT COUNT(DISTINCT name) FROM test_users").collect()[0][0]
        assert distinct == 3
    
    def run_cleanup(self):
        """Clean up test data."""
        spark.catalog.dropTempView("test_users")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 2: Context-Aware Test Fixture
# MAGIC 
# MAGIC This approach uses the `ContextAwareTestFixture` which provides explicit
# MAGIC dependency injection of spark, dbutils, and other fixtures.

# COMMAND ----------

class TestWithContext(ContextAwareTestFixture):
    """Context-aware test fixture with explicit dependencies.
    
    The spark session is injected via the context, making tests
    more portable and testable locally.
    """
    
    def __init__(self, context=None, **kwargs):
        super().__init__(context, **kwargs)
        # Access spark from context or fallback to global
        if self.spark is None:
            self.spark = spark  # Use global Databricks spark
    
    def run_setup(self):
        """Set up test data using injected spark."""
        self.products = self.spark.createDataFrame([
            ("P001", "Laptop", "Electronics", 999.99),
            ("P002", "Mouse", "Electronics", 29.99),
            ("P003", "Desk", "Furniture", 299.99),
            ("P004", "Chair", "Furniture", 199.99),
        ], ["product_id", "name", "category", "price"])
        
        self.products.createOrReplaceTempView("products")
    
    def test_category_counts(self):
        """Test category aggregation."""
        from pyspark.sql.functions import count
        
        result = self.products.groupBy("category").agg(count("*").alias("cnt"))
        result_dict = {row["category"]: row["cnt"] for row in result.collect()}
        
        assert result_dict["Electronics"] == 2
        assert result_dict["Furniture"] == 2
    
    def test_price_calculations(self):
        """Test price statistics."""
        from pyspark.sql.functions import avg, max, min
        
        stats = self.products.agg(
            avg("price").alias("avg_price"),
            max("price").alias("max_price"),
            min("price").alias("min_price")
        ).collect()[0]
        
        assert stats["max_price"] == 999.99
        assert stats["min_price"] == 29.99
    
    def test_expensive_products(self):
        """Test filtering expensive products."""
        expensive = self.products.filter("price > 100").count()
        assert expensive == 3
    
    def run_cleanup(self):
        """Clean up test data."""
        self.spark.catalog.dropTempView("products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 3: Using notebook_test_session Context Manager
# MAGIC 
# MAGIC This approach bridges pytest-style fixture injection with notebook execution.
# MAGIC Useful when you want to simulate the pytest environment in a notebook.

# COMMAND ----------

# Create a context manually (simulating what pytest fixtures would provide)
context = NotebookContext(
    spark=spark,
    dbutils=dbutils,
    temp_path="/tmp/test_session",
    parameters={"env": "development", "batch_date": "2024-01-15"}
)

# Use the notebook_test_session context manager
with notebook_test_session(spark, dbutils, temp_path="/tmp/test_session") as runner:
    
    class TestWithSession(ContextAwareTestFixture):
        """Test using the session context manager."""
        
        def run_setup(self):
            self.orders = self.spark.createDataFrame([
                (1, "ORD-001", 150.00, "completed"),
                (2, "ORD-002", 75.50, "pending"),
                (3, "ORD-003", 220.00, "completed"),
                (4, "ORD-004", 50.00, "cancelled"),
            ], ["order_id", "order_number", "amount", "status"])
        
        def test_order_count(self):
            assert self.orders.count() == 4
        
        def test_completed_orders(self):
            completed = self.orders.filter("status = 'completed'")
            assert completed.count() == 2
        
        def test_total_revenue(self):
            """Calculate revenue from completed orders."""
            from pyspark.sql.functions import sum
            
            revenue = self.orders.filter("status = 'completed'") \
                .agg(sum("amount")).collect()[0][0]
            assert revenue == 370.00
    
    # Run the tests
    session_result = runner.run_fixture(TestWithSession)
    
    print(f"Session tests: {session_result.passed}/{session_result.total} passed")
    if not session_result.success:
        print("Failed tests:")
        for t in session_result.test_results:
            if t["status"] != "passed":
                print(f"  - {t['name']}: {t.get('error_message', 'Unknown')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running All Tests
# MAGIC 
# MAGIC Use `run_notebook_tests` to execute all test fixtures and collect results.

# COMMAND ----------

# Run all traditional NotebookTestFixture tests
results = run_notebook_tests([TestTraditionalStyle, TestWithContext])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Return Results to CLI
# MAGIC 
# MAGIC When running via the dbx_test CLI, return results using `dbutils.notebook.exit()`.

# COMMAND ----------

# Return results for CLI consumption
dbutils.notebook.exit(json.dumps(results))

