# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Test with Pytest Decorators
# MAGIC 
# MAGIC This notebook demonstrates using pytest decorators (parametrize, skip, xfail)
# MAGIC in Databricks notebook tests. The same test classes work both:
# MAGIC - When run via `pytest` locally or in CI
# MAGIC - When run via `run_notebook_tests()` in Databricks notebooks

# COMMAND ----------

# Install dbx_test if needed
# %pip install dbx_test

# COMMAND ----------

import pytest
from dbx_test import NotebookTestFixture, run_notebook_tests
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test with Parametrization

# COMMAND ----------

class TestDataTransforms(NotebookTestFixture):
    """Test data transformations with parametrized inputs."""
    
    def run_setup(self):
        """Create test DataFrame."""
        self.df = spark.createDataFrame([
            (1, "Alice", 100.0),
            (2, "Bob", 200.0),
            (3, "Charlie", 300.0),
        ], ["id", "name", "amount"])
        self.df.createOrReplaceTempView("test_data")
    
    @pytest.mark.parametrize("column,expected_type", [
        ("id", "int"),
        ("name", "string"),
        ("amount", "double"),
    ])
    def test_column_types(self, column, expected_type):
        """Verify column data types."""
        schema = {f.name: f.dataType.simpleString() for f in self.df.schema.fields}
        assert schema[column] == expected_type, f"Expected {expected_type}, got {schema[column]}"
    
    @pytest.mark.parametrize("filter_amount,expected_count", [
        (50, 3),   # All rows > 50
        (150, 2),  # 2 rows > 150
        (250, 1),  # 1 row > 250
        (350, 0),  # No rows > 350
    ])
    def test_filter_counts(self, filter_amount, expected_count):
        """Test row counts after filtering."""
        filtered = self.df.filter(f"amount > {filter_amount}")
        actual_count = filtered.count()
        assert actual_count == expected_count, f"Expected {expected_count}, got {actual_count}"
    
    def run_cleanup(self):
        spark.catalog.dropTempView("test_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test with Skip Conditions

# COMMAND ----------

class TestConditionalFeatures(NotebookTestFixture):
    """Tests with skip conditions based on environment."""
    
    def test_always_runs(self):
        """This test always runs."""
        assert spark is not None
    
    @pytest.mark.skip(reason="Feature under development")
    def test_new_feature(self):
        """Skipped until feature is complete."""
        assert False, "Should not run"
    
    @pytest.mark.skipif(
        spark.version < "3.4.0",
        reason="Requires Spark 3.4+"
    )
    def test_spark_34_feature(self):
        """Test requiring Spark 3.4 features."""
        # Use Spark 3.4+ specific functionality
        assert True
    
    @pytest.mark.skipif(
        "delta" not in [f.name for f in spark.catalog.listFunctions()],
        reason="Delta Lake not available"
    )
    def test_delta_operations(self):
        """Test requiring Delta Lake."""
        assert True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test with Expected Failures

# COMMAND ----------

class TestEdgeCases(NotebookTestFixture):
    """Tests for edge cases with xfail markers."""
    
    def run_setup(self):
        self.df = spark.createDataFrame([(1,), (2,), (3,)], ["value"])
    
    def test_normal_aggregation(self):
        """Normal aggregation works fine."""
        from pyspark.sql.functions import sum as spark_sum
        result = self.df.agg(spark_sum("value")).collect()[0][0]
        assert result == 6
    
    @pytest.mark.xfail(reason="Known issue with null handling")
    def test_null_aggregation(self):
        """Test aggregation with nulls - known issue."""
        df_with_nulls = spark.createDataFrame([(1,), (None,), (3,)], ["value"])
        from pyspark.sql.functions import sum as spark_sum
        result = df_with_nulls.agg(spark_sum("value")).collect()[0][0]
        # This assertion deliberately fails to demonstrate xfail
        assert result == 6, f"Expected 6, got {result}"  # Actually returns 4
    
    @pytest.mark.xfail(raises=Exception, reason="Empty DataFrame edge case")
    def test_empty_dataframe(self):
        """Test operations on empty DataFrame."""
        empty_df = spark.createDataFrame([], self.df.schema)
        # This would raise an error in real scenarios
        first_row = empty_df.first()
        assert first_row is not None  # Will xfail since first() returns None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined Decorators

# COMMAND ----------

class TestComplexScenarios(NotebookTestFixture):
    """Tests combining multiple decorators."""
    
    def run_setup(self):
        self.sample_data = [
            ("2024-01-01", 100),
            ("2024-01-02", 200),
            ("2024-01-03", 300),
        ]
        self.df = spark.createDataFrame(self.sample_data, ["date", "value"])
    
    @pytest.mark.slow
    @pytest.mark.parametrize("window_days,expected_rows", [
        (1, 3),
        (2, 3),
        (7, 3),
    ])
    def test_window_functions(self, window_days, expected_rows):
        """Parametrized window function tests."""
        from pyspark.sql.functions import col
        result = self.df.filter(col("value") > 0)
        assert result.count() == expected_rows
    
    @pytest.mark.integration
    @pytest.mark.parametrize("agg_func,expected", [
        ("sum", 600),
        ("avg", 200.0),
        ("max", 300),
        ("min", 100),
    ])
    def test_aggregation_functions(self, agg_func, expected):
        """Test various aggregation functions."""
        from pyspark.sql import functions as F
        
        agg_fn = getattr(F, agg_func)
        result = self.df.agg(agg_fn("value")).collect()[0][0]
        
        if agg_func == "avg":
            assert abs(result - expected) < 0.01
        else:
            assert result == expected

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run All Tests

# COMMAND ----------

# Execute all tests
results = run_notebook_tests([
    TestDataTransforms,
    TestConditionalFeatures,
    TestEdgeCases,
    TestComplexScenarios,
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# Display detailed results
print("=" * 70)
print("TEST RESULTS SUMMARY")
print("=" * 70)
print(f"Total:   {results['total']}")
print(f"Passed:  {results['passed']} ✓")
print(f"Failed:  {results['failed']} ✗")
print(f"Errors:  {results['errors']} ✗")
print(f"Skipped: {results['skipped']} ⊘")
print(f"XFailed: {results['xfailed']} ⊗ (expected failures)")
print(f"XPassed: {results['xpassed']} ⊕ (unexpected passes)")
print("=" * 70)

# Show any failures
failures = [r for r in results.get('all_results', []) if r['status'] == 'failed']
if failures:
    print("\nFAILED TESTS:")
    for f in failures:
        print(f"  - {f['name']}: {f.get('error_message', 'Unknown error')}")

# COMMAND ----------

# Return results to CLI (when run via dbx_test CLI)
dbutils.notebook.exit(json.dumps(results))

