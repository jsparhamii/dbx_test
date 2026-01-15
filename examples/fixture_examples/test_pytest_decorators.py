"""
Example tests demonstrating pytest decorator support in NotebookTestFixture.

This file shows how to use standard pytest decorators with dbx_test's
NotebookTestFixture classes. All decorators work seamlessly in both:
- Local pytest execution
- Databricks notebook execution

Supported Decorators:
    - @pytest.mark.parametrize - Run tests with multiple parameter sets
    - @pytest.mark.skip - Skip tests unconditionally
    - @pytest.mark.skipif - Skip tests conditionally
    - @pytest.mark.xfail - Mark tests as expected to fail
    - @pytest.mark.timeout - Set test timeout (requires pytest-timeout)
    - Custom @pytest.mark.* markers
"""

import pytest
import sys
import os
from dbx_test import NotebookTestFixture


# ============================================================================
# Parametrize Examples
# ============================================================================

class TestParametrize(NotebookTestFixture):
    """Examples of @pytest.mark.parametrize usage."""
    
    @pytest.mark.parametrize("x,y,expected", [
        (1, 2, 3),
        (2, 3, 5),
        (10, 20, 30),
        (0, 0, 0),
    ])
    def test_addition(self, x, y, expected):
        """Test addition with multiple parameter sets."""
        assert x + y == expected
    
    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_positive_numbers(self, value):
        """Test that values are positive."""
        assert value > 0
    
    @pytest.mark.parametrize("text,expected_length", [
        ("hello", 5),
        ("world", 5),
        ("pytest", 6),
        ("", 0),
    ])
    def test_string_length(self, text, expected_length):
        """Test string length calculations."""
        assert len(text) == expected_length
    
    @pytest.mark.parametrize("a", [1, 2])
    @pytest.mark.parametrize("b", [10, 20])
    def test_multiple_parametrize(self, a, b):
        """Test with multiple parametrize decorators (creates cross-product).
        
        Note: Multiple parametrize decorators are combined by pytest.
        This test runs 4 times: (1,10), (1,20), (2,10), (2,20)
        """
        assert a + b > 0


class TestParametrizeWithIds(NotebookTestFixture):
    """Examples of parametrize with custom IDs."""
    
    @pytest.mark.parametrize("input_val,expected", [
        pytest.param(1, 1, id="identity"),
        pytest.param(2, 4, id="square_of_two"),
        pytest.param(3, 9, id="square_of_three"),
    ])
    def test_with_param_ids(self, input_val, expected):
        """Test with custom parameter IDs for clearer reporting."""
        assert input_val ** 2 == expected
    
    @pytest.mark.parametrize("data", [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ], ids=["alice_data", "bob_data"])
    def test_with_ids_list(self, data):
        """Test with IDs provided as a list."""
        assert "name" in data
        assert "age" in data
        assert data["age"] > 0


# ============================================================================
# Skip Examples
# ============================================================================

class TestSkip(NotebookTestFixture):
    """Examples of @pytest.mark.skip and @pytest.mark.skipif usage."""
    
    def test_normal_test(self):
        """A normal test that should run."""
        assert True
    
    @pytest.mark.skip(reason="Feature not implemented yet")
    def test_skip_unconditional(self):
        """This test is always skipped."""
        assert False, "This should never run"
    
    @pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
    def test_skip_python_version(self):
        """Skip if Python version is too old."""
        # Use Python 3.10+ syntax
        assert True
    
    @pytest.mark.skipif(os.environ.get("CI") is None, reason="Only runs in CI")
    def test_skip_not_ci(self):
        """Skip if not running in CI environment."""
        assert True
    
    @pytest.mark.skipif(
        not os.environ.get("DATABRICKS_HOST"),
        reason="Requires Databricks connection"
    )
    def test_skip_no_databricks(self):
        """Skip if Databricks is not configured."""
        assert True


# ============================================================================
# XFail Examples
# ============================================================================

class TestXFail(NotebookTestFixture):
    """Examples of @pytest.mark.xfail usage."""
    
    @pytest.mark.xfail(reason="Known bug in calculation")
    def test_xfail_expected(self):
        """Test that is expected to fail."""
        # This will be marked as xfailed, not failed
        assert 1 + 1 == 3
    
    @pytest.mark.xfail(reason="This bug was fixed!")
    def test_xfail_but_passes(self):
        """Test marked as xfail but actually passes (xpassed)."""
        # This will be marked as xpassed (unexpectedly passed)
        assert 1 + 1 == 2
    
    @pytest.mark.xfail(strict=True, reason="Must fail")
    def test_xfail_strict(self):
        """Strict xfail - passing is a failure."""
        # If this passes, it's treated as a test failure
        assert 1 + 1 == 3
    
    @pytest.mark.xfail(raises=ZeroDivisionError, reason="Division by zero expected")
    def test_xfail_specific_exception(self):
        """XFail expecting a specific exception."""
        result = 1 / 0  # noqa
    
    @pytest.mark.xfail(
        condition=sys.platform == "win32",
        reason="Known Windows issue"
    )
    def test_xfail_conditional(self):
        """XFail only on Windows."""
        assert True


# ============================================================================
# Custom Markers
# ============================================================================

class TestCustomMarkers(NotebookTestFixture):
    """Examples of custom pytest markers."""
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test marked as slow - can be filtered with -m 'not slow'."""
        import time
        time.sleep(0.1)
        assert True
    
    @pytest.mark.integration
    def test_integration(self):
        """Test marked as integration test."""
        assert True
    
    @pytest.mark.databricks
    def test_databricks_specific(self):
        """Test that requires Databricks - skipped if not available."""
        assert True
    
    @pytest.mark.smoke
    def test_smoke_test(self):
        """Smoke test - can be run with -m smoke."""
        assert True
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_multiple_markers(self):
        """Test with multiple custom markers."""
        assert True


# ============================================================================
# Combined Decorators
# ============================================================================

class TestCombinedDecorators(NotebookTestFixture):
    """Examples combining multiple decorators."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("value", [1, 2, 3])
    def test_slow_parametrized(self, value):
        """Parametrized test marked as slow."""
        import time
        time.sleep(0.01)
        assert value > 0
    
    @pytest.mark.skipif(os.environ.get("SKIP_FLAKY") == "true", reason="Flaky test skipped")
    @pytest.mark.parametrize("attempt", range(3))
    def test_flaky_parametrized(self, attempt):
        """Parametrized test that can be conditionally skipped."""
        assert True
    
    @pytest.mark.xfail(reason="Edge case known to fail")
    @pytest.mark.parametrize("edge_case", [
        pytest.param(None, id="null_input"),
        pytest.param("", id="empty_string"),
        pytest.param([], id="empty_list"),
    ])
    def test_edge_cases_xfail(self, edge_case):
        """Edge cases expected to fail."""
        # Simulate failing on edge cases
        assert edge_case, f"Edge case failed: {edge_case}"


# ============================================================================
# Spark Integration Example
# ============================================================================

class TestSparkWithDecorators(NotebookTestFixture):
    """Example combining Spark tests with pytest decorators.
    
    Note: These tests require SparkSession to be available.
    Run with actual Spark or mock appropriately.
    """
    
    def run_setup(self):
        """Setup Spark test data."""
        # This would use real Spark in production
        self.test_data = [
            (1, "Alice", 100),
            (2, "Bob", 200),
            (3, "Charlie", 300),
        ]
    
    @pytest.mark.parametrize("column_name,expected_values", [
        ("id", [1, 2, 3]),
        ("name", ["Alice", "Bob", "Charlie"]),
        ("amount", [100, 200, 300]),
    ])
    @pytest.mark.skipif(
        os.environ.get("SPARK_HOME") is None and os.environ.get("DATABRICKS_RUNTIME_VERSION") is None,
        reason="Spark not available"
    )
    def test_column_values(self, column_name, expected_values):
        """Test column values with parametrize."""
        # In real usage, this would query actual Spark DataFrames
        columns = {
            "id": [row[0] for row in self.test_data],
            "name": [row[1] for row in self.test_data],
            "amount": [row[2] for row in self.test_data],
        }
        assert columns[column_name] == expected_values
    
    @pytest.mark.xfail(reason="Aggregation edge case")
    def test_empty_aggregation(self):
        """Test aggregation on empty data."""
        empty_data = []
        # This would fail on real Spark with empty DataFrame
        assert sum(row[2] for row in empty_data) == 0


# ============================================================================
# Running Tests
# ============================================================================

if __name__ == "__main__":
    # Can run directly in a notebook with run_notebook_tests
    from dbx_test import run_notebook_tests
    import json
    
    results = run_notebook_tests([
        TestParametrize,
        TestParametrizeWithIds,
        TestSkip,
        TestXFail,
        TestCustomMarkers,
        TestCombinedDecorators,
    ])
    
    print("\n" + "=" * 60)
    print("PYTEST DECORATOR TEST RESULTS")
    print("=" * 60)
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"XFailed: {results['xfailed']}")
    print(f"XPassed: {results['xpassed']}")
    print("=" * 60)

