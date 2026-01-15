"""
Example tests demonstrating pytest.fixture support in NotebookTestFixture.

This file shows how to use pytest fixtures with dbx_test's NotebookTestFixture
classes. Fixtures work in both:
- Local pytest execution (full pytest DI)
- Databricks notebook execution (via register_fixture)

Usage:
    # Via pytest (fixtures auto-injected)
    pytest test_fixture_injection.py
    
    # Via notebook (register fixtures first)
    from dbx_test.fixtures import register_fixture
    register_fixture("spark_session", spark)
    results = run_notebook_tests(TestWithFixtures)
"""

import pytest
from dbx_test import NotebookTestFixture


# ============================================================================
# Define pytest fixtures (work in pytest execution)
# ============================================================================

@pytest.fixture(scope="session")
def database_connection():
    """Session-scoped database connection fixture."""
    # Simulate database connection
    class MockConnection:
        def __init__(self):
            self.connected = True
        
        def query(self, sql):
            return [{"id": 1, "name": "test"}]
        
        def close(self):
            self.connected = False
    
    conn = MockConnection()
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def sample_records():
    """Function-scoped sample data fixture."""
    return [
        {"id": 1, "name": "Alice", "amount": 100},
        {"id": 2, "name": "Bob", "amount": 200},
        {"id": 3, "name": "Charlie", "amount": 300},
    ]


@pytest.fixture
def empty_list():
    """Simple fixture returning an empty list."""
    return []


@pytest.fixture
def config():
    """Configuration fixture."""
    return {
        "environment": "test",
        "debug": True,
        "max_retries": 3,
    }


# ============================================================================
# Tests using fixtures
# ============================================================================

class TestWithFixtures(NotebookTestFixture):
    """Tests demonstrating fixture injection."""
    
    def test_simple_fixture(self, sample_records):
        """Test receiving a fixture as a parameter."""
        assert len(sample_records) == 3
        assert sample_records[0]["name"] == "Alice"
    
    def test_fixture_with_assertion(self, config):
        """Test using config fixture."""
        assert config["environment"] == "test"
        assert config["debug"] is True
        assert config["max_retries"] == 3
    
    def test_empty_fixture(self, empty_list):
        """Test with empty list fixture."""
        assert empty_list == []
        empty_list.append("item")
        assert len(empty_list) == 1
    
    def test_database_fixture(self, database_connection):
        """Test with session-scoped database fixture."""
        assert database_connection.connected
        result = database_connection.query("SELECT * FROM test")
        assert len(result) == 1


class TestFixturesWithParametrize(NotebookTestFixture):
    """Tests combining fixtures with parametrize."""
    
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_record_by_index(self, sample_records, index):
        """Access specific record using parametrize and fixture."""
        record = sample_records[index]
        assert "id" in record
        assert "name" in record
        assert record["id"] == index + 1
    
    @pytest.mark.parametrize("key", ["environment", "debug", "max_retries"])
    def test_config_keys(self, config, key):
        """Verify config has expected keys."""
        assert key in config
    
    @pytest.mark.parametrize("expected_count", [3])
    def test_record_count(self, sample_records, expected_count):
        """Test record count with parametrize."""
        assert len(sample_records) == expected_count


class TestMultipleFixtures(NotebookTestFixture):
    """Tests using multiple fixtures simultaneously."""
    
    def test_two_fixtures(self, sample_records, config):
        """Test receiving two fixtures."""
        assert len(sample_records) > 0
        assert config["environment"] == "test"
    
    def test_three_fixtures(self, sample_records, config, empty_list):
        """Test receiving three fixtures."""
        # Combine data from all fixtures
        for record in sample_records:
            if config["debug"]:
                empty_list.append(record["name"])
        
        assert len(empty_list) == 3
        assert "Alice" in empty_list


class TestFixtureWithMarkers(NotebookTestFixture):
    """Tests combining fixtures with skip/xfail markers."""
    
    @pytest.mark.skip(reason="Demonstrating skip with fixture")
    def test_skipped_with_fixture(self, sample_records):
        """This test is skipped but would receive fixture."""
        assert False, "Should not run"
    
    @pytest.mark.xfail(reason="Expected to fail")
    def test_xfail_with_fixture(self, sample_records):
        """Test expected to fail that receives fixture."""
        assert len(sample_records) == 999  # Will fail as expected
    
    @pytest.mark.skipif(False, reason="Condition is False")
    def test_conditional_with_fixture(self, config):
        """Test that runs because condition is False."""
        assert config is not None


# ============================================================================
# For notebook execution: Register fixtures before running
# ============================================================================

def setup_notebook_fixtures():
    """Setup fixtures for notebook execution.
    
    Call this before running tests in a notebook:
    
        setup_notebook_fixtures()
        results = run_notebook_tests(TestWithFixtures)
    """
    from dbx_test.fixtures import register_fixture, notebook_fixture
    
    # Register simple values
    register_fixture("sample_records", [
        {"id": 1, "name": "Alice", "amount": 100},
        {"id": 2, "name": "Bob", "amount": 200},
        {"id": 3, "name": "Charlie", "amount": 300},
    ])
    
    register_fixture("config", {
        "environment": "test",
        "debug": True,
        "max_retries": 3,
    })
    
    register_fixture("empty_list", [])
    
    # Or define fixtures with the decorator
    @notebook_fixture(scope="session")
    def database_connection():
        class MockConnection:
            def __init__(self):
                self.connected = True
            def query(self, sql):
                return [{"id": 1, "name": "test"}]
            def close(self):
                self.connected = False
        return MockConnection()


# ============================================================================
# Notebook execution example
# ============================================================================

if __name__ == "__main__":
    # This code runs when executed directly in a notebook
    from dbx_test import run_notebook_tests
    
    # Setup fixtures for notebook context
    setup_notebook_fixtures()
    
    # Run tests
    results = run_notebook_tests([
        TestWithFixtures,
        TestFixturesWithParametrize,
        TestMultipleFixtures,
        TestFixtureWithMarkers,
    ])
    
    print("\n" + "=" * 60)
    print("FIXTURE INJECTION TEST RESULTS")
    print("=" * 60)
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print("=" * 60)

