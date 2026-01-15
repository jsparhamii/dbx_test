"""
Pytest configuration and shared fixtures for dbx_test.

This conftest.py demonstrates how to integrate the dbx_test fixture
architecture into your test suite. It provides:

- SparkSession lifecycle management (session-scoped)
- Databricks Connect/Runtime detection
- Temporary path management
- Test data generation
- Notebook test infrastructure

Environment Variables:
    DBX_TEST_USE_CONNECT=true    Enable Databricks Connect
    DBX_TEST_CLUSTER_ID          Cluster ID for Connect
    DBX_TEST_PROFILE             Databricks CLI profile
    DATABRICKS_HOST              Workspace URL
    DATABRICKS_TOKEN             Access token
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Generator

# Import fixture configurations
from dbx_test.fixtures.spark import SparkConfig
from dbx_test.fixtures.databricks import DatabricksConfig, is_databricks_runtime
from dbx_test.fixtures.paths import PathConfig

# Import all fixtures from the dbx_test.fixtures module
# This makes them available to all tests automatically
from dbx_test.fixtures import (
    # Spark fixtures
    spark_config,
    spark_session,
    spark_context,
    local_spark,
    # Databricks fixtures
    databricks_config,
    workspace_client,
    databricks_client,
    dbutils,
    # Path fixtures
    temp_dbfs_path,
    temp_workspace_path,
    temp_volume_path,
    # Data fixtures
    sample_dataframe,
    test_table,
    test_catalog_schema,
    parquet_file,
    delta_table,
    # Notebook fixtures
    notebook_context,
    notebook_runner,
    notebook_test_result,
)

# Re-export fixtures for pytest discovery
__all__ = [
    "spark_config",
    "spark_session", 
    "spark_context",
    "local_spark",
    "databricks_config",
    "workspace_client",
    "databricks_client",
    "dbutils",
    "temp_dbfs_path",
    "temp_workspace_path",
    "temp_volume_path",
    "sample_dataframe",
    "test_table",
    "test_catalog_schema",
    "parquet_file",
    "delta_table",
    "notebook_context",
    "notebook_runner",
    "notebook_test_result",
]


# ============================================================================
# Custom Spark Configuration (override defaults)
# ============================================================================

@pytest.fixture(scope="session")
def spark_config() -> SparkConfig:
    """Override default Spark configuration for this project.
    
    Customize this fixture to set project-specific Spark settings.
    """
    return SparkConfig(
        app_name="dbx_test_suite",
        master="local[2]",
        config={
            "spark.sql.shuffle.partitions": "2",
            "spark.default.parallelism": "2",
            "spark.ui.enabled": "false",
            "spark.sql.warehouse.dir": "/tmp/dbx_test_warehouse",
            "spark.driver.extraJavaOptions": "-Dderby.system.home=/tmp/derby",
        },
        use_databricks_connect=os.environ.get("DBX_TEST_USE_CONNECT", "").lower() == "true",
        databricks_cluster_id=os.environ.get("DBX_TEST_CLUSTER_ID"),
        databricks_profile=os.environ.get("DBX_TEST_PROFILE"),
    )


# ============================================================================
# Legacy Fixtures (for backwards compatibility)
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing.
    
    Legacy fixture - prefer temp_local_path from dbx_test.fixtures.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_test_notebook() -> str:
    """Create a sample test notebook content.
    
    Demonstrates the structure of a Nutter-style test notebook.
    """
    return '''
from dbx_test import NotebookTestFixture, run_notebook_tests
import json

class TestExample(NotebookTestFixture):
    def run_setup(self):
        self.data = [1, 2, 3, 4, 5]
    
    def test_sum(self):
        assert sum(self.data) == 15
    
    def test_length(self):
        assert len(self.data) == 5
    
    def test_max(self):
        assert max(self.data) == 5
    
    def run_cleanup(self):
        del self.data

# Run tests
results = run_notebook_tests(TestExample)
dbutils.notebook.exit(json.dumps(results))
'''


@pytest.fixture
def sample_config_dict() -> dict:
    """Create a sample configuration dictionary."""
    return {
        "workspace": {
            "host": "https://adb-123.azuredatabricks.net",
            "profile": "adb",
        },
        "cluster": {
            "size": "M",
            "libraries": [
                {"pypi": {"package": "pandas==2.0.0"}},
                {"pypi": {"package": "dbx_test"}},
            ],
        },
        "execution": {
            "timeout": 600,
            "parallel": False,
        },
        "paths": {
            "workspace_root": "/Workspace/Repos/tests",
            "test_pattern": "**/*_test.py",
        },
        "reporting": {
            "output_dir": ".dbx-test-results",
            "formats": ["junit", "console", "json"],
        },
    }


@pytest.fixture
def sample_test_results() -> dict:
    """Create sample test results."""
    return {
        "run_timestamp": "2025-01-01T10:00:00",
        "summary": {
            "total": 3,
            "passed": 2,
            "failed": 1,
            "skipped": 0,
            "duration": 5.5,
        },
        "tests": [
            {
                "notebook": "test_example",
                "test_name": "test_sum",
                "class_name": "TestExample",
                "status": "passed",
                "duration": 1.2,
                "timestamp": "2025-01-01T10:00:01",
            },
            {
                "notebook": "test_example",
                "test_name": "test_length",
                "class_name": "TestExample",
                "status": "passed",
                "duration": 0.8,
                "timestamp": "2025-01-01T10:00:02",
            },
            {
                "notebook": "test_example",
                "test_name": "test_max",
                "class_name": "TestExample",
                "status": "failed",
                "duration": 3.5,
                "error_message": "AssertionError: Expected 10, got 5",
                "error_traceback": "Traceback (most recent call last):\n  ...",
                "timestamp": "2025-01-01T10:00:03",
            },
        ],
    }


@pytest.fixture
def create_test_file(temp_dir: Path):
    """Factory fixture to create test files."""
    def _create_file(filename: str, content: str) -> Path:
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    
    return _create_file


@pytest.fixture
def mock_databricks_workspace(monkeypatch):
    """Mock Databricks workspace interactions.
    
    Legacy fixture - prefer workspace_client with proper mocking.
    """
    class MockWorkspaceClient:
        def __init__(self, *args, **kwargs):
            self.host = kwargs.get("host", "https://mock.databricks.com")
            self.token = kwargs.get("token", "mock-token")
        
        class workspace:
            @staticmethod
            def list(path):
                return []
            
            @staticmethod
            def get_status(path):
                return {"path": path, "object_type": "NOTEBOOK"}
        
        class jobs:
            @staticmethod
            def submit(run_name, tasks):
                return {"run_id": 12345}
            
            @staticmethod
            def get_run(run_id):
                return {
                    "run_id": run_id,
                    "state": {
                        "life_cycle_state": "TERMINATED",
                        "result_state": "SUCCESS",
                    },
                }
    
    return MockWorkspaceClient


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_databricks: mark test as requiring Databricks connection"
    )
    config.addinivalue_line(
        "markers", "requires_spark: mark test as requiring SparkSession"
    )
    config.addinivalue_line(
        "markers", "notebook: mark test as a notebook test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.nodeid or hasattr(item, "callspec") and "slow" in str(item.callspec):
            item.add_marker(pytest.mark.slow)
        
        # Auto-skip Databricks tests when not connected
        if "requires_databricks" in [m.name for m in item.iter_markers()]:
            if not is_databricks_runtime() and not os.environ.get("DATABRICKS_HOST"):
                item.add_marker(pytest.mark.skip(reason="Databricks connection not available"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
    parser.addoption(
        "--databricks-profile",
        action="store",
        default=None,
        help="Databricks CLI profile to use",
    )


def pytest_runtest_setup(item):
    """Skip tests based on markers and CLI options."""
    # Skip slow tests unless --run-slow is provided
    if "slow" in [m.name for m in item.iter_markers()]:
        if not item.config.getoption("--run-slow"):
            pytest.skip("Need --run-slow option to run")
    
    # Skip integration tests unless --run-integration is provided
    if "integration" in [m.name for m in item.iter_markers()]:
        if not item.config.getoption("--run-integration"):
            pytest.skip("Need --run-integration option to run")


# ============================================================================
# Session-scoped Cleanup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_session(request):
    """Session-scoped fixture for cleanup after all tests."""
    yield
    
    # Cleanup actions after test session
    # Add any global cleanup here if needed
    pass
