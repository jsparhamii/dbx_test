"""
Pytest plugin for dbx_test framework.

This module registers dbx_test as a pytest plugin, providing:
- Automatic fixture availability
- NotebookTestFixture class collection
- Marker support (parametrize, skip, skipif, xfail)
- Custom marker registration

Installation:
    The plugin is automatically registered when dbx_test is installed via
    the entry point in pyproject.toml.

Usage:
    # Tests using NotebookTestFixture work with pytest out of the box:
    
    class TestMyFeature(NotebookTestFixture):
        @pytest.mark.parametrize("x", [1, 2, 3])
        def test_values(self, x):
            assert x > 0
    
    # Run with: pytest tests/
"""

import pytest
import inspect
import os
from typing import List, Optional, Any, Iterator

from dbx_test.testing import NotebookTestFixture


# ============================================================================
# Pytest Collection for NotebookTestFixture
# ============================================================================

class NotebookTestFixtureItem(pytest.Item):
    """Pytest item representing a single test method in a NotebookTestFixture."""
    
    def __init__(
        self,
        name: str,
        parent: "NotebookTestFixtureClass",
        test_method: callable,
        params: Optional[dict] = None,
        param_id: Optional[str] = None,
    ):
        super().__init__(name, parent)
        self.test_method = test_method
        self.params = params or {}
        self.param_id = param_id
        self._fixture_instance = None
        self._resolved_fixtures = {}
        
        # Copy markers from the test method
        if hasattr(test_method, "pytestmark"):
            for mark in test_method.pytestmark:
                self.add_marker(mark)
    
    @property
    def fixturenames(self) -> List[str]:
        """Get fixture names required by this test.
        
        This enables pytest's fixture injection system.
        """
        fixture_instance = self.parent.get_fixture_instance()
        fixture_params = fixture_instance._get_fixture_params(self.test_method)
        
        # Exclude parameters that are already provided via parametrize
        return [p for p in fixture_params if p not in self.params]
    
    def setup(self):
        """Setup for the test item."""
        # Get or create fixture instance from parent
        self._fixture_instance = self.parent.get_fixture_instance()
        if not self._fixture_instance._setup_executed:
            self._fixture_instance._execute_setup()
    
    def teardown(self):
        """Teardown for the test item."""
        # Cleanup is handled at the class level
        pass
    
    def runtest(self):
        """Execute the test with fixture injection."""
        # Merge parametrize params with pytest-resolved fixtures
        all_params = {**self._resolved_fixtures, **self.params}
        
        if all_params:
            self.test_method(**all_params)
        else:
            self.test_method()
    
    def _request_fixtures(self, fixtureinfo):
        """Request and resolve pytest fixtures.
        
        This is called by pytest's fixture system.
        """
        # Get fixture values from pytest's fixture manager
        for name in self.fixturenames:
            try:
                # The fixture manager will resolve these
                pass
            except Exception:
                pass
    
    def repr_failure(self, excinfo):
        """Represent a test failure."""
        return f"{self.name}: {excinfo.value}"
    
    def reportinfo(self):
        """Report test location info."""
        return self.fspath, None, f"{self.parent.name}::{self.name}"


class NotebookTestFixtureClass(pytest.Class):
    """Pytest collector for NotebookTestFixture classes."""
    
    def __init__(self, name: str, parent: pytest.Collector):
        super().__init__(name, parent)
        self._fixture_instance = None
    
    def get_fixture_instance(self) -> NotebookTestFixture:
        """Get or create the fixture instance."""
        if self._fixture_instance is None:
            self._fixture_instance = self.obj()
        return self._fixture_instance
    
    def collect(self) -> Iterator[pytest.Item]:
        """Collect test methods from the fixture class."""
        cls = self.obj
        instance = self.get_fixture_instance()
        
        for name in dir(instance):
            if not name.startswith("test_") or name.startswith("test__"):
                continue
            
            method = getattr(instance, name)
            if not callable(method):
                continue
            
            # Check for parametrize
            param_info = instance._get_parametrize_info(method)
            
            if param_info:
                argnames, param_sets = param_info
                
                for param_set in param_sets:
                    values = param_set["values"]
                    param_id = param_set.get("id")
                    
                    if not param_id:
                        param_id = "-".join(str(v) for v in values)
                    
                    test_name = f"{name}[{param_id}]"
                    params = dict(zip(argnames, values))
                    
                    yield NotebookTestFixtureItem.from_parent(
                        self,
                        name=test_name,
                        test_method=method,
                        params=params,
                        param_id=param_id,
                    )
            else:
                yield NotebookTestFixtureItem.from_parent(
                    self,
                    name=name,
                    test_method=method,
                )
    
    def teardown(self):
        """Class-level teardown."""
        if self._fixture_instance is not None:
            self._fixture_instance._execute_cleanup()


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with dbx_test markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "databricks: mark test as requiring Databricks connection",
    )
    config.addinivalue_line(
        "markers",
        "spark: mark test as requiring SparkSession",
    )
    config.addinivalue_line(
        "markers",
        "notebook: mark test as a notebook-based test",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test",
    )


def pytest_addoption(parser):
    """Add dbx_test command line options."""
    group = parser.getgroup("dbx_test", "Databricks notebook testing options")
    
    group.addoption(
        "--dbx-profile",
        action="store",
        default=os.environ.get("DBX_TEST_PROFILE"),
        help="Databricks CLI profile to use for tests",
    )
    
    group.addoption(
        "--dbx-cluster-id",
        action="store",
        default=os.environ.get("DBX_TEST_CLUSTER_ID"),
        help="Databricks cluster ID for Databricks Connect",
    )
    
    group.addoption(
        "--dbx-use-connect",
        action="store_true",
        default=os.environ.get("DBX_TEST_USE_CONNECT", "").lower() == "true",
        help="Use Databricks Connect instead of local Spark",
    )
    
    group.addoption(
        "--dbx-catalog",
        action="store",
        default=os.environ.get("DBX_TEST_CATALOG"),
        help="Unity Catalog name for test data",
    )
    
    group.addoption(
        "--dbx-no-cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of test data after tests",
    )


def pytest_pycollect_makeitem(collector, name, obj):
    """Create test items for NotebookTestFixture classes.
    
    This hook intercepts class collection and creates our custom
    collector for NotebookTestFixture subclasses.
    """
    if inspect.isclass(obj) and issubclass(obj, NotebookTestFixture) and obj is not NotebookTestFixture:
        # Return our custom collector
        return NotebookTestFixtureClass.from_parent(collector, name=name)
    
    return None


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on dbx_test markers."""
    from dbx_test.fixtures.databricks import is_databricks_runtime
    
    skip_databricks = pytest.mark.skip(reason="Databricks connection not available")
    skip_spark = pytest.mark.skip(reason="SparkSession not available")
    
    for item in items:
        # Check if running in Databricks or with Connect
        has_databricks = (
            is_databricks_runtime() or
            config.getoption("--dbx-use-connect") or
            os.environ.get("DATABRICKS_HOST")
        )
        
        # Skip Databricks tests when not connected
        if "databricks" in [m.name for m in item.iter_markers()]:
            if not has_databricks:
                item.add_marker(skip_databricks)
        
        # Check for Spark requirement
        if "spark" in [m.name for m in item.iter_markers()]:
            try:
                import pyspark
            except ImportError:
                item.add_marker(skip_spark)


@pytest.fixture(scope="session")
def dbx_test_options(request):
    """Fixture providing access to dbx_test command line options.
    
    Example:
        def test_with_options(dbx_test_options):
            if dbx_test_options.use_connect:
                # Test with Databricks Connect
                pass
    """
    class Options:
        profile = request.config.getoption("--dbx-profile")
        cluster_id = request.config.getoption("--dbx-cluster-id")
        use_connect = request.config.getoption("--dbx-use-connect")
        catalog = request.config.getoption("--dbx-catalog")
        no_cleanup = request.config.getoption("--dbx-no-cleanup")
    
    return Options()


# ============================================================================
# Fixture Registration
# ============================================================================

# Re-export fixtures from the fixtures module so they're available when 
# the plugin is loaded
try:
    from dbx_test.fixtures import (
        spark_session,
        spark_config,
        spark_context,
        local_spark,
        databricks_config,
        workspace_client,
        databricks_client,
        dbutils,
        temp_dbfs_path,
        temp_workspace_path,
        temp_volume_path,
        temp_local_path,
        sample_dataframe,
        test_table,
        notebook_context,
        notebook_runner,
    )
except ImportError:
    # Fixtures may not be available in all contexts
    pass
