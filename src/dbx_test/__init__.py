"""
Databricks Notebook Test Framework

A comprehensive testing framework for Databricks notebooks with 
production-ready pytest fixtures.

Usage:
    # In notebooks:
    from dbx_test import NotebookTestFixture, run_notebook_tests
    
    # In pytest:
    from dbx_test.fixtures import spark_session, dbutils, notebook_context
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from dbx_test.config import TestConfig
from dbx_test.runner_remote import RemoteTestRunner
from dbx_test.reporting import TestReporter
from dbx_test.testing import NotebookTestFixture, run_tests
from dbx_test.notebook_runner import (
    NotebookRunner,
    run_notebook_tests,
    quick_test,
    install_notebook_package,
)

# Re-export key fixture components for convenience
from dbx_test.fixtures.notebook import (
    ContextAwareTestFixture,
    NotebookContext,
    NotebookTestResult,
    NotebookTestRunner,
    notebook_test_session,
)

# Re-export pytest compatibility utilities
from dbx_test.pytest_compat import (
    parametrize,
    skip,
    skipif,
    xfail,
    mark,
    get_marks,
    should_skip,
    get_xfail_info,
    get_parametrize_info,
)

__all__ = [
    # Core classes
    "TestConfig",
    "RemoteTestRunner",
    "TestReporter",
    # Test fixture base classes
    "NotebookTestFixture",
    "ContextAwareTestFixture",
    # Runners and utilities
    "run_tests",
    "NotebookRunner",
    "run_notebook_tests",
    "quick_test",
    "install_notebook_package",
    # Context and results
    "NotebookContext",
    "NotebookTestResult",
    "NotebookTestRunner",
    "notebook_test_session",
    # Pytest compatibility (for notebook use without pytest installed)
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "mark",
]

