"""
Notebook execution fixtures for pytest.

Provides fixtures for:
- Notebook execution context management
- Integration with NotebookTestFixture (Nutter-style tests)
- Result collection and assertion helpers
- Notebook dependency injection
"""

import os
import json
import pytest
from typing import Optional, Dict, Any, List, Generator, Type, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

# Import the NotebookTestFixture for integration
from dbx_test.testing import NotebookTestFixture, run_tests


@dataclass
class NotebookContext:
    """Context object passed to notebook tests.
    
    Provides access to fixtures and test infrastructure within notebooks.
    
    Attributes:
        spark: SparkSession instance
        dbutils: dbutils instance (real or mock)
        config: Test configuration dictionary
        temp_path: Temporary DBFS path for this test
        parameters: Widget parameters passed to the notebook
    """
    spark: Any
    dbutils: Any
    config: Dict[str, Any] = field(default_factory=dict)
    temp_path: Optional[str] = None
    parameters: Dict[str, str] = field(default_factory=dict)
    
    def get_param(self, name: str, default: str = "") -> str:
        """Get a notebook parameter value.
        
        Checks widgets first, then falls back to parameters dict.
        """
        try:
            return self.dbutils.widgets.get(name)
        except Exception:
            return self.parameters.get(name, default)
    
    def exit(self, result: Any):
        """Exit notebook with result.
        
        Converts result to JSON if necessary.
        """
        if isinstance(result, dict):
            self.dbutils.notebook.exit(json.dumps(result))
        else:
            self.dbutils.notebook.exit(str(result))


@dataclass
class NotebookTestResult:
    """Result from a notebook test execution.
    
    Attributes:
        notebook_name: Name of the executed notebook
        fixture_name: Name of the test fixture class
        total: Total number of tests
        passed: Number of passed tests
        failed: Number of failed tests
        errors: Number of error tests
        duration: Total execution duration
        test_results: List of individual test results
        raw_output: Raw output from notebook.exit()
    """
    notebook_name: str
    fixture_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration: float = 0.0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0
    
    def assert_passed(self):
        """Assert that all tests passed, with detailed failure message."""
        if not self.success:
            failures = [
                t for t in self.test_results 
                if t.get("status") in ("failed", "error")
            ]
            failure_msgs = "\n".join([
                f"  - {t['name']}: {t.get('error_message', 'Unknown error')}"
                for t in failures
            ])
            raise AssertionError(
                f"Notebook {self.notebook_name} had {self.failed} failures, "
                f"{self.errors} errors:\n{failure_msgs}"
            )


class NotebookTestRunner:
    """Runner for executing NotebookTestFixture classes.
    
    Bridges pytest fixtures with Nutter-style notebook tests.
    """
    
    def __init__(self, context: NotebookContext):
        self.context = context
        self._results: List[NotebookTestResult] = []
    
    def run_fixture(
        self,
        fixture_class: Type[NotebookTestFixture],
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> NotebookTestResult:
        """Run a single NotebookTestFixture class.
        
        Args:
            fixture_class: NotebookTestFixture subclass to run
            parallel: Whether to run tests in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            NotebookTestResult with test outcomes
        """
        # Inject context into fixture if it accepts it
        if hasattr(fixture_class, '__init__'):
            import inspect
            sig = inspect.signature(fixture_class.__init__)
            params = list(sig.parameters.keys())
            
            if 'context' in params:
                fixture = fixture_class(context=self.context)
            elif 'spark' in params:
                fixture = fixture_class(spark=self.context.spark)
            else:
                fixture = fixture_class()
        else:
            fixture = fixture_class()
        
        # Make context available as attribute
        fixture._test_context = self.context
        
        # Execute tests
        import time
        start_time = time.time()
        fixture.execute_tests(parallel=parallel, max_workers=max_workers)
        duration = time.time() - start_time
        
        # Collect results
        summary = fixture.get_results()
        
        result = NotebookTestResult(
            notebook_name=getattr(self.context, 'notebook_name', 'unknown'),
            fixture_name=fixture_class.__name__,
            total=summary['total'],
            passed=summary['passed'],
            failed=summary['failed'],
            errors=summary['errors'],
            duration=duration,
            test_results=summary['results'],
        )
        
        self._results.append(result)
        return result
    
    def run_all(
        self,
        fixture_classes: List[Type[NotebookTestFixture]],
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> List[NotebookTestResult]:
        """Run multiple NotebookTestFixture classes.
        
        Args:
            fixture_classes: List of fixture classes to run
            parallel: Whether to run tests in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            List of NotebookTestResult objects
        """
        results = []
        for fixture_class in fixture_classes:
            result = self.run_fixture(fixture_class, parallel, max_workers)
            results.append(result)
        return results
    
    def get_aggregated_result(self) -> Dict[str, Any]:
        """Get aggregated results from all executed fixtures.
        
        Returns result in the format expected by dbx_test CLI.
        """
        total = sum(r.total for r in self._results)
        passed = sum(r.passed for r in self._results)
        failed = sum(r.failed for r in self._results)
        errors = sum(r.errors for r in self._results)
        
        fixtures = []
        for r in self._results:
            fixtures.append({
                "fixture_name": r.fixture_name,
                "summary": {
                    "total": r.total,
                    "passed": r.passed,
                    "failed": r.failed,
                    "errors": r.errors,
                    "results": r.test_results,
                }
            })
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "fixtures": fixtures,
        }


class ContextAwareTestFixture(NotebookTestFixture):
    """Base class for test fixtures that need access to pytest fixtures.
    
    Extend this class to create notebook tests that can consume
    pytest fixtures like spark_session, temp paths, etc.
    
    Example:
        class TestMyFeature(ContextAwareTestFixture):
            def __init__(self, context: NotebookContext):
                super().__init__()
                self.spark = context.spark
                self.temp_path = context.temp_path
            
            def run_setup(self):
                self.test_df = self.spark.createDataFrame([(1, "a")], ["id", "val"])
            
            def test_data_present(self):
                assert self.test_df.count() > 0
    """
    
    def __init__(self, context: Optional[NotebookContext] = None, **kwargs):
        super().__init__()
        self._context = context
        
        # Allow passing spark directly for convenience
        if 'spark' in kwargs:
            self.spark = kwargs['spark']
        elif context:
            self.spark = context.spark
    
    @property
    def context(self) -> Optional[NotebookContext]:
        """Get the test context."""
        return self._context or getattr(self, '_test_context', None)


@pytest.fixture(scope="function")
def notebook_context(spark_session, dbutils, temp_dbfs_path) -> NotebookContext:
    """Function-scoped notebook context fixture.
    
    Provides a context object with access to all test infrastructure.
    Pass this to your NotebookTestFixture classes.
    
    Example:
        def test_notebook_logic(notebook_context):
            class TestMyLogic(ContextAwareTestFixture):
                def run_setup(self):
                    self.df = self.spark.range(10)
                
                def test_count(self):
                    assert self.df.count() == 10
            
            runner = NotebookTestRunner(notebook_context)
            result = runner.run_fixture(TestMyLogic)
            result.assert_passed()
    """
    return NotebookContext(
        spark=spark_session,
        dbutils=dbutils,
        temp_path=temp_dbfs_path,
    )


@pytest.fixture(scope="function")
def notebook_runner(notebook_context: NotebookContext) -> NotebookTestRunner:
    """Function-scoped notebook test runner fixture.
    
    Provides a runner for executing NotebookTestFixture classes.
    
    Example:
        def test_multiple_fixtures(notebook_runner):
            result1 = notebook_runner.run_fixture(TestFeatureA)
            result2 = notebook_runner.run_fixture(TestFeatureB)
            
            # Assert all passed
            result1.assert_passed()
            result2.assert_passed()
    """
    return NotebookTestRunner(notebook_context)


@pytest.fixture(scope="function")
def notebook_test_result() -> Callable:
    """Factory fixture to create NotebookTestResult objects.
    
    Useful for testing result handling without running actual notebooks.
    
    Example:
        def test_result_parsing(notebook_test_result):
            result = notebook_test_result(
                fixture_name="TestExample",
                total=3,
                passed=2,
                failed=1,
            )
            assert not result.success
    """
    def _factory(
        notebook_name: str = "test_notebook",
        fixture_name: str = "TestFixture",
        total: int = 0,
        passed: int = 0,
        failed: int = 0,
        errors: int = 0,
        duration: float = 0.0,
        test_results: Optional[List[Dict[str, Any]]] = None,
    ) -> NotebookTestResult:
        return NotebookTestResult(
            notebook_name=notebook_name,
            fixture_name=fixture_name,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            duration=duration,
            test_results=test_results or [],
        )
    
    return _factory


@contextmanager
def notebook_test_session(spark, dbutils=None, temp_path: Optional[str] = None):
    """Context manager for running notebook tests outside pytest.
    
    Useful when you want to run tests directly in a notebook cell
    but still use the fixture infrastructure.
    
    Example:
        # In a Databricks notebook:
        from dbx_test.fixtures.notebook import notebook_test_session, ContextAwareTestFixture
        
        with notebook_test_session(spark, dbutils) as runner:
            class TestMyFeature(ContextAwareTestFixture):
                def test_something(self):
                    assert True
            
            result = runner.run_fixture(TestMyFeature)
            result.assert_passed()
    """
    # Create mock dbutils if not provided
    if dbutils is None:
        from dbx_test.fixtures.databricks import MockDBUtils
        dbutils = MockDBUtils()
    
    context = NotebookContext(
        spark=spark,
        dbutils=dbutils,
        temp_path=temp_path or "/tmp/notebook_test",
    )
    
    runner = NotebookTestRunner(context)
    yield runner


# Export for use in notebooks
__all__ = [
    "NotebookContext",
    "NotebookTestResult", 
    "NotebookTestRunner",
    "ContextAwareTestFixture",
    "notebook_context",
    "notebook_runner",
    "notebook_test_result",
    "notebook_test_session",
]

