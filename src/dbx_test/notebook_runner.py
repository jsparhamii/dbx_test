"""
Notebook runner for executing tests directly in Databricks notebooks.

This module provides utilities for running tests interactively in Databricks notebooks
without using the CLI.
"""

from typing import Dict, Any, Optional, List, Union
import sys
import traceback
from dbx_test.testing import (
    NotebookTestFixture,
    discover_fixtures,
    run_tests as run_all_tests,
)


class NotebookRunner:
    """
    Runner for executing tests directly in Databricks notebooks.
    
    This provides a convenient way to run tests interactively during development.
    """
    
    def __init__(self, verbose: bool = True, parallel: bool = False, max_workers: Optional[int] = None):
        """
        Initialize the notebook runner.
        
        Args:
            verbose: Whether to print detailed output
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of parallel workers (None = use CPU count)
        """
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers
    
    def run(self, test_fixture_class: Optional[Union[type, List[type]]] = None) -> Dict[str, Any]:
        """
        Run tests from the current notebook.
        
        Args:
            test_fixture_class: Optional test fixture class(es) to run.
                              Can be a single class, a list of classes, or None.
                              If None, discovers and runs all fixtures in the notebook.
        
        Returns:
            Dictionary with test results
        
        Example:
            # Run all tests in the notebook
            runner = NotebookRunner()
            results = runner.run()
            
            # Run a specific test class
            results = runner.run(TestMyFeature)
            
            # Run multiple test classes
            results = runner.run([TestMyFirstTest, TestMySecondTest])
            
            # Run tests in parallel
            runner = NotebookRunner(parallel=True, max_workers=4)
            results = runner.run()
        """
        if test_fixture_class is None:
            return self._run_all_fixtures()
        elif isinstance(test_fixture_class, list):
            return self._run_multiple_fixtures(test_fixture_class)
        else:
            return self._run_single_fixture(test_fixture_class)
    
    def _run_single_fixture(self, fixture_class: type) -> Dict[str, Any]:
        """Run a single test fixture."""
        if not issubclass(fixture_class, NotebookTestFixture):
            raise ValueError(
                f"{fixture_class.__name__} must inherit from NotebookTestFixture"
            )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running {fixture_class.__name__}")
            if self.parallel:
                print(f"Parallel execution enabled (max_workers={self.max_workers or 'auto'})")
            print(f"{'='*60}\n")
        
        fixture = fixture_class()
        results = fixture.execute_tests(parallel=self.parallel, max_workers=self.max_workers)
        summary = fixture.get_results()
        
        if self.verbose:
            self._print_summary(summary)
        
        return summary
    
    def _run_multiple_fixtures(self, fixture_classes: List[type]) -> Dict[str, Any]:
        """Run multiple test fixtures."""
        if not fixture_classes:
            print("No test fixtures provided.")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "fixtures": [],
            }
        
        all_results = []
        fixture_summaries = []
        
        for fixture_class in fixture_classes:
            if not issubclass(fixture_class, NotebookTestFixture):
                raise ValueError(
                    f"{fixture_class.__name__} must inherit from NotebookTestFixture"
                )
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running {fixture_class.__name__}")
                if self.parallel:
                    print(f"Parallel execution enabled (max_workers={self.max_workers or 'auto'})")
                print(f"{'='*60}\n")
            
            fixture = fixture_class()
            results = fixture.execute_tests(parallel=self.parallel, max_workers=self.max_workers)
            summary = fixture.get_results()
            
            fixture_summaries.append({
                "fixture_name": fixture_class.__name__,
                "summary": summary,
            })
            
            all_results.extend(results)
        
        # Aggregate results
        total = len(all_results)
        passed = sum(1 for r in all_results if r.status == "passed")
        failed = sum(1 for r in all_results if r.status == "failed")
        errors = sum(1 for r in all_results if r.status == "error")
        
        aggregated = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "fixtures": fixture_summaries,
        }
        
        if self.verbose:
            self._print_summary(aggregated)
        
        return aggregated
    
    def _discover_caller_fixtures(self) -> List[type]:
        """Find fixture classes in the calling notebook's module scope.

        Walk up the stack to the nearest frame OUTSIDE this package (dbx_test) and use its globals.
        A fixed `_getframe(N)` is fragile: the depth to the notebook differs between calling
        `run_notebook_tests()` (which wraps `NotebookRunner.run`) and calling `NotebookRunner().run()`
        directly, so a hardcoded index lands on a dbx_test frame and discovers nothing. Walking to the
        first non-dbx_test frame is depth-independent and works for both entry points.
        """
        depth = 1
        while True:
            try:
                frame = sys._getframe(depth)
            except ValueError:
                return []  # ran off the top of the stack without finding a caller frame
            module = frame.f_globals.get("__name__", "")
            # Skip frames inside this package only. Match the package exactly ("dbx_test" or a
            # "dbx_test." submodule) rather than a bare prefix, so a caller module that merely
            # starts with "dbx_test" (e.g. a notebook named dbx_test_demo) is NOT skipped.
            if module != "dbx_test" and not module.startswith("dbx_test."):
                return discover_fixtures(frame.f_globals)
            depth += 1

    def _run_all_fixtures(self) -> Dict[str, Any]:
        """Discover and run all test fixtures in the current notebook."""
        fixtures = self._discover_caller_fixtures()

        if not fixtures:
            print("No test fixtures found in the notebook.")
            print("Make sure your test classes inherit from NotebookTestFixture.")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "fixtures": [],
            }
        
        all_results = []
        fixture_summaries = []
        
        for fixture_class in fixtures:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running {fixture_class.__name__}")
                if self.parallel:
                    print(f"Parallel execution enabled (max_workers={self.max_workers or 'auto'})")
                print(f"{'='*60}\n")
            
            fixture = fixture_class()
            results = fixture.execute_tests(parallel=self.parallel, max_workers=self.max_workers)
            summary = fixture.get_results()
            
            fixture_summaries.append({
                "fixture_name": fixture_class.__name__,
                "summary": summary,
            })
            
            all_results.extend(results)
        
        # Aggregate results
        total = len(all_results)
        passed = sum(1 for r in all_results if r.status == "passed")
        failed = sum(1 for r in all_results if r.status == "failed")
        errors = sum(1 for r in all_results if r.status == "error")
        
        aggregated = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "fixtures": fixture_summaries,
        }
        
        if self.verbose:
            self._print_summary(aggregated)
        
        return aggregated
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total']}")
        print(f"✓ Passed: {summary['passed']}")
        print(f"✗ Failed: {summary['failed']}")
        print(f"✗ Errors: {summary['errors']}")
        
        if summary['passed'] == summary['total'] and summary['total'] > 0:
            print(f"\n🎉 All tests passed!")
        elif summary['failed'] > 0 or summary['errors'] > 0:
            print(f"\n❌ Some tests failed")
        
        print(f"{'='*60}\n")


def run_notebook_tests(
    test_fixture_class: Optional[Union[type, List[type]]] = None,
    verbose: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to run tests directly in a Databricks notebook.
    
    This is the simplest way to run tests interactively.
    
    Args:
        test_fixture_class: Optional test fixture class(es) to run.
                          Can be a single class, a list of classes, or None.
                          If None, discovers and runs all fixtures.
        verbose: Whether to print detailed output
        parallel: Whether to run tests in parallel (default: False)
        max_workers: Maximum number of parallel workers (None = use CPU count)
    
    Returns:
        Dictionary with test results
    
    Example:
        # At the end of your test notebook, add:
        from dbx_test import run_notebook_tests
        
        # Run all tests
        run_notebook_tests()
        
        # Run a specific test class
        run_notebook_tests(TestMyFeature)
        
        # Run multiple test classes
        run_notebook_tests([TestMyFirstTest, TestMySecondTest])
        
        # Run tests in parallel
        run_notebook_tests(parallel=True, max_workers=4)
    """
    runner = NotebookRunner(verbose=verbose, parallel=parallel, max_workers=max_workers)
    return runner.run(test_fixture_class)


def install_notebook_package(package_path: str):
    """
    Install the test framework package in the notebook.
    
    Args:
        package_path: Path to the wheel file (DBFS path or local path)
    
    Example:
        # Install from DBFS
        install_notebook_package("/dbfs/FileStore/wheels/dbx_test-0.1.0-py3-none-any.whl")
        
        # Install from PyPI (if uploaded)
        install_notebook_package("databricks-notebook-test-framework")
    """
    import subprocess
    
    print(f"Installing {package_path}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_path],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print("✓ Installation successful!")
            print("\nYou can now use:")
            print("  from dbx_test import NotebookTestFixture, run_notebook_tests")
        else:
            print(f"✗ Installation failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"✗ Installation failed: {e}")
        traceback.print_exc()


# Simplified API for notebook usage
def quick_test(test_class: type) -> bool:
    """
    Quick test runner that returns True/False for pass/fail.
    
    Useful for simple assertions in notebook cells.
    
    Args:
        test_class: Test fixture class to run
    
    Returns:
        True if all tests passed, False otherwise
    
    Example:
        class TestQuick(NotebookTestFixture):
            def test_something(self):
                assert 1 + 1 == 2
        
        # Returns True/False
        passed = quick_test(TestQuick)
        assert passed, "Tests failed!"
    """
    runner = NotebookRunner(verbose=True)
    results = runner.run(test_class)
    return results['failed'] == 0 and results['errors'] == 0

