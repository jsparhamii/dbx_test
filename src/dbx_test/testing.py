"""
Notebook testing base class implementation.

This module provides the NotebookTestFixture base class for organizing
and executing tests in Databricks notebooks.

Pytest Decorator Support:
    NotebookTestFixture natively supports pytest decorators:
    - @pytest.mark.parametrize - Run tests with multiple parameter sets
    - @pytest.mark.skip - Skip tests unconditionally
    - @pytest.mark.skipif - Skip tests conditionally
    - @pytest.mark.xfail - Mark tests as expected to fail
    - @pytest.mark.timeout - Set test timeout
    - @pytest.fixture - Inject fixtures into test methods
    - Custom @pytest.mark.* markers

Fixture Injection:
    Test methods can receive fixtures as parameters:
    
    class TestWithFixtures(NotebookTestFixture):
        def test_with_spark(self, spark_session):
            df = spark_session.range(10)
            assert df.count() == 10
    
    In pytest, standard pytest fixtures work automatically.
    In notebooks, use register_fixture() or @notebook_fixture.

Example:
    import pytest
    from dbx_test import NotebookTestFixture
    
    class TestMyFeature(NotebookTestFixture):
        @pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (2, 3, 5)])
        def test_add(self, x, y, expected):
            assert x + y == expected
        
        @pytest.mark.skip(reason="Not implemented")
        def test_future(self):
            pass
        
        def test_with_fixture(self, sample_data):
            # sample_data is injected from fixtures
            assert sample_data is not None
"""

import inspect
import time
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestResult:
    """Represents the result of a single test."""
    
    def __init__(
        self,
        test_name: str,
        status: str,
        duration: float,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        markers: Optional[List[str]] = None,
    ):
        self.test_name = test_name
        self.status = status  # passed, failed, error, skipped, xfailed, xpassed
        self.duration = duration
        self.error_message = error_message
        self.error_traceback = error_traceback
        self.parameters = parameters
        self.markers = markers or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
        }
        if self.parameters:
            result["parameters"] = self.parameters
        if self.markers:
            result["markers"] = self.markers
        return result


class NotebookTestFixture(ABC):
    """
    Base class for notebook test fixtures.
    
    Provides a pattern for organizing notebook tests with setup, test methods, and cleanup.
    Fully compatible with pytest decorators including parametrize, skip, skipif, and xfail.
    Supports fixture injection for test methods.
    
    Usage:
        class TestMyNotebook(NotebookTestFixture):
            def run_setup(self):
                # Setup code
                pass
            
            def test_something(self):
                # Test code
                assert True, "Test failed"
            
            @pytest.mark.parametrize("value", [1, 2, 3])
            def test_values(self, value):
                assert value > 0
            
            def test_with_fixture(self, spark_session):
                # spark_session is injected
                df = spark_session.range(10)
                assert df.count() == 10
            
            def run_cleanup(self):
                # Cleanup code
                pass
    
    Fixture Injection:
        In pytest: Standard @pytest.fixture works automatically.
        In notebooks: Register fixtures before running tests:
        
            from dbx_test.fixtures.injection import register_fixture
            register_fixture("spark_session", spark)
            register_fixture("dbutils", dbutils)
    """
    
    # Class-level fixture registry for this fixture class
    _fixture_registry = None
    _fixture_values: Dict[str, Any] = {}
    
    def __init__(self, **fixtures):
        """Initialize the test fixture.
        
        Args:
            **fixtures: Pre-provided fixture values for injection
        """
        self.results: List[TestResult] = []
        self._setup_executed = False
        self._cleanup_executed = False
        self._provided_fixtures = fixtures
    
    def run_setup(self):
        """Override this method to provide setup logic."""
        pass
    
    def run_cleanup(self):
        """Override this method to provide cleanup logic."""
        pass
    
    def _execute_setup(self) -> bool:
        """Execute the setup method."""
        if self._setup_executed:
            return True
        
        try:
            self.run_setup()
            self._setup_executed = True
            return True
        except Exception as e:
            print(f"Setup failed: {e}")
            traceback.print_exc()
            return False
    
    def _execute_cleanup(self):
        """Execute the cleanup method."""
        if self._cleanup_executed:
            return
        
        try:
            self.run_cleanup()
            self._cleanup_executed = True
        except Exception as e:
            print(f"Cleanup failed: {e}")
            traceback.print_exc()
    
    def _get_marks(self, method: Callable) -> List[Dict[str, Any]]:
        """Extract pytest marks from a method.
        
        Args:
            method: Method to extract marks from
            
        Returns:
            List of mark info dictionaries
        """
        marks = []
        pytestmark = getattr(method, "pytestmark", [])
        if not isinstance(pytestmark, list):
            pytestmark = [pytestmark]
        
        for mark in pytestmark:
            if hasattr(mark, "name"):
                marks.append({
                    "name": mark.name,
                    "args": getattr(mark, "args", ()),
                    "kwargs": getattr(mark, "kwargs", {}),
                })
        
        return marks
    
    def _should_skip(self, method: Callable) -> Tuple[bool, Optional[str]]:
        """Check if a method should be skipped.
        
        Args:
            method: Method to check
            
        Returns:
            Tuple of (should_skip, reason)
        """
        marks = self._get_marks(method)
        
        for mark in marks:
            if mark["name"] == "skip":
                reason = mark["kwargs"].get("reason", "")
                if not reason and mark["args"]:
                    reason = mark["args"][0]
                return (True, reason or "unconditional skip")
            
            elif mark["name"] == "skipif":
                if mark["args"]:
                    condition = mark["args"][0]
                    if callable(condition):
                        condition = condition()
                    if condition:
                        reason = mark["kwargs"].get("reason", "condition was True")
                        return (True, reason)
        
        return (False, None)
    
    def _get_xfail_info(self, method: Callable) -> Optional[Dict[str, Any]]:
        """Get xfail information from a method.
        
        Args:
            method: Method to check
            
        Returns:
            Dict with xfail info or None
        """
        marks = self._get_marks(method)
        
        for mark in marks:
            if mark["name"] == "xfail":
                condition = mark["args"][0] if mark["args"] else True
                if callable(condition):
                    condition = condition()
                
                if condition:
                    return {
                        "reason": mark["kwargs"].get("reason", ""),
                        "strict": mark["kwargs"].get("strict", False),
                        "raises": mark["kwargs"].get("raises"),
                        "run": mark["kwargs"].get("run", True),
                    }
        
        return None
    
    def _get_parametrize_info(self, method: Callable) -> Optional[Tuple[List[str], List[Tuple]]]:
        """Extract parametrize information from a method.
        
        Args:
            method: Method to extract parametrize info from
            
        Returns:
            Tuple of (argnames, argvalues) or None
        """
        marks = self._get_marks(method)
        
        for mark in marks:
            if mark["name"] == "parametrize":
                if len(mark["args"]) >= 2:
                    argnames = mark["args"][0]
                    argvalues = mark["args"][1]
                    ids = mark["kwargs"].get("ids")
                    
                    # Parse argnames
                    if isinstance(argnames, str):
                        argnames = [a.strip() for a in argnames.replace(",", " ").split()]
                    
                    # Parse argvalues with ids
                    parsed_values = []
                    for i, values in enumerate(argvalues):
                        # Handle pytest.param() objects
                        if hasattr(values, "values"):
                            param_values = values.values
                            param_id = getattr(values, "id", None)
                            param_marks = getattr(values, "marks", [])
                        else:
                            if not isinstance(values, (tuple, list)):
                                values = (values,)
                            param_values = tuple(values)
                            param_id = ids[i] if ids and i < len(ids) else None
                            param_marks = []
                        
                        parsed_values.append({
                            "values": param_values,
                            "id": param_id,
                            "marks": param_marks,
                        })
                    
                    return (argnames, parsed_values)
        
        return None
    
    def _get_custom_markers(self, method: Callable) -> List[str]:
        """Get custom marker names from a method."""
        marks = self._get_marks(method)
        known_markers = {"skip", "skipif", "xfail", "parametrize", "usefixtures", "timeout"}
        return [m["name"] for m in marks if m["name"] not in known_markers]
    
    def _get_fixture_params(self, method: Callable) -> List[str]:
        """Get fixture parameter names from a method signature.
        
        Returns parameter names that are not 'self' and could be fixtures.
        
        Args:
            method: Method to inspect
            
        Returns:
            List of parameter names that might need fixture injection
        """
        # Get the underlying function if it's a bound method
        func = method.__func__ if hasattr(method, '__func__') else method
        sig = inspect.signature(func)
        
        params = []
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            # Skip *args and **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            params.append(name)
        
        return params
    
    def _resolve_fixtures(
        self, 
        method: Callable, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve fixtures for a test method.
        
        Args:
            method: Test method that may require fixtures
            params: Already-provided parameters (from parametrize)
            
        Returns:
            Dictionary of resolved fixture values
        """
        params = params or {}
        fixture_params = self._get_fixture_params(method)
        resolved = {}
        
        for name in fixture_params:
            # Skip if already provided (e.g., from parametrize)
            if name in params:
                resolved[name] = params[name]
                continue
            
            # Check provided fixtures from constructor
            if name in self._provided_fixtures:
                resolved[name] = self._provided_fixtures[name]
                continue
            
            # Check class-level fixture values
            if name in self._fixture_values:
                resolved[name] = self._fixture_values[name]
                continue
            
            # Try global registry
            try:
                from dbx_test.fixtures.injection import get_global_registry
                registry = get_global_registry()
                value = registry._resolve_single(name, self._provided_fixtures, resolved)
                resolved[name] = value
            except (ImportError, ValueError):
                # Fixture not found - might be provided by pytest
                # or might cause an error when test runs
                pass
        
        return resolved
    
    @classmethod
    def register_fixture(cls, name: str, value: Any):
        """Register a fixture value for this test class.
        
        Args:
            name: Fixture name
            value: Fixture value
            
        Example:
            TestMyFeature.register_fixture("spark_session", spark)
            TestMyFeature.register_fixture("dbutils", dbutils)
        """
        cls._fixture_values[name] = value
    
    def _get_test_methods(self) -> List[Tuple[str, Callable, Optional[Tuple], Optional[str]]]:
        """Get all test methods, expanding parametrized tests.
        
        Returns:
            List of (test_name, method, params, param_id) tuples
        """
        test_methods = []
        
        for name in dir(self):
            if not name.startswith("test_") or name.startswith("test__"):
                continue
            
                method = getattr(self, name)
            if not callable(method):
                continue
            
            # Check for parametrize
            param_info = self._get_parametrize_info(method)
            
            if param_info:
                argnames, param_sets = param_info
                
                for i, param_set in enumerate(param_sets):
                    values = param_set["values"]
                    param_id = param_set.get("id")
                    
                    # Generate parameter ID if not provided
                    if not param_id:
                        param_id = "-".join(str(v) for v in values)
                    
                    test_name = f"{name}[{param_id}]"
                    
                    # Create params dict
                    params = dict(zip(argnames, values))
                    
                    test_methods.append((test_name, method, params, param_id))
            else:
                test_methods.append((name, method, None, None))
        
        return test_methods
    
    def _execute_test(
        self, 
        test_name: str, 
        test_method: Callable,
        params: Optional[Dict[str, Any]] = None,
    ) -> TestResult:
        """Execute a single test method with marker support.
        
        Args:
            test_name: Name of the test
            test_method: Test method to execute
            params: Optional parameters for parametrized tests
            
        Returns:
            TestResult with status
        """
        start_time = time.time()
        markers = self._get_custom_markers(test_method)
        
        # Check skip markers
        should_skip, skip_reason = self._should_skip(test_method)
        if should_skip:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                status="skipped",
                duration=duration,
                error_message=skip_reason,
                parameters=params,
                markers=markers,
            )
        
        # Check xfail markers
        xfail_info = self._get_xfail_info(test_method)
        
        # If xfail with run=False, skip execution
        if xfail_info and not xfail_info.get("run", True):
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                status="xfailed",
                duration=duration,
                error_message=f"Expected failure: {xfail_info.get('reason', '')}",
                parameters=params,
                markers=markers,
            )
        
        # Resolve fixtures for the test method
        try:
            resolved_fixtures = self._resolve_fixtures(test_method, params)
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                error_message=f"Fixture resolution failed: {str(e)}",
                error_traceback=traceback.format_exc(),
                parameters=params,
                markers=markers,
            )
        
        # Merge params with resolved fixtures (params take precedence)
        all_params = {**resolved_fixtures}
        if params:
            all_params.update(params)
        
        # Execute the test
        try:
            if all_params:
                test_method(**all_params)
            else:
                test_method()
            
            duration = time.time() - start_time
            
            # Test passed
            if xfail_info:
                # Expected to fail but passed
                if xfail_info.get("strict", False):
                    return TestResult(
                        test_name=test_name,
                        status="failed",
                        duration=duration,
                        error_message=f"[XPASS(strict)] {xfail_info.get('reason', '')}",
                        parameters=params,
                        markers=markers,
                    )
                else:
                    return TestResult(
                        test_name=test_name,
                        status="xpassed",
                        duration=duration,
                        error_message=f"Unexpectedly passed: {xfail_info.get('reason', '')}",
                        parameters=params,
                        markers=markers,
                    )
            
            return TestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                parameters=params,
                markers=markers,
            )
        
        except AssertionError as e:
            duration = time.time() - start_time
            
            if xfail_info:
                # Check if it's the expected exception type
                expected_raises = xfail_info.get("raises")
                if expected_raises is None or isinstance(e, expected_raises):
                    return TestResult(
                        test_name=test_name,
                        status="xfailed",
                        duration=duration,
                        error_message=f"Expected failure: {xfail_info.get('reason', '')} - {str(e)}",
                        parameters=params,
                        markers=markers,
                    )
            
            return TestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                parameters=params,
                markers=markers,
            )
        
        except Exception as e:
            duration = time.time() - start_time
            
            if xfail_info:
                expected_raises = xfail_info.get("raises")
                if expected_raises is None or isinstance(e, expected_raises):
                    return TestResult(
                        test_name=test_name,
                        status="xfailed",
                        duration=duration,
                        error_message=f"Expected failure: {xfail_info.get('reason', '')} - {str(e)}",
                        parameters=params,
                        markers=markers,
                    )
            
            return TestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                error_message=f"Unexpected error: {str(e)}",
                error_traceback=traceback.format_exc(),
                parameters=params,
                markers=markers,
            )
    
    def execute_tests(self, parallel: bool = False, max_workers: Optional[int] = None) -> List[TestResult]:
        """
        Execute all tests in the fixture.
        
        Args:
            parallel: Whether to run tests in parallel (default: False)
            max_workers: Maximum number of parallel workers (default: None = use CPU count)
        
        Returns:
            List of TestResult objects
        """
        self.results = []
        
        # Execute setup
        if not self._execute_setup():
            print(f"Skipping tests due to setup failure")
            return self.results
        
        # Get and execute test methods (including expanded parametrized tests)
        test_methods = self._get_test_methods()
        
        if parallel and len(test_methods) > 1:
            self.results = self._execute_tests_parallel(test_methods, max_workers)
        else:
            self.results = self._execute_tests_sequential(test_methods)
        
        # Execute cleanup
        self._execute_cleanup()
        
        return self.results
    
    def _execute_tests_sequential(self, test_methods: List[Tuple]) -> List[TestResult]:
        """Execute tests sequentially."""
        results = []
        for test_name, test_method, params, param_id in test_methods:
            print(f"Running {test_name}...")
            result = self._execute_test(test_name, test_method, params)
            results.append(result)
            
            status_icon = {
                "passed": "✓",
                "failed": "✗",
                "error": "✗",
                "skipped": "⊘",
                "xfailed": "⊗",
                "xpassed": "⊕",
            }.get(result.status, "?")
            
            status_text = result.status.upper()
            if result.error_message and result.status != "passed":
                print(f"  {status_icon} {status_text}: {result.error_message}")
            else:
                print(f"  {status_icon} {status_text}")
        
        return results
    
    def _execute_tests_parallel(self, test_methods: List[Tuple], max_workers: Optional[int] = None) -> List[TestResult]:
        """Execute tests in parallel using ThreadPoolExecutor."""
        results = []
        
        print(f"Running {len(test_methods)} tests in parallel (max_workers={max_workers or 'auto'})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._execute_test, test_name, test_method, params): test_name
                for test_name, test_method, params, param_id in test_methods
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status_icon = {
                        "passed": "✓",
                        "failed": "✗",
                        "error": "✗",
                        "skipped": "⊘",
                        "xfailed": "⊗",
                        "xpassed": "⊕",
                    }.get(result.status, "?")
                    
                    if result.error_message and result.status not in ("passed", "xpassed"):
                        print(f"  {status_icon} {test_name} {result.status.upper()}: {result.error_message}")
                    else:
                        print(f"  {status_icon} {test_name} {result.status.upper()}")
                        
                except Exception as e:
                    print(f"  ✗ {test_name} FAILED TO EXECUTE: {e}")
                    results.append(TestResult(
                        test_name=test_name,
                        status="error",
                        duration=0.0,
                        error_message=f"Failed to execute: {str(e)}",
                        error_traceback=traceback.format_exc(),
                    ))
        
        # Sort results by test name to maintain consistent ordering
        results.sort(key=lambda r: r.test_name)
        
        return results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get test results summary.
        
        Returns:
            Dictionary with test results
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        errors = sum(1 for r in self.results if r.status == "error")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        xfailed = sum(1 for r in self.results if r.status == "xfailed")
        xpassed = sum(1 for r in self.results if r.status == "xpassed")
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "xfailed": xfailed,
            "xpassed": xpassed,
            "results": [r.to_dict() for r in self.results],
        }


def discover_fixtures(module_or_globals) -> List[type]:
    """
    Discover all NotebookTestFixture subclasses in a module or globals dict.
    
    Args:
        module_or_globals: Module object or globals() dict
    
    Returns:
        List of NotebookTestFixture subclasses
    """
    fixtures = []
    
    if isinstance(module_or_globals, dict):
        items = module_or_globals.items()
    else:
        items = inspect.getmembers(module_or_globals)
    
    for name, obj in items:
        if (inspect.isclass(obj) and 
            issubclass(obj, NotebookTestFixture) and 
            obj is not NotebookTestFixture):
            fixtures.append(obj)
    
    return fixtures


def run_tests(module_or_globals) -> Dict[str, Any]:
    """
    Discover and run all tests in a module or globals dict.
    
    Args:
        module_or_globals: Module object or globals() dict
    
    Returns:
        Dictionary with aggregated test results
    """
    fixtures = discover_fixtures(module_or_globals)
    
    all_results = []
    fixture_summaries = []
    
    for fixture_class in fixtures:
        print(f"\n{'='*60}")
        print(f"Running {fixture_class.__name__}")
        print(f"{'='*60}\n")
        
        fixture = fixture_class()
        results = fixture.execute_tests()
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
    skipped = sum(1 for r in all_results if r.status == "skipped")
    xfailed = sum(1 for r in all_results if r.status == "xfailed")
    xpassed = sum(1 for r in all_results if r.status == "xpassed")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"XFailed: {xfailed}")
    print(f"XPassed: {xpassed}")
    print(f"{'='*60}\n")
    
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "fixtures": fixture_summaries,
        "all_results": [r.to_dict() for r in all_results],
    }


# Legacy alias for backwards compatibility (will be removed in future versions)
NutterFixture = NotebookTestFixture
