"""
Unit tests for the testing module (NotebookTestFixture, TestResult, etc.)
"""

import pytest
import time
from dbx_test.testing import (
    NotebookTestFixture,
    TestResult,
    discover_fixtures,
    run_tests,
)


class TestTestResult:
    """Tests for TestResult class."""
    
    def test_create_passed_result(self):
        """Test creating a passed test result."""
        result = TestResult(
            test_name="test_example",
            status="passed",
            duration=1.5,
        )
        
        assert result.test_name == "test_example"
        assert result.status == "passed"
        assert result.duration == 1.5
        assert result.error_message is None
        assert result.error_traceback is None
    
    def test_create_failed_result(self):
        """Test creating a failed test result."""
        result = TestResult(
            test_name="test_example",
            status="failed",
            duration=1.5,
            error_message="Assertion failed",
            error_traceback="Traceback...",
        )
        
        assert result.test_name == "test_example"
        assert result.status == "failed"
        assert result.error_message == "Assertion failed"
        assert result.error_traceback == "Traceback..."
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = TestResult(
            test_name="test_example",
            status="passed",
            duration=1.5,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["name"] == "test_example"
        assert result_dict["status"] == "passed"
        assert result_dict["duration"] == 1.5
        assert result_dict["error_message"] is None
        assert result_dict["error_traceback"] is None


class TestNotebookTestFixture:
    """Tests for NotebookTestFixture class."""
    
    def test_initialization(self):
        """Test fixture initialization."""
        fixture = NotebookTestFixture()
        
        assert fixture.results == []
        assert fixture._setup_executed is False
        assert fixture._cleanup_executed is False
    
    def test_execute_setup(self):
        """Test setup execution."""
        class TestFixture(NotebookTestFixture):
            def run_setup(self):
                self.setup_called = True
        
        fixture = TestFixture()
        assert fixture._execute_setup() is True
        assert hasattr(fixture, "setup_called")
        assert fixture.setup_called is True
        assert fixture._setup_executed is True
    
    def test_execute_setup_only_once(self):
        """Test that setup is only executed once."""
        class TestFixture(NotebookTestFixture):
            def run_setup(self):
                if not hasattr(self, "call_count"):
                    self.call_count = 0
                self.call_count += 1
        
        fixture = TestFixture()
        fixture._execute_setup()
        fixture._execute_setup()
        
        assert fixture.call_count == 1
    
    def test_execute_setup_failure(self):
        """Test setup failure handling."""
        class TestFixture(NotebookTestFixture):
            def run_setup(self):
                raise RuntimeError("Setup failed")
        
        fixture = TestFixture()
        assert fixture._execute_setup() is False
        assert fixture._setup_executed is False
    
    def test_execute_cleanup(self):
        """Test cleanup execution."""
        class TestFixture(NotebookTestFixture):
            def run_cleanup(self):
                self.cleanup_called = True
        
        fixture = TestFixture()
        fixture._execute_cleanup()
        
        assert hasattr(fixture, "cleanup_called")
        assert fixture.cleanup_called is True
        assert fixture._cleanup_executed is True
    
    def test_execute_cleanup_only_once(self):
        """Test that cleanup is only executed once."""
        class TestFixture(NotebookTestFixture):
            def run_cleanup(self):
                if not hasattr(self, "call_count"):
                    self.call_count = 0
                self.call_count += 1
        
        fixture = TestFixture()
        fixture._execute_cleanup()
        fixture._execute_cleanup()
        
        assert fixture.call_count == 1
    
    def test_execute_cleanup_continues_on_failure(self):
        """Test that cleanup failures don't raise exceptions."""
        class TestFixture(NotebookTestFixture):
            def run_cleanup(self):
                raise RuntimeError("Cleanup failed")
        
        fixture = TestFixture()
        # Should not raise, but _cleanup_executed is only set on success
        # The method catches exceptions but doesn't mark as executed if it fails
        fixture._execute_cleanup()
        # Since cleanup failed, _cleanup_executed is not True
        # Just verify it doesn't raise
        assert fixture._cleanup_executed is False
    
    def test_get_test_methods(self):
        """Test getting test methods from fixture."""
        class TestFixture(NotebookTestFixture):
            def test_one(self):
                pass
            
            def test_two(self):
                pass
            
            def not_a_test(self):
                pass
            
            def test__private(self):
                pass
        
        fixture = TestFixture()
        test_methods = fixture._get_test_methods()

        # _get_test_methods yields (test_name, method, params, param_id) tuples.
        # Should find test_one and test_two, but not others.
        test_names = [entry[0] for entry in test_methods]
        assert "test_one" in test_names
        assert "test_two" in test_names
        assert "not_a_test" not in test_names
        assert "test__private" not in test_names
    
    def test_execute_passing_test(self):
        """Test executing a passing test."""
        class TestFixture(NotebookTestFixture):
            def test_passes(self):
                assert True
        
        fixture = TestFixture()
        method = fixture._get_test_methods()[0][1]
        result = fixture._execute_test("test_passes", method)
        
        assert result.status == "passed"
        assert result.test_name == "test_passes"
        assert result.duration >= 0  # Duration should be non-negative
        assert result.error_message is None
    
    def test_execute_failing_test(self):
        """Test executing a failing test."""
        class TestFixture(NotebookTestFixture):
            def test_fails(self):
                assert False, "This should fail"
        
        fixture = TestFixture()
        method = fixture._get_test_methods()[0][1]
        result = fixture._execute_test("test_fails", method)
        
        assert result.status == "failed"
        assert result.test_name == "test_fails"
        assert "This should fail" in result.error_message
        assert result.error_traceback is not None
    
    def test_execute_error_test(self):
        """Test executing a test that raises an unexpected error."""
        class TestFixture(NotebookTestFixture):
            def test_error(self):
                raise ValueError("Unexpected error")
        
        fixture = TestFixture()
        method = fixture._get_test_methods()[0][1]
        result = fixture._execute_test("test_error", method)
        
        assert result.status == "error"
        assert result.test_name == "test_error"
        assert "Unexpected error" in result.error_message
        assert result.error_traceback is not None
    
    def test_execute_tests_sequential(self):
        """Test executing tests sequentially."""
        class TestFixture(NotebookTestFixture):
            def run_setup(self):
                self.values = []
            
            def test_one(self):
                self.values.append(1)
                assert True
            
            def test_two(self):
                self.values.append(2)
                assert True
            
            def run_cleanup(self):
                self.cleanup_values = self.values
        
        fixture = TestFixture()
        results = fixture.execute_tests(parallel=False)
        
        assert len(results) == 2
        assert all(r.status == "passed" for r in results)
        assert hasattr(fixture, "cleanup_values")
        assert fixture.cleanup_values == [1, 2]
    
    def test_execute_tests_skip_on_setup_failure(self):
        """Test that tests are skipped if setup fails."""
        class TestFixture(NotebookTestFixture):
            def run_setup(self):
                raise RuntimeError("Setup failed")
            
            def test_one(self):
                assert True
        
        fixture = TestFixture()
        results = fixture.execute_tests()
        
        assert len(results) == 0
    
    def test_get_results_summary(self):
        """Test getting test results summary."""
        class TestFixture(NotebookTestFixture):
            def test_pass(self):
                assert True
            
            def test_fail(self):
                assert False
        
        fixture = TestFixture()
        fixture.execute_tests()
        summary = fixture.get_results()
        
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["errors"] == 0
        assert len(summary["results"]) == 2


class TestDiscoverFixtures:
    """Tests for discover_fixtures function."""
    
    def test_discover_from_globals(self):
        """Test discovering fixtures from globals dict."""
        class TestFixture1(NotebookTestFixture):
            pass
        
        class TestFixture2(NotebookTestFixture):
            pass
        
        class NotAFixture:
            pass
        
        fixtures = discover_fixtures(locals())
        
        assert len(fixtures) == 2
        assert TestFixture1 in fixtures
        assert TestFixture2 in fixtures
        assert NotAFixture not in fixtures
    
    def test_discover_excludes_base_class(self):
        """Test that base NotebookTestFixture class is not discovered."""
        fixtures = discover_fixtures(globals())
        
        assert NotebookTestFixture not in fixtures


class TestRunTests:
    """Tests for run_tests function."""
    
    def test_run_single_fixture(self):
        """Test running tests from a single fixture."""
        class TestFixture(NotebookTestFixture):
            def test_one(self):
                assert True
            
            def test_two(self):
                assert True
        
        results = run_tests(locals())
        
        assert results["total"] == 2
        assert results["passed"] == 2
        assert results["failed"] == 0
        assert results["errors"] == 0
        assert len(results["fixtures"]) == 1
    
    def test_run_multiple_fixtures(self):
        """Test running tests from multiple fixtures."""
        class TestFixture1(NotebookTestFixture):
            def test_one(self):
                assert True
        
        class TestFixture2(NotebookTestFixture):
            def test_two(self):
                assert True
        
        results = run_tests(locals())
        
        assert results["total"] == 2
        assert results["passed"] == 2
        assert len(results["fixtures"]) == 2
    
    def test_run_with_failures(self):
        """Test running tests with failures."""
        class TestFixture(NotebookTestFixture):
            def test_pass(self):
                assert True
            
            def test_fail(self):
                assert False
        
        results = run_tests(locals())
        
        assert results["total"] == 2
        assert results["passed"] == 1
        assert results["failed"] == 1


class TestNotebookTestFixtureParallel:
    """Tests for parallel test execution."""
    
    def test_execute_tests_parallel(self):
        """Test executing tests in parallel."""
        class TestFixture(NotebookTestFixture):
            def test_one(self):
                time.sleep(0.1)
                assert True
            
            def test_two(self):
                time.sleep(0.1)
                assert True
            
            def test_three(self):
                time.sleep(0.1)
                assert True
        
        fixture = TestFixture()
        
        # Measure sequential execution time
        start = time.time()
        results_seq = fixture.execute_tests(parallel=False)
        seq_time = time.time() - start
        
        # Reset fixture
        fixture = TestFixture()
        
        # Measure parallel execution time
        start = time.time()
        results_par = fixture.execute_tests(parallel=True, max_workers=3)
        par_time = time.time() - start
        
        assert len(results_seq) == 3
        assert len(results_par) == 3
        assert all(r.status == "passed" for r in results_par)
        
        # Parallel should be significantly faster (at least 2x)
        # We use a conservative threshold to avoid flaky tests
        assert par_time < seq_time * 0.8
    
    def test_parallel_with_single_test(self):
        """Test that parallel mode works with a single test."""
        class TestFixture(NotebookTestFixture):
            def test_one(self):
                assert True
        
        fixture = TestFixture()
        results = fixture.execute_tests(parallel=True)
        
        # Should execute sequentially when only one test
        assert len(results) == 1
        assert results[0].status == "passed"

