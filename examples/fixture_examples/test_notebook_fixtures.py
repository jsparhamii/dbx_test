"""
Example tests demonstrating notebook fixture integration.

These tests show how to combine pytest fixtures with NotebookTestFixture
classes for hybrid testing approaches.
"""

import pytest
from dbx_test import NotebookTestFixture
from dbx_test.fixtures.notebook import (
    ContextAwareTestFixture,
    NotebookTestRunner,
    NotebookContext,
    notebook_test_session,
)


class TestNotebookFixtureIntegration:
    """Tests demonstrating notebook fixture integration with pytest."""
    
    def test_context_aware_fixture(self, notebook_context):
        """Test a ContextAwareTestFixture with injected context."""
        
        class TestWithContext(ContextAwareTestFixture):
            """Test fixture that uses the injected context."""
            
            def run_setup(self):
                # Access spark from context
                self.df = self.spark.createDataFrame([
                    (1, "Alice", 100),
                    (2, "Bob", 200),
                ], ["id", "name", "amount"])
            
            def test_count(self):
                """Test row count."""
                assert self.df.count() == 2
            
            def test_sum(self):
                """Test sum aggregation."""
                from pyspark.sql.functions import sum
                total = self.df.agg(sum("amount")).collect()[0][0]
                assert total == 300
            
            def test_spark_available(self):
                """Verify Spark is accessible."""
                assert self.spark is not None
                assert self.spark.version is not None
        
        # Run the fixture using the notebook runner
        runner = NotebookTestRunner(notebook_context)
        result = runner.run_fixture(TestWithContext)
        
        # Assert all tests passed
        assert result.total == 3
        assert result.passed == 3
        assert result.failed == 0
        result.assert_passed()
    
    def test_notebook_runner_multiple_fixtures(self, notebook_runner):
        """Test running multiple fixtures with the notebook runner."""
        
        class TestFeatureA(ContextAwareTestFixture):
            def test_feature_a(self):
                assert True
        
        class TestFeatureB(ContextAwareTestFixture):
            def test_feature_b(self):
                assert True
            
            def test_feature_b_2(self):
                assert True
        
        # Run both fixtures
        results = notebook_runner.run_all([TestFeatureA, TestFeatureB])
        
        assert len(results) == 2
        assert all(r.success for r in results)
        
        # Check aggregated results
        aggregated = notebook_runner.get_aggregated_result()
        assert aggregated["total"] == 3
        assert aggregated["passed"] == 3
    
    def test_fixture_with_setup_cleanup(self, notebook_context):
        """Test fixture with setup and cleanup methods."""
        cleanup_called = []
        
        class TestWithLifecycle(ContextAwareTestFixture):
            def run_setup(self):
                self.data = [1, 2, 3, 4, 5]
                self.temp_view_name = "lifecycle_test_view"
                df = self.spark.createDataFrame(
                    [(x,) for x in self.data], ["value"]
                )
                df.createOrReplaceTempView(self.temp_view_name)
            
            def test_data_available(self):
                result = self.spark.sql(
                    f"SELECT SUM(value) as total FROM {self.temp_view_name}"
                ).collect()[0]["total"]
                assert result == 15
            
            def run_cleanup(self):
                # Clean up the temp view
                self.spark.catalog.dropTempView(self.temp_view_name)
                cleanup_called.append(True)
        
        runner = NotebookTestRunner(notebook_context)
        result = runner.run_fixture(TestWithLifecycle)
        
        result.assert_passed()
        assert len(cleanup_called) == 1, "Cleanup should have been called"
    
    def test_fixture_failure_handling(self, notebook_runner):
        """Test that fixture failures are properly captured."""
        
        class TestWithFailure(ContextAwareTestFixture):
            def test_passing(self):
                assert True
            
            def test_failing(self):
                assert False, "This test should fail"
            
            def test_also_passing(self):
                assert True
        
        result = notebook_runner.run_fixture(TestWithFailure)
        
        assert result.total == 3
        assert result.passed == 2
        assert result.failed == 1
        assert not result.success
        
        # Verify the failure details are captured
        failed_tests = [t for t in result.test_results if t["status"] == "failed"]
        assert len(failed_tests) == 1
        assert "test_failing" in failed_tests[0]["name"]


class TestNotebookContextFeatures:
    """Tests demonstrating NotebookContext features."""
    
    def test_context_temp_path(self, notebook_context):
        """Test that temp_path is available in context."""
        assert notebook_context.temp_path is not None
        assert isinstance(notebook_context.temp_path, str)
    
    def test_context_dbutils(self, notebook_context):
        """Test that dbutils is available in context."""
        assert notebook_context.dbutils is not None
        
        # Test mock dbutils operations
        notebook_context.dbutils.fs.put("/tmp/test.txt", "hello")
        content = notebook_context.dbutils.fs.head("/tmp/test.txt")
        assert content == "hello"
    
    def test_context_parameters(self, notebook_context):
        """Test parameter handling in context."""
        notebook_context.parameters = {"env": "test", "batch_size": "100"}
        
        assert notebook_context.get_param("env") == "test"
        assert notebook_context.get_param("batch_size") == "100"
        assert notebook_context.get_param("missing", "default") == "default"


class TestNotebookTestSessionContextManager:
    """Tests demonstrating the notebook_test_session context manager."""
    
    def test_notebook_test_session_basic(self, spark_session):
        """Test using notebook_test_session context manager."""
        
        with notebook_test_session(spark_session) as runner:
            class QuickTest(ContextAwareTestFixture):
                def test_quick(self):
                    df = self.spark.range(10)
                    assert df.count() == 10
            
            result = runner.run_fixture(QuickTest)
            result.assert_passed()
    
    def test_notebook_test_session_with_dbutils(self, spark_session, dbutils):
        """Test notebook_test_session with real dbutils fixture."""
        
        with notebook_test_session(spark_session, dbutils) as runner:
            class TestWithDbutils(ContextAwareTestFixture):
                def test_dbutils_available(self):
                    ctx = self.context
                    assert ctx.dbutils is not None
            
            result = runner.run_fixture(TestWithDbutils)
            result.assert_passed()


class TestParallelExecution:
    """Tests demonstrating parallel test execution."""
    
    def test_parallel_fixture_execution(self, notebook_runner):
        """Test running fixture tests in parallel."""
        import time
        
        class TestParallel(ContextAwareTestFixture):
            def test_1(self):
                time.sleep(0.1)
                assert True
            
            def test_2(self):
                time.sleep(0.1)
                assert True
            
            def test_3(self):
                time.sleep(0.1)
                assert True
            
            def test_4(self):
                time.sleep(0.1)
                assert True
        
        # Run sequentially
        start_seq = time.time()
        result_seq = notebook_runner.run_fixture(TestParallel, parallel=False)
        duration_seq = time.time() - start_seq
        
        # Run in parallel
        start_par = time.time()
        result_par = notebook_runner.run_fixture(TestParallel, parallel=True, max_workers=4)
        duration_par = time.time() - start_par
        
        # Both should pass
        result_seq.assert_passed()
        result_par.assert_passed()
        
        # Parallel should be faster (though not guaranteed due to overhead)
        # Just verify it completed successfully
        assert result_par.total == result_seq.total == 4

