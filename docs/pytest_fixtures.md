# Pytest Fixtures Architecture

This document describes the pytest fixture architecture in `dbx_test`, designed to support notebook testing with clean dependency injection, reusable components, and seamless local/CI execution.

## Overview

The `dbx_test.fixtures` module provides production-ready pytest fixtures for:

- **SparkSession lifecycle management** (session-scoped)
- **Databricks Connect/Runtime detection**
- **Temporary path management** (DBFS, Workspace, Volumes)
- **Test data generation**
- **Notebook test infrastructure** (Nutter integration)

## Architecture

```
src/dbx_test/fixtures/
├── __init__.py       # Public API exports
├── spark.py          # SparkSession fixtures
├── databricks.py     # Databricks client fixtures
├── paths.py          # Temporary path fixtures
├── data.py           # Test data fixtures
└── notebook.py       # Notebook execution fixtures
```

## Quick Start

### 1. Install dbx_test

```bash
pip install dbx_test
```

### 2. Create conftest.py

```python
# tests/conftest.py
from dbx_test.fixtures import (
    spark_session,
    dbutils,
    sample_dataframe,
    notebook_context,
)
```

### 3. Write Tests

```python
# tests/test_my_feature.py
def test_data_processing(spark_session, sample_dataframe):
    df = sample_dataframe("users", num_rows=100)
    result = df.groupBy("role").count()
    assert result.count() > 0
```

## Fixture Reference

### Spark Fixtures

#### `spark_session` (session-scoped)

Primary SparkSession fixture. Automatically selects the appropriate backend:

1. **Databricks Runtime**: Uses existing session
2. **Databricks Connect**: Creates remote session
3. **Local**: Creates local SparkSession

```python
def test_spark_operations(spark_session):
    df = spark_session.createDataFrame([(1, "a")], ["id", "val"])
    assert df.count() == 1
```

#### `spark_config` (session-scoped)

Configuration for SparkSession. Override to customize:

```python
@pytest.fixture(scope="session")
def spark_config():
    return SparkConfig(
        app_name="my_tests",
        master="local[4]",
        config={"spark.sql.shuffle.partitions": "4"},
        use_databricks_connect=True,
        databricks_profile="my-workspace",
    )
```

#### `local_spark` (function-scoped)

Isolated SparkSession for tests that need complete isolation:

```python
@pytest.mark.slow
def test_isolated(local_spark):
    # Gets fresh SparkSession
    local_spark.sql("CREATE VIEW temp AS SELECT 1")
```

### Databricks Fixtures

#### `dbutils` (session-scoped)

Provides dbutils (real in Databricks, mock locally):

```python
def test_with_dbutils(dbutils):
    dbutils.fs.put("/tmp/test.txt", "content")
    assert dbutils.fs.head("/tmp/test.txt") == "content"
```

#### `workspace_client` (session-scoped)

Authenticated Databricks SDK WorkspaceClient:

```python
@pytest.mark.databricks
def test_workspace_ops(workspace_client):
    notebooks = workspace_client.workspace.list("/Users/me")
```

#### `databricks_config` (session-scoped)

Configuration for Databricks connectivity:

```python
@pytest.fixture(scope="session")
def databricks_config():
    return DatabricksConfig(
        profile="my-workspace",
        cluster_id="0123-456789-abcdef",
    )
```

### Path Fixtures

#### `temp_dbfs_path` (function-scoped)

Temporary DBFS path, auto-cleaned:

```python
def test_write_data(spark_session, temp_dbfs_path):
    df = spark_session.range(100)
    df.write.parquet(temp_dbfs_path)
```

#### `temp_workspace_path` (function-scoped)

Temporary workspace path for notebooks:

```python
def test_notebook_upload(workspace_client, temp_workspace_path):
    workspace_client.workspace.mkdirs(temp_workspace_path)
```

#### `temp_volume_path` (function-scoped)

Unity Catalog Volume path (requires configuration):

```python
def test_volume_write(spark_session, temp_volume_path):
    df = spark_session.range(100)
    df.write.parquet(temp_volume_path)
```

### Data Fixtures

#### `sample_dataframe` (function-scoped, factory)

Factory for creating test DataFrames:

```python
def test_with_data(sample_dataframe):
    # Built-in data types
    users = sample_dataframe("users", num_rows=100)
    transactions = sample_dataframe("transactions", num_rows=1000)
    
    # Custom data
    custom = sample_dataframe(
        data=[(1, "a"), (2, "b")],
        schema=["id", "val"]
    )
```

#### `test_table` (function-scoped, factory)

Factory for creating test tables (auto-cleaned):

```python
def test_table_ops(spark_session, test_table):
    table_name = test_table("users", num_rows=100)
    result = spark_session.sql(f"SELECT * FROM {table_name}")
```

#### `delta_table` (function-scoped, factory)

Factory for creating Delta tables:

```python
def test_delta(delta_table):
    path = delta_table("transactions", num_rows=1000)
    # Path to Delta table
```

### Notebook Fixtures

#### `notebook_context` (function-scoped)

Context object for notebook tests:

```python
def test_notebook_logic(notebook_context):
    class TestLogic(ContextAwareTestFixture):
        def test_something(self):
            df = self.spark.range(10)
            assert df.count() == 10
    
    runner = NotebookTestRunner(notebook_context)
    result = runner.run_fixture(TestLogic)
    result.assert_passed()
```

#### `notebook_runner` (function-scoped)

Runner for executing NotebookTestFixture classes:

```python
def test_multiple_fixtures(notebook_runner):
    result1 = notebook_runner.run_fixture(TestFeatureA)
    result2 = notebook_runner.run_fixture(TestFeatureB)
    
    result1.assert_passed()
    result2.assert_passed()
```

## Environment Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DBX_TEST_USE_CONNECT` | Set to `true` for Databricks Connect |
| `DBX_TEST_CLUSTER_ID` | Cluster ID for Connect |
| `DBX_TEST_PROFILE` | Databricks CLI profile |
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | Access token |
| `DBX_TEST_CATALOG` | Unity Catalog name |
| `DBX_TEST_VOLUME_CATALOG` | Catalog for volumes |
| `DBX_TEST_VOLUME_SCHEMA` | Schema for volumes |
| `DBX_TEST_CLEANUP` | Set to `false` to skip cleanup |

### Local Development

```bash
# Run with local Spark
pytest tests/

# Run with Databricks Connect
export DBX_TEST_USE_CONNECT=true
export DBX_TEST_PROFILE=my-workspace
pytest tests/
```

### CI/CD (Azure DevOps)

```yaml
- script: |
    export DATABRICKS_HOST=$(DATABRICKS_HOST)
    export DATABRICKS_TOKEN=$(DATABRICKS_TOKEN)
    export DBX_TEST_USE_CONNECT=true
    pytest tests/ --junitxml=results.xml
  displayName: 'Run Tests'
```

## Integration with Nutter

The fixtures are designed to work with Nutter-style `NotebookTestFixture` classes:

### Traditional Style (in Notebook)

```python
from dbx_test import NotebookTestFixture, run_notebook_tests

class TestMyFeature(NotebookTestFixture):
    def run_setup(self):
        self.df = spark.createDataFrame([(1, "a")], ["id", "val"])
    
    def test_count(self):
        assert self.df.count() == 1
    
    def run_cleanup(self):
        pass

results = run_notebook_tests(TestMyFeature)
dbutils.notebook.exit(json.dumps(results))
```

### Context-Aware Style (Hybrid)

```python
from dbx_test.fixtures.notebook import ContextAwareTestFixture

class TestMyFeature(ContextAwareTestFixture):
    def __init__(self, context=None, **kwargs):
        super().__init__(context, **kwargs)
    
    def run_setup(self):
        # self.spark is injected
        self.df = self.spark.createDataFrame([(1, "a")], ["id", "val"])
    
    def test_count(self):
        assert self.df.count() == 1
```

This can be run both:
- In a notebook with `run_notebook_tests()`
- Via pytest with `notebook_runner.run_fixture(TestMyFeature)`

## Best Practices

### 1. Session-Scoped Spark

Always use `spark_session` (session-scoped) for efficiency:

```python
# Good: Reuses session
def test_a(spark_session): ...
def test_b(spark_session): ...

# Avoid: Creates new session per test
def test_c(local_spark): ...  # Only when isolation needed
```

### 2. Use Factory Fixtures

Prefer factory fixtures for flexibility:

```python
# Good: Factory creates on demand
def test_users(sample_dataframe):
    df = sample_dataframe("users", num_rows=50)

# Less flexible: Fixed fixture
@pytest.fixture
def users_df(spark_session):
    return spark_session.createDataFrame(...)
```

### 3. Environment-Driven Configuration

Use environment variables for CI/CD:

```python
@pytest.fixture(scope="session")
def spark_config():
    return SparkConfig.from_env()  # Reads from env vars
```

### 4. Explicit Cleanup

Though auto-cleanup is provided, be explicit when needed:

```python
def test_with_cleanup(spark_session, data_manager):
    table = data_manager.create_table(df)
    # ... test ...
    # Automatic cleanup at session end
```

### 5. Mark Tests Appropriately

Use markers for conditional execution:

```python
@pytest.mark.databricks
def test_workspace_api(workspace_client):
    # Skipped if not connected to Databricks
    ...

@pytest.mark.slow
def test_large_dataset(spark_session):
    # Run with: pytest --run-slow
    ...
```

## Directory Structure

Recommended project structure:

```
my_project/
├── src/
│   └── my_package/
│       └── ...
├── tests/
│   ├── conftest.py          # Import dbx_test fixtures
│   ├── unit/
│   │   └── test_transforms.py
│   ├── integration/
│   │   └── test_pipelines.py
│   └── notebooks/
│       └── test_notebook.py  # Databricks notebook test
├── notebooks/
│   └── my_notebook.py
└── pyproject.toml
```

## Troubleshooting

### SparkSession Not Available

```
E   ImportError: No module named 'pyspark'
```

Install PySpark: `pip install pyspark`

### Databricks Connect Errors

```
E   DatabricksConnectException: ...
```

1. Check cluster ID is correct
2. Verify Databricks profile is configured
3. Ensure cluster is running

### Cleanup Not Running

Set `DBX_TEST_CLEANUP=true` or verify `cleanup_paths` fixture is included.

