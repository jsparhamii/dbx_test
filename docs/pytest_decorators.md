# Pytest Decorator and Fixture Support

`dbx_test` provides native support for standard pytest decorators and fixtures in `NotebookTestFixture` classes. This allows you to write tests that work seamlessly in both:

- **Local pytest execution** - Run with `pytest tests/`
- **Databricks notebook execution** - Run with `run_notebook_tests()`
- **CI/CD pipelines** - Works with any pytest-compatible CI system

## Supported Decorators

| Decorator | Description | Works in Notebooks |
|-----------|-------------|-------------------|
| `@pytest.mark.parametrize` | Run test with multiple parameter sets | ✅ Yes |
| `@pytest.mark.skip` | Skip test unconditionally | ✅ Yes |
| `@pytest.mark.skipif` | Skip test conditionally | ✅ Yes |
| `@pytest.mark.xfail` | Mark test as expected to fail | ✅ Yes |
| `@pytest.mark.timeout` | Set test timeout | ⚠️ Pytest only |
| `@pytest.fixture` | Define and inject fixtures | ✅ Yes (with registration) |
| Custom `@pytest.mark.*` | Custom markers | ✅ Yes (filtering in pytest) |

## Quick Examples

### Parametrize

```python
import pytest
from dbx_test import NotebookTestFixture

class TestCalculations(NotebookTestFixture):
    @pytest.mark.parametrize("x,y,expected", [
        (1, 2, 3),
        (2, 3, 5),
        (10, 20, 30),
    ])
    def test_addition(self, x, y, expected):
        assert x + y == expected
```

**Output:**
```
Running test_addition[1-2-3]...
  ✓ PASSED
Running test_addition[2-3-5]...
  ✓ PASSED
Running test_addition[10-20-30]...
  ✓ PASSED
```

### Skip and SkipIf

```python
import pytest
import os
from dbx_test import NotebookTestFixture

class TestFeatures(NotebookTestFixture):
    @pytest.mark.skip(reason="Feature not implemented yet")
    def test_future_feature(self):
        pass
    
    @pytest.mark.skipif(
        os.environ.get("DATABRICKS_RUNTIME_VERSION") is None,
        reason="Requires Databricks"
    )
    def test_databricks_only(self):
        # This test only runs in Databricks
        pass
```

**Output:**
```
Running test_future_feature...
  ⊘ SKIPPED: Feature not implemented yet
Running test_databricks_only...
  ⊘ SKIPPED: Requires Databricks
```

### XFail (Expected Failures)

```python
import pytest
from dbx_test import NotebookTestFixture

class TestEdgeCases(NotebookTestFixture):
    @pytest.mark.xfail(reason="Known bug in calculation")
    def test_known_bug(self):
        # This test fails but is expected to
        assert 1 + 1 == 3
    
    @pytest.mark.xfail(strict=True, reason="Must fail")
    def test_strict_xfail(self):
        # If this passes, it's a test failure
        assert 1 + 1 == 3
    
    @pytest.mark.xfail(raises=ZeroDivisionError)
    def test_expected_exception(self):
        result = 1 / 0
```

**Output:**
```
Running test_known_bug...
  ⊗ XFAILED: Expected failure: Known bug in calculation
```

### Custom Markers

```python
import pytest
from dbx_test import NotebookTestFixture

class TestWithMarkers(NotebookTestFixture):
    @pytest.mark.slow
    def test_slow_operation(self):
        import time
        time.sleep(1)
        assert True
    
    @pytest.mark.integration
    def test_integration(self):
        assert True
    
    @pytest.mark.databricks
    def test_databricks_specific(self):
        assert True
```

**Filtering with markers:**
```bash
# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run integration tests
pytest -m integration
```

## Parametrize Advanced Usage

### With Custom IDs

```python
@pytest.mark.parametrize("input_val,expected", [
    pytest.param(1, 1, id="identity"),
    pytest.param(2, 4, id="square_of_two"),
    pytest.param(3, 9, id="square_of_three"),
])
def test_squares(self, input_val, expected):
    assert input_val ** 2 == expected
```

### Multiple Parametrize (Cross-Product)

```python
@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [10, 20])
def test_combinations(self, a, b):
    # Runs 4 times: (1,10), (1,20), (2,10), (2,20)
    assert a + b > 0
```

### Parametrize with Marks

```python
@pytest.mark.parametrize("value", [
    pytest.param(1, id="normal"),
    pytest.param(0, id="zero", marks=pytest.mark.xfail),
    pytest.param(-1, id="negative", marks=pytest.mark.skip),
])
def test_values(self, value):
    assert value > 0
```

## Fixture Injection

Test methods can receive fixtures as parameters, just like standard pytest tests.

### In Pytest Execution

Standard `@pytest.fixture` decorators work automatically:

```python
import pytest
from dbx_test import NotebookTestFixture

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

@pytest.fixture(scope="session")
def database_connection():
    conn = create_connection()
    yield conn
    conn.close()

class TestWithFixtures(NotebookTestFixture):
    def test_with_fixture(self, sample_data):
        """Fixture is automatically injected."""
        assert len(sample_data) == 5
    
    def test_with_session_fixture(self, database_connection):
        """Session-scoped fixture."""
        assert database_connection.is_connected
```

### In Notebook Execution

For notebook execution, register fixtures before running tests:

```python
from dbx_test import NotebookTestFixture, run_notebook_tests
from dbx_test.fixtures import register_fixture, notebook_fixture

# Method 1: Register values directly
register_fixture("spark_session", spark)
register_fixture("dbutils", dbutils)
register_fixture("sample_data", [1, 2, 3, 4, 5])

# Method 2: Use @notebook_fixture decorator
@notebook_fixture(scope="session")
def database_connection():
    return create_connection()

# Now fixtures are available to tests
class TestWithFixtures(NotebookTestFixture):
    def test_with_spark(self, spark_session):
        df = spark_session.range(10)
        assert df.count() == 10
    
    def test_with_data(self, sample_data):
        assert len(sample_data) == 5

results = run_notebook_tests(TestWithFixtures)
```

### Fixture Scopes

| Scope | Description |
|-------|-------------|
| `function` | New instance for each test (default) |
| `class` | Shared within test class |
| `module` | Shared within module |
| `session` | Shared across all tests |

### Combining Fixtures with Parametrize

```python
@pytest.fixture
def records():
    return [{"id": 1}, {"id": 2}, {"id": 3}]

class TestCombined(NotebookTestFixture):
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_record_by_index(self, records, index):
        """Both fixture and parametrize work together."""
        assert records[index]["id"] == index + 1
```

## Combining Decorators

Decorators can be combined:

```python
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_performance(self, size):
    data = list(range(size))
    assert len(data) == size
```

## Test Status Reference

| Status | Symbol | Description |
|--------|--------|-------------|
| `passed` | ✓ | Test passed |
| `failed` | ✗ | Test failed |
| `error` | ✗ | Test had an error (exception) |
| `skipped` | ⊘ | Test was skipped |
| `xfailed` | ⊗ | Test failed as expected |
| `xpassed` | ⊕ | Test passed unexpectedly (xfail but passed) |

## Running Tests

### Via pytest (Local/CI)

```bash
# Run all tests
pytest tests/

# Run with markers
pytest -m "not slow" tests/

# Verbose output
pytest -v tests/

# Show skip reasons
pytest -rs tests/
```

### Via Databricks Notebook

```python
from dbx_test import run_notebook_tests

class TestMyFeature(NotebookTestFixture):
    @pytest.mark.parametrize("x", [1, 2, 3])
    def test_values(self, x):
        assert x > 0

results = run_notebook_tests(TestMyFeature)
dbutils.notebook.exit(json.dumps(results))
```

### Via dbx_test CLI

```bash
# Run tests in Databricks workspace
dbx_test run --tests-dir /Workspace/Tests
```

## Limitations

### In Notebook Execution

1. **`@pytest.mark.timeout`** - Timeout is not enforced in notebook execution (works in pytest)

2. **`@pytest.mark.usefixtures`** - Pytest fixtures are not injected in notebook execution

3. **Marker Filtering** - `-m` marker filtering is a pytest feature, not available in notebooks

### In Pytest Execution

1. **Full pytest compatibility** - All decorators work as expected

2. **Fixtures** - Full pytest fixture support when running via pytest

## Best Practices

### 1. Use Parametrize for Data-Driven Tests

```python
# Good: Test multiple scenarios with parametrize
@pytest.mark.parametrize("input,expected", test_cases)
def test_transform(self, input, expected):
    result = transform(input)
    assert result == expected
```

### 2. Use Skip for Environment-Specific Tests

```python
@pytest.mark.skipif(
    not HAS_DELTA,
    reason="Delta Lake required"
)
def test_delta_merge(self):
    # Only runs when Delta is available
    pass
```

### 3. Use XFail for Known Issues

```python
@pytest.mark.xfail(reason="Issue #123 - Fix in progress")
def test_known_bug(self):
    # Track known bugs without failing CI
    pass
```

### 4. Use Custom Markers for Organization

```python
@pytest.mark.smoke
def test_basic_connectivity(self):
    # Run smoke tests with: pytest -m smoke
    pass

@pytest.mark.integration
def test_full_pipeline(self):
    # Run integration tests with: pytest -m integration
    pass
```

## Migration from Pure Nutter

If you have existing Nutter tests, they continue to work unchanged:

```python
# Before (still works)
class TestOldStyle(NotebookTestFixture):
    def test_something(self):
        assert True

# After (enhanced with decorators)
class TestNewStyle(NotebookTestFixture):
    @pytest.mark.parametrize("x", [1, 2, 3])
    def test_something(self, x):
        assert x > 0
```

No breaking changes - existing tests work as-is, and you can incrementally add decorators.

