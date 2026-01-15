# Building dbx_test

This guide explains how to build the `dbx_test` package locally into a wheel (`.whl`) file.

## Prerequisites

- Python 3.10+
- The `build` package

## Building the Package

### Step 1: Create a Virtual Environment

Modern Python installations (especially via Homebrew) are externally managed and require a virtual environment:

```bash
cd /path/to/dbx_test
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Build Tools

```bash
pip install build
```

### Step 3: Build the Wheel

```bash
python -m build --wheel
```

### Complete One-Liner

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install build && python -m build --wheel
```

### Expected Output

```
Successfully built dbx_test-0.1.0-py3-none-any.whl
```

The wheel file will be created in the `dist/` directory:

```
dist/
└── dbx_test-0.1.0-py3-none-any.whl  (~57 KB)
```

### Package Contents

The wheel includes the following modules:

```
dbx_test/
├── __init__.py
├── artifacts.py
├── bundle.py
├── cli.py
├── config.py
├── discovery.py
├── notebook_runner.py
├── plugin.py
├── reporting.py
├── runner_remote.py
├── testing.py
├── fixtures/
│   ├── __init__.py
│   ├── data.py
│   ├── databricks.py
│   ├── notebook.py
│   ├── paths.py
│   └── spark.py
└── utils/
    ├── __init__.py
    ├── databricks.py
    ├── notebook.py
    └── validation.py
```

## Installing the Local Wheel

### Install Directly

```bash
pip install dist/dbx_test-0.1.0-py3-none-any.whl
```

### Install with Optional Dependencies

```bash
# With Spark support
pip install "dist/dbx_test-0.1.0-py3-none-any.whl[spark]"

# With Databricks Connect support
pip install "dist/dbx_test-0.1.0-py3-none-any.whl[connect]"

# With all optional dependencies
pip install "dist/dbx_test-0.1.0-py3-none-any.whl[all]"

# With development dependencies
pip install "dist/dbx_test-0.1.0-py3-none-any.whl[dev]"
```

## Development Install

For active development, install in editable mode:

```bash
pip install -e ".[dev]"
```

This installs the package in development mode, so changes to the source code are immediately available without rebuilding.

## Clean Build

Remove previous build artifacts before rebuilding:

```bash
rm -rf build/ dist/ src/dbx_test.egg-info
python -m build --wheel
```

## Uploading to DBFS

After building, upload the wheel to DBFS for use in Databricks:

```bash
# Using Databricks CLI
databricks fs cp dist/dbx_test-0.1.0-py3-none-any.whl dbfs:/FileStore/wheels/
```

## Installing in Databricks Notebook

```python
# From DBFS
%pip install /dbfs/FileStore/wheels/dbx_test-0.1.0-py3-none-any.whl

# Or directly from GitHub
%pip install git+https://github.com/your-org/dbx_test.git
```

## Versioning

Update the version in `pyproject.toml` before releasing:

```toml
[project]
version = "0.2.0"
```

## Build Troubleshooting

### "externally-managed-environment" Error

This occurs with Homebrew Python. Solution: use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install build
python -m build --wheel
```

### Missing `build` Module

```bash
pip install --upgrade build
```

### Permission Errors

Use a virtual environment (recommended) or:

```bash
pip install --user build
python -m build
```

### Old Artifacts Causing Issues

```bash
rm -rf build/ dist/ src/dbx_test.egg-info __pycache__ .pytest_cache
python -m build --wheel
```

### License Deprecation Warnings

You may see warnings about license format:

```
SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated
```

These are warnings only and don't affect the build. To fix, update `pyproject.toml`:

```toml
# Change from:
license = {text = "MIT"}

# To:
license = "MIT"
```

## Verifying the Build

Check the wheel contents:

```bash
python -m zipfile -l dist/dbx_test-0.1.0-py3-none-any.whl
```

Verify the package installs correctly:

```bash
pip install dist/dbx_test-0.1.0-py3-none-any.whl
python -c "from dbx_test.fixtures import spark_session; print('✓ Import successful')"
```

## CI/CD Build Script

Example script for CI pipelines:

```bash
#!/bin/bash
set -e

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install build dependencies
pip install build

# Clean previous builds
rm -rf build/ dist/ src/dbx_test.egg-info

# Build the wheel
python -m build --wheel

# Output the built file
echo "Built wheel:"
ls -la dist/*.whl
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `python3 -m venv .venv` | Create virtual environment |
| `source .venv/bin/activate` | Activate virtual environment |
| `pip install build` | Install build tools |
| `python -m build --wheel` | Build wheel only |
| `python -m build` | Build wheel + source dist |
| `pip install dist/*.whl` | Install the wheel |
