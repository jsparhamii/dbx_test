"""
Databricks client and configuration fixtures for pytest.

Provides fixtures for:
- Databricks SDK WorkspaceClient
- Databricks Connect configuration
- dbutils mock/real access
- Environment detection
"""

import os
import pytest
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass, field
from unittest.mock import MagicMock


@dataclass
class DatabricksConfig:
    """Configuration for Databricks connectivity.
    
    Attributes:
        host: Databricks workspace URL
        token: Personal access token (prefer token_env for security)
        token_env: Environment variable containing the token
        profile: Databricks CLI profile name
        cluster_id: Default cluster ID for operations
        warehouse_id: SQL warehouse ID for SQL operations
    """
    host: Optional[str] = None
    token: Optional[str] = None
    token_env: Optional[str] = None
    profile: Optional[str] = None
    cluster_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "DatabricksConfig":
        """Create DatabricksConfig from environment variables.
        
        Environment Variables:
            DATABRICKS_HOST: Workspace URL
            DATABRICKS_TOKEN: Personal access token
            DATABRICKS_PROFILE: CLI profile name (alternative to token)
            DBX_TEST_CLUSTER_ID: Default cluster ID
            DBX_TEST_WAREHOUSE_ID: SQL warehouse ID
        """
        return cls(
            host=os.environ.get("DATABRICKS_HOST"),
            token=os.environ.get("DATABRICKS_TOKEN"),
            profile=os.environ.get("DATABRICKS_PROFILE"),
            cluster_id=os.environ.get("DBX_TEST_CLUSTER_ID"),
            warehouse_id=os.environ.get("DBX_TEST_WAREHOUSE_ID"),
        )
    
    def get_auth_kwargs(self) -> Dict[str, Any]:
        """Get authentication kwargs for WorkspaceClient."""
        kwargs = {}
        
        if self.host:
            kwargs["host"] = self.host
        
        if self.profile:
            kwargs["profile"] = self.profile
        elif self.token:
            kwargs["token"] = self.token
        elif self.token_env:
            token = os.environ.get(self.token_env)
            if token:
                kwargs["token"] = token
        
        return kwargs


def is_databricks_runtime() -> bool:
    """Check if code is running in Databricks Runtime.
    
    Returns:
        True if running in Databricks Runtime, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _get_real_dbutils():
    """Get real dbutils when running in Databricks."""
    try:
        # Try to get dbutils from the IPython kernel
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            return ipython.user_ns.get("dbutils")
    except Exception:
        pass
    
    try:
        # Alternative: import from pyspark.dbutils
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return DBUtils(spark)
    except Exception:
        pass
    
    return None


class MockDBUtils:
    """Mock dbutils for local testing.
    
    Provides a minimal mock implementation of Databricks dbutils
    for running tests locally without a Databricks connection.
    """
    
    def __init__(self):
        self._widgets = {}
        self._secrets = {}
        self._notebook_exit_value = None
    
    class fs:
        """Mock filesystem operations."""
        
        _files: Dict[str, bytes] = {}
        
        @classmethod
        def ls(cls, path: str):
            """List files in a path."""
            return [MagicMock(path=p, name=p.split("/")[-1], size=len(c))
                    for p, c in cls._files.items() if p.startswith(path)]
        
        @classmethod
        def mkdirs(cls, path: str):
            """Create directories (no-op in mock)."""
            pass
        
        @classmethod
        def rm(cls, path: str, recurse: bool = False):
            """Remove files."""
            to_remove = [p for p in cls._files if p.startswith(path)]
            for p in to_remove:
                del cls._files[p]
        
        @classmethod
        def put(cls, path: str, contents: str, overwrite: bool = False):
            """Write a file."""
            cls._files[path] = contents.encode() if isinstance(contents, str) else contents
        
        @classmethod
        def head(cls, path: str, max_bytes: int = 65536) -> str:
            """Read the beginning of a file."""
            if path in cls._files:
                return cls._files[path][:max_bytes].decode()
            raise FileNotFoundError(f"File not found: {path}")
        
        @classmethod
        def cp(cls, source: str, dest: str, recurse: bool = False):
            """Copy files."""
            if source in cls._files:
                cls._files[dest] = cls._files[source]
        
        @classmethod
        def mv(cls, source: str, dest: str, recurse: bool = False):
            """Move files."""
            if source in cls._files:
                cls._files[dest] = cls._files[source]
                del cls._files[source]
    
    class widgets:
        """Mock widget operations."""
        
        _values: Dict[str, str] = {}
        
        @classmethod
        def text(cls, name: str, default_value: str = "", label: str = ""):
            """Create a text widget."""
            if name not in cls._values:
                cls._values[name] = default_value
        
        @classmethod
        def dropdown(cls, name: str, default_value: str = "", choices: list = None, label: str = ""):
            """Create a dropdown widget."""
            if name not in cls._values:
                cls._values[name] = default_value
        
        @classmethod
        def get(cls, name: str) -> str:
            """Get widget value."""
            return cls._values.get(name, "")
        
        @classmethod
        def getAll(cls) -> Dict[str, str]:
            """Get all widget values."""
            return cls._values.copy()
        
        @classmethod
        def remove(cls, name: str):
            """Remove a widget."""
            cls._values.pop(name, None)
        
        @classmethod
        def removeAll(cls):
            """Remove all widgets."""
            cls._values.clear()
    
    class secrets:
        """Mock secrets operations."""
        
        _secrets: Dict[str, Dict[str, str]] = {}
        
        @classmethod
        def get(cls, scope: str, key: str) -> str:
            """Get a secret value."""
            # Check environment variable first (for CI)
            env_key = f"DBX_SECRET_{scope.upper()}_{key.upper()}"
            env_value = os.environ.get(env_key)
            if env_value:
                return env_value
            
            # Then check mock secrets
            if scope in cls._secrets and key in cls._secrets[scope]:
                return cls._secrets[scope][key]
            
            raise ValueError(f"Secret not found: {scope}/{key}")
        
        @classmethod
        def list(cls, scope: str):
            """List secrets in a scope."""
            if scope in cls._secrets:
                return [MagicMock(key=k) for k in cls._secrets[scope]]
            return []
        
        @classmethod
        def listScopes(cls):
            """List all scopes."""
            return [MagicMock(name=s) for s in cls._secrets]
        
        @classmethod
        def _set_secret(cls, scope: str, key: str, value: str):
            """Set a mock secret (for testing)."""
            if scope not in cls._secrets:
                cls._secrets[scope] = {}
            cls._secrets[scope][key] = value
    
    class notebook:
        """Mock notebook operations."""
        
        _exit_value: Optional[str] = None
        
        @classmethod
        def exit(cls, value: str):
            """Exit notebook with a value."""
            cls._exit_value = value
            # Don't actually exit in tests
        
        @classmethod
        def run(cls, path: str, timeout_seconds: int = 300, arguments: Dict[str, str] = None) -> str:
            """Run a notebook (mock returns empty string)."""
            return ""
        
        @classmethod
        def getContext(cls):
            """Get notebook context (mock)."""
            return MagicMock(
                toJson=MagicMock(return_value='{"tags": {}}'),
                tags=MagicMock(return_value={"user": "test@example.com"})
            )


@pytest.fixture(scope="session")
def databricks_config() -> DatabricksConfig:
    """Session-scoped Databricks configuration fixture.
    
    Override this fixture to customize Databricks settings:
    
    Example:
        @pytest.fixture(scope="session")
        def databricks_config():
            return DatabricksConfig(
                profile="my-workspace",
                cluster_id="0123-456789-abcdef"
            )
    """
    return DatabricksConfig.from_env()


@pytest.fixture(scope="session")
def workspace_client(databricks_config: DatabricksConfig) -> Generator:
    """Session-scoped Databricks WorkspaceClient fixture.
    
    Provides authenticated access to Databricks workspace APIs.
    
    Example:
        def test_list_notebooks(workspace_client):
            notebooks = workspace_client.workspace.list("/Users/me")
            assert len(list(notebooks)) > 0
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        auth_kwargs = databricks_config.get_auth_kwargs()
        client = WorkspaceClient(**auth_kwargs)
        
        yield client
    
    except ImportError:
        pytest.skip("databricks-sdk not installed")


@pytest.fixture(scope="session")
def databricks_client(workspace_client):
    """Alias for workspace_client for backwards compatibility."""
    return workspace_client


@pytest.fixture(scope="session")
def dbutils() -> Generator:
    """Session-scoped dbutils fixture.
    
    Provides either real dbutils (in Databricks) or a mock (locally).
    
    The mock supports common operations:
    - dbutils.fs: File operations
    - dbutils.widgets: Widget operations
    - dbutils.secrets: Secret access (reads from env vars as fallback)
    - dbutils.notebook: Notebook operations
    
    Example:
        def test_with_dbutils(dbutils, spark_session):
            # Works both locally and in Databricks
            dbutils.fs.put("/tmp/test.txt", "hello")
            content = dbutils.fs.head("/tmp/test.txt")
            assert content == "hello"
    """
    if is_databricks_runtime():
        real_dbutils = _get_real_dbutils()
        if real_dbutils:
            yield real_dbutils
            return
    
    # Use mock dbutils
    mock = MockDBUtils()
    yield mock
    
    # Cleanup mock state
    mock.fs._files.clear()
    mock.widgets._values.clear()
    mock.secrets._secrets.clear()

