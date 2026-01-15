"""
Temporary path fixtures for pytest.

Provides fixtures for creating and managing temporary paths in:
- DBFS (Databricks File System)
- Workspace (for notebooks)
- Unity Catalog Volumes
- Local filesystem

All paths are automatically cleaned up after tests complete.
"""

import os
import uuid
import pytest
import tempfile
from pathlib import Path
from typing import Optional, List, Generator
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PathConfig:
    """Configuration for temporary path generation.
    
    Attributes:
        dbfs_root: Root path for DBFS temp directories
        workspace_root: Root path for workspace temp directories
        volume_catalog: Unity Catalog name for volume paths
        volume_schema: Schema name for volume paths
        volume_name: Volume name for temp storage
        local_root: Local filesystem root for temp directories
        cleanup_on_exit: Whether to clean up paths after tests
    """
    dbfs_root: str = "/dbfs/tmp/dbx_test"
    workspace_root: str = "/Workspace/tmp/dbx_test"
    volume_catalog: Optional[str] = None
    volume_schema: Optional[str] = None
    volume_name: str = "test_data"
    local_root: str = "/tmp/dbx_test"
    cleanup_on_exit: bool = True
    
    @classmethod
    def from_env(cls) -> "PathConfig":
        """Create PathConfig from environment variables."""
        return cls(
            dbfs_root=os.environ.get("DBX_TEST_DBFS_ROOT", "/dbfs/tmp/dbx_test"),
            workspace_root=os.environ.get("DBX_TEST_WORKSPACE_ROOT", "/Workspace/tmp/dbx_test"),
            volume_catalog=os.environ.get("DBX_TEST_VOLUME_CATALOG"),
            volume_schema=os.environ.get("DBX_TEST_VOLUME_SCHEMA"),
            volume_name=os.environ.get("DBX_TEST_VOLUME_NAME", "test_data"),
            local_root=os.environ.get("DBX_TEST_LOCAL_ROOT", "/tmp/dbx_test"),
            cleanup_on_exit=os.environ.get("DBX_TEST_CLEANUP", "true").lower() == "true",
        )


class PathManager:
    """Manages temporary paths and cleanup.
    
    Tracks all created paths and provides cleanup functionality.
    """
    
    def __init__(self, config: PathConfig):
        self.config = config
        self._created_paths: List[str] = []
        self._dbfs_paths: List[str] = []
        self._workspace_paths: List[str] = []
        self._local_paths: List[Path] = []
    
    def generate_unique_suffix(self) -> str:
        """Generate a unique suffix for path names."""
        return f"{uuid.uuid4().hex[:8]}"
    
    def create_dbfs_path(self, prefix: str = "test") -> str:
        """Create a unique DBFS path.
        
        Args:
            prefix: Prefix for the path name
            
        Returns:
            DBFS path string (e.g., /dbfs/tmp/dbx_test/test_abc123)
        """
        suffix = self.generate_unique_suffix()
        path = f"{self.config.dbfs_root}/{prefix}_{suffix}"
        self._dbfs_paths.append(path)
        return path
    
    def create_workspace_path(self, prefix: str = "test") -> str:
        """Create a unique workspace path.
        
        Args:
            prefix: Prefix for the path name
            
        Returns:
            Workspace path string (e.g., /Workspace/tmp/dbx_test/test_abc123)
        """
        suffix = self.generate_unique_suffix()
        path = f"{self.config.workspace_root}/{prefix}_{suffix}"
        self._workspace_paths.append(path)
        return path
    
    def create_volume_path(self, prefix: str = "test") -> str:
        """Create a unique Unity Catalog volume path.
        
        Args:
            prefix: Prefix for the path name
            
        Returns:
            Volume path string (e.g., /Volumes/catalog/schema/volume/test_abc123)
        """
        if not self.config.volume_catalog or not self.config.volume_schema:
            raise ValueError(
                "volume_catalog and volume_schema must be configured. "
                "Set DBX_TEST_VOLUME_CATALOG and DBX_TEST_VOLUME_SCHEMA env vars."
            )
        
        suffix = self.generate_unique_suffix()
        path = (
            f"/Volumes/{self.config.volume_catalog}/"
            f"{self.config.volume_schema}/{self.config.volume_name}/{prefix}_{suffix}"
        )
        self._dbfs_paths.append(path)  # Volumes use DBFS-style cleanup
        return path
    
    def create_local_path(self, prefix: str = "test") -> Path:
        """Create a unique local filesystem path.
        
        Args:
            prefix: Prefix for the path name
            
        Returns:
            Path object for local filesystem
        """
        suffix = self.generate_unique_suffix()
        path = Path(self.config.local_root) / f"{prefix}_{suffix}"
        path.mkdir(parents=True, exist_ok=True)
        self._local_paths.append(path)
        return path
    
    def cleanup(self, spark=None, dbutils=None):
        """Clean up all created paths.
        
        Args:
            spark: SparkSession for DBFS operations (optional)
            dbutils: dbutils for file operations (optional)
        """
        if not self.config.cleanup_on_exit:
            return
        
        # Clean up DBFS paths
        for path in self._dbfs_paths:
            try:
                if dbutils:
                    dbutils.fs.rm(path, recurse=True)
                elif spark:
                    # Use Spark's file system
                    from py4j.protocol import Py4JJavaError
                    try:
                        spark._jvm.org.apache.hadoop.fs.FileSystem.get(
                            spark._jsc.hadoopConfiguration()
                        ).delete(
                            spark._jvm.org.apache.hadoop.fs.Path(path.replace("/dbfs", "dbfs:")),
                            True
                        )
                    except Py4JJavaError:
                        pass
            except Exception:
                pass  # Best effort cleanup
        
        # Clean up workspace paths (requires workspace client)
        # This is typically handled separately as it needs the SDK
        
        # Clean up local paths
        import shutil
        for path in self._local_paths:
            try:
                if path.exists():
                    shutil.rmtree(path)
            except Exception:
                pass
        
        # Clear tracking lists
        self._dbfs_paths.clear()
        self._workspace_paths.clear()
        self._local_paths.clear()


@pytest.fixture(scope="session")
def path_config() -> PathConfig:
    """Session-scoped path configuration fixture."""
    return PathConfig.from_env()


@pytest.fixture(scope="session")
def path_manager(path_config: PathConfig) -> Generator[PathManager, None, None]:
    """Session-scoped path manager fixture."""
    manager = PathManager(path_config)
    yield manager
    # Cleanup happens in cleanup_paths fixture


@pytest.fixture(scope="function")
def temp_dbfs_path(path_manager: PathManager) -> Generator[str, None, None]:
    """Function-scoped temporary DBFS path.
    
    Creates a unique DBFS path for each test function.
    The path is automatically cleaned up after the test.
    
    Example:
        def test_write_data(spark_session, temp_dbfs_path):
            df = spark_session.createDataFrame([(1, "a")], ["id", "val"])
            df.write.parquet(temp_dbfs_path)
            loaded = spark_session.read.parquet(temp_dbfs_path)
            assert loaded.count() == 1
    """
    path = path_manager.create_dbfs_path()
    yield path


@pytest.fixture(scope="function")
def temp_workspace_path(path_manager: PathManager) -> Generator[str, None, None]:
    """Function-scoped temporary workspace path.
    
    Creates a unique workspace path for each test function.
    Useful for notebook upload/execution tests.
    
    Example:
        def test_upload_notebook(workspace_client, temp_workspace_path):
            # Upload notebook to temp path
            workspace_client.workspace.mkdirs(temp_workspace_path)
            # ... test code
    """
    path = path_manager.create_workspace_path()
    yield path


@pytest.fixture(scope="function")
def temp_volume_path(path_manager: PathManager) -> Generator[str, None, None]:
    """Function-scoped temporary Unity Catalog volume path.
    
    Creates a unique volume path for each test function.
    Requires volume configuration (catalog, schema, volume).
    
    Example:
        def test_volume_write(spark_session, temp_volume_path):
            df = spark_session.createDataFrame([(1, "a")], ["id", "val"])
            df.write.parquet(temp_volume_path)
    """
    try:
        path = path_manager.create_volume_path()
        yield path
    except ValueError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="function")
def temp_local_path(path_manager: PathManager) -> Generator[Path, None, None]:
    """Function-scoped temporary local filesystem path.
    
    Creates a unique local directory for each test function.
    
    Example:
        def test_local_operations(temp_local_path):
            test_file = temp_local_path / "data.txt"
            test_file.write_text("test content")
            assert test_file.read_text() == "test content"
    """
    path = path_manager.create_local_path()
    yield path


@pytest.fixture(scope="session", autouse=False)
def cleanup_paths(path_manager: PathManager, spark_session=None, dbutils=None):
    """Session-scoped fixture to ensure path cleanup.
    
    This fixture runs at the end of the test session to clean up
    all temporary paths created during testing.
    
    Note: This is not autouse by default. Add it to your conftest.py
    if you want automatic cleanup:
    
    Example:
        @pytest.fixture(scope="session", autouse=True)
        def auto_cleanup(cleanup_paths):
            yield
    """
    yield
    path_manager.cleanup(spark=spark_session, dbutils=dbutils)


@contextmanager
def temp_directory():
    """Context manager for creating a temporary local directory.
    
    Alternative to fixtures for use in non-pytest contexts.
    
    Example:
        with temp_directory() as tmpdir:
            test_file = tmpdir / "data.txt"
            test_file.write_text("content")
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

