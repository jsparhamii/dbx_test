"""
Fixture injection support for NotebookTestFixture.

This module provides a fixture registry and injection system that works
in both pytest and notebook execution contexts.

In pytest:
    - Standard @pytest.fixture decorators work automatically
    - Fixtures are injected via pytest's DI system

In notebooks:
    - Use @notebook_fixture to define fixtures
    - Use FixtureRegistry to register and resolve fixtures
    - Fixtures are injected based on parameter names

Example:
    from dbx_test import NotebookTestFixture
    from dbx_test.fixtures.injection import notebook_fixture, FixtureRegistry
    
    # Define a fixture
    @notebook_fixture
    def sample_data(spark):
        return spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
    
    # Use in test
    class TestWithFixtures(NotebookTestFixture):
        def test_data(self, sample_data):
            assert sample_data.count() == 2
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager


class FixtureScope(Enum):
    """Scope of fixture lifecycle."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SESSION = "session"


@dataclass
class FixtureDefinition:
    """Definition of a fixture."""
    name: str
    func: Callable
    scope: FixtureScope = FixtureScope.FUNCTION
    params: Optional[List[Any]] = None
    autouse: bool = False
    ids: Optional[List[str]] = None
    
    @property
    def dependencies(self) -> List[str]:
        """Get fixture dependencies from function signature."""
        sig = inspect.signature(self.func)
        # Skip 'self' if present
        params = [p for p in sig.parameters.keys() if p != 'self']
        return params


class FixtureRegistry:
    """Registry for fixtures in notebook context.
    
    This class manages fixture definitions and resolution for
    notebook execution where pytest's DI isn't available.
    
    Example:
        registry = FixtureRegistry()
        
        @registry.fixture
        def spark_data(spark_session):
            return spark_session.range(10)
        
        # Resolve fixtures for a test
        fixtures = registry.resolve(['spark_data'], {'spark_session': spark})
    """
    
    def __init__(self):
        self._fixtures: Dict[str, FixtureDefinition] = {}
        self._cache: Dict[str, Dict[str, Any]] = {
            "session": {},
            "module": {},
            "class": {},
            "function": {},
        }
        self._finalizers: List[Callable] = []
    
    def fixture(
        self,
        func: Optional[Callable] = None,
        *,
        scope: str = "function",
        params: Optional[List[Any]] = None,
        autouse: bool = False,
        ids: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> Union[Callable, FixtureDefinition]:
        """Register a fixture.
        
        Can be used as a decorator with or without arguments:
        
            @registry.fixture
            def my_fixture():
                return "value"
            
            @registry.fixture(scope="session")
            def session_fixture():
                return "session_value"
        
        Args:
            func: Function to register as fixture
            scope: Fixture scope (function, class, module, session)
            params: Parameters for parametrized fixtures
            autouse: Whether to auto-use this fixture
            ids: IDs for parametrized fixtures
            name: Override fixture name
        """
        def decorator(f: Callable) -> Callable:
            fixture_name = name or f.__name__
            fixture_def = FixtureDefinition(
                name=fixture_name,
                func=f,
                scope=FixtureScope(scope),
                params=params,
                autouse=autouse,
                ids=ids,
            )
            self._fixtures[fixture_name] = fixture_def
            
            # Mark function as a fixture
            f._fixture_definition = fixture_def
            return f
        
        if func is not None:
            return decorator(func)
        return decorator
    
    def register(self, name: str, value: Any, scope: str = "session"):
        """Register a pre-created fixture value.
        
        Useful for registering external objects like SparkSession.
        
        Args:
            name: Fixture name
            value: Fixture value
            scope: Cache scope
        """
        self._cache[scope][name] = value
    
    def resolve(
        self,
        names: List[str],
        provided: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve fixtures by name.
        
        Args:
            names: List of fixture names to resolve
            provided: Pre-provided fixture values
            
        Returns:
            Dictionary of resolved fixture values
        """
        provided = provided or {}
        resolved = {}
        
        for name in names:
            resolved[name] = self._resolve_single(name, provided, resolved)
        
        return resolved
    
    def _resolve_single(
        self,
        name: str,
        provided: Dict[str, Any],
        resolved: Dict[str, Any],
    ) -> Any:
        """Resolve a single fixture."""
        # Check if already provided
        if name in provided:
            return provided[name]
        
        # Check if already resolved
        if name in resolved:
            return resolved[name]
        
        # Check caches (session -> module -> class -> function)
        for scope in ["session", "module", "class", "function"]:
            if name in self._cache[scope]:
                return self._cache[scope][name]
        
        # Look up fixture definition
        if name not in self._fixtures:
            raise ValueError(
                f"Fixture '{name}' not found. "
                f"Available fixtures: {list(self._fixtures.keys())}"
            )
        
        fixture_def = self._fixtures[name]
        
        # Resolve dependencies first
        deps = {}
        for dep_name in fixture_def.dependencies:
            deps[dep_name] = self._resolve_single(dep_name, provided, {**resolved, **deps})
        
        # Call fixture function
        value = fixture_def.func(**deps)
        
        # Cache based on scope
        scope_key = fixture_def.scope.value
        self._cache[scope_key][name] = value
        
        return value
    
    def clear_cache(self, scope: Optional[str] = None):
        """Clear fixture cache.
        
        Args:
            scope: Scope to clear (None = all scopes)
        """
        if scope:
            self._cache[scope].clear()
        else:
            for scope_cache in self._cache.values():
                scope_cache.clear()
    
    def teardown(self):
        """Run all registered finalizers."""
        for finalizer in reversed(self._finalizers):
            try:
                finalizer()
            except Exception:
                pass
        self._finalizers.clear()
    
    def add_finalizer(self, func: Callable):
        """Add a teardown finalizer."""
        self._finalizers.append(func)


# Global registry for notebook fixtures
_global_registry = FixtureRegistry()


def notebook_fixture(
    func: Optional[Callable] = None,
    *,
    scope: str = "function",
    params: Optional[List[Any]] = None,
    autouse: bool = False,
    ids: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Callable:
    """Decorator to define a fixture for notebook execution.
    
    This works similarly to @pytest.fixture but is designed for
    notebook execution where pytest isn't available.
    
    Example:
        @notebook_fixture
        def test_data():
            return [1, 2, 3]
        
        @notebook_fixture(scope="session")
        def spark_data(spark_session):
            return spark_session.range(100)
    
    Args:
        func: Function to register as fixture
        scope: Fixture scope (function, class, module, session)
        params: Parameters for parametrized fixtures
        autouse: Whether to auto-use this fixture
        ids: IDs for parametrized fixtures
        name: Override fixture name
    """
    return _global_registry.fixture(
        func,
        scope=scope,
        params=params,
        autouse=autouse,
        ids=ids,
        name=name,
    )


def get_global_registry() -> FixtureRegistry:
    """Get the global fixture registry."""
    return _global_registry


def register_fixture(name: str, value: Any, scope: str = "session"):
    """Register a fixture value in the global registry.
    
    Example:
        # Register spark session for fixture injection
        register_fixture("spark_session", spark, scope="session")
        register_fixture("spark", spark, scope="session")
    """
    _global_registry.register(name, value, scope)


def get_fixture_params(func: Callable) -> List[str]:
    """Get parameter names for a function that might need fixture injection.
    
    Args:
        func: Function to inspect
        
    Returns:
        List of parameter names (excluding 'self')
    """
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


def is_fixture_param(name: str, registry: Optional[FixtureRegistry] = None) -> bool:
    """Check if a parameter name corresponds to a registered fixture.
    
    Args:
        name: Parameter name to check
        registry: Fixture registry to check (uses global if None)
        
    Returns:
        True if name is a registered fixture
    """
    reg = registry or _global_registry
    return name in reg._fixtures or any(name in cache for cache in reg._cache.values())


@contextmanager
def fixture_scope(scope: str = "function", registry: Optional[FixtureRegistry] = None):
    """Context manager for fixture scope lifecycle.
    
    Clears the specified scope cache on exit.
    
    Example:
        with fixture_scope("function"):
            # Run test
            pass
        # Function-scoped fixtures are cleared
    """
    reg = registry or _global_registry
    try:
        yield reg
    finally:
        reg.clear_cache(scope)


# Common fixture definitions that can be auto-registered
def setup_common_fixtures(registry: Optional[FixtureRegistry] = None):
    """Setup common fixtures in the registry.
    
    Call this after registering spark_session to enable
    common convenience fixtures.
    """
    reg = registry or _global_registry
    
    @reg.fixture(scope="session")
    def spark(spark_session):
        """Alias for spark_session."""
        return spark_session
    
    @reg.fixture(scope="function")
    def tmp_path():
        """Temporary directory path."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

