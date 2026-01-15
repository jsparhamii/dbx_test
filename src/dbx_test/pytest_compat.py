"""
Pytest decorator compatibility layer for NotebookTestFixture.

This module provides seamless integration between pytest decorators and
Nutter-style NotebookTestFixture classes, allowing the same test class
to work with both pytest execution and notebook execution.

Supported Decorators:
    - @pytest.mark.parametrize
    - @pytest.mark.skip
    - @pytest.mark.skipif  
    - @pytest.mark.xfail
    - @pytest.mark.usefixtures
    - @pytest.mark.timeout
    - Custom @pytest.mark.* markers

Usage:
    from dbx_test import NotebookTestFixture
    import pytest

    class TestMyFeature(NotebookTestFixture):
        @pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (2, 3, 5)])
        def test_add(self, x, y, expected):
            assert x + y == expected
        
        @pytest.mark.skip(reason="Not implemented yet")
        def test_future_feature(self):
            pass
        
        @pytest.mark.skipif(not HAS_DELTA, reason="Delta not available")
        def test_delta_operations(self):
            pass
"""

import functools
import inspect
import os
import sys
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, 
    Iterator, NamedTuple, TYPE_CHECKING
)
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    import pytest


class MarkerType(Enum):
    """Types of pytest markers we handle."""
    SKIP = "skip"
    SKIPIF = "skipif"
    XFAIL = "xfail"
    PARAMETRIZE = "parametrize"
    USEFIXTURES = "usefixtures"
    TIMEOUT = "timeout"
    CUSTOM = "custom"


@dataclass
class MarkInfo:
    """Information about a pytest mark applied to a test."""
    name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def marker_type(self) -> MarkerType:
        """Get the type of this marker."""
        try:
            return MarkerType(self.name)
        except ValueError:
            return MarkerType.CUSTOM


@dataclass
class ParameterSet:
    """A single parameter set for parametrized tests."""
    values: Tuple[Any, ...]
    id: Optional[str] = None
    marks: List[MarkInfo] = field(default_factory=list)


@dataclass
class TestOutcome:
    """Outcome of a test execution with marker awareness."""
    status: str  # "passed", "failed", "skipped", "xfailed", "xpassed"
    reason: Optional[str] = None
    exception: Optional[Exception] = None
    

def get_marks(func: Callable) -> List[MarkInfo]:
    """Extract pytest marks from a function.
    
    Args:
        func: Function to extract marks from
        
    Returns:
        List of MarkInfo objects
    """
    marks = []
    
    # Check for pytestmark attribute (set by pytest.mark decorators)
    pytestmark = getattr(func, "pytestmark", [])
    if not isinstance(pytestmark, list):
        pytestmark = [pytestmark]
    
    for mark in pytestmark:
        if hasattr(mark, "name"):
            marks.append(MarkInfo(
                name=mark.name,
                args=getattr(mark, "args", ()),
                kwargs=getattr(mark, "kwargs", {}),
            ))
    
    return marks


def get_parametrize_info(func: Callable) -> Optional[Tuple[str, List[ParameterSet]]]:
    """Extract parametrize information from a function.
    
    Args:
        func: Function to extract parametrize info from
        
    Returns:
        Tuple of (argnames, list of ParameterSets) or None
    """
    marks = get_marks(func)
    
    for mark in marks:
        if mark.name == "parametrize":
            if len(mark.args) >= 2:
                argnames = mark.args[0]
                argvalues = mark.args[1]
                ids = mark.kwargs.get("ids")
                
                # Parse argnames
                if isinstance(argnames, str):
                    argnames = [a.strip() for a in argnames.replace(",", " ").split()]
                
                # Parse argvalues into ParameterSets
                param_sets = []
                for i, values in enumerate(argvalues):
                    # Handle pytest.param() objects
                    if hasattr(values, "values"):
                        param_set = ParameterSet(
                            values=values.values,
                            id=getattr(values, "id", None),
                            marks=[MarkInfo(m.name, m.args, m.kwargs) 
                                   for m in getattr(values, "marks", [])],
                        )
                    else:
                        # Plain tuple/value
                        if not isinstance(values, (tuple, list)):
                            values = (values,)
                        param_id = ids[i] if ids and i < len(ids) else None
                        param_set = ParameterSet(values=tuple(values), id=param_id)
                    
                    param_sets.append(param_set)
                
                return (argnames, param_sets)
    
    return None


def should_skip(func: Callable) -> Tuple[bool, Optional[str]]:
    """Check if a function should be skipped based on markers.
    
    Args:
        func: Function to check
        
    Returns:
        Tuple of (should_skip, reason)
    """
    marks = get_marks(func)
    
    for mark in marks:
        if mark.name == "skip":
            reason = mark.kwargs.get("reason", mark.args[0] if mark.args else "")
            return (True, reason)
        
        elif mark.name == "skipif":
            if mark.args:
                condition = mark.args[0]
                # Evaluate condition if it's callable
                if callable(condition):
                    condition = condition()
                if condition:
                    reason = mark.kwargs.get("reason", "condition was True")
                    return (True, reason)
    
    return (False, None)


def get_xfail_info(func: Callable) -> Optional[Dict[str, Any]]:
    """Get xfail information from a function.
    
    Args:
        func: Function to check
        
    Returns:
        Dict with xfail info or None
    """
    marks = get_marks(func)
    
    for mark in marks:
        if mark.name == "xfail":
            condition = mark.args[0] if mark.args else True
            if callable(condition):
                condition = condition()
            
            if condition:
                return {
                    "reason": mark.kwargs.get("reason", ""),
                    "strict": mark.kwargs.get("strict", False),
                    "raises": mark.kwargs.get("raises"),
                    "run": mark.kwargs.get("run", True),
                }
    
    return None


def get_timeout(func: Callable) -> Optional[float]:
    """Get timeout value from a function.
    
    Args:
        func: Function to check
        
    Returns:
        Timeout in seconds or None
    """
    marks = get_marks(func)
    
    for mark in marks:
        if mark.name == "timeout":
            if mark.args:
                return float(mark.args[0])
            elif "timeout" in mark.kwargs:
                return float(mark.kwargs["timeout"])
    
    return None


def get_custom_markers(func: Callable) -> List[MarkInfo]:
    """Get custom markers from a function.
    
    Args:
        func: Function to check
        
    Returns:
        List of custom markers
    """
    marks = get_marks(func)
    known_markers = {"skip", "skipif", "xfail", "parametrize", "usefixtures", "timeout"}
    return [m for m in marks if m.name not in known_markers]


class MarkerAwareTestRunner:
    """Test runner that respects pytest markers.
    
    This class wraps test execution to honor pytest decorators like
    skip, skipif, xfail, and parametrize.
    """
    
    def __init__(self, fixture_instance: Any):
        """Initialize the runner.
        
        Args:
            fixture_instance: Instance of NotebookTestFixture
        """
        self.fixture = fixture_instance
        self._expanded_tests: Dict[str, List[Tuple[str, Callable, Optional[Tuple]]]] = {}
    
    def expand_parametrized_tests(self) -> List[Tuple[str, Callable, Optional[Tuple], Optional[str]]]:
        """Expand parametrized tests into individual test cases.
        
        Returns:
            List of (test_name, test_method, params, param_id) tuples
        """
        expanded = []
        
        for name in dir(self.fixture):
            if not name.startswith("test_"):
                continue
            
            method = getattr(self.fixture, name)
            if not callable(method):
                continue
            
            # Check for parametrize
            param_info = get_parametrize_info(method)
            
            if param_info:
                argnames, param_sets = param_info
                
                for i, param_set in enumerate(param_sets):
                    # Generate parameter ID
                    if param_set.id:
                        param_id = param_set.id
                    else:
                        param_id = "-".join(str(v) for v in param_set.values)
                    
                    test_name = f"{name}[{param_id}]"
                    expanded.append((test_name, method, param_set.values, param_id))
            else:
                expanded.append((name, method, None, None))
        
        return expanded
    
    def run_test(
        self,
        test_name: str,
        test_method: Callable,
        params: Optional[Tuple] = None,
    ) -> TestOutcome:
        """Run a single test with marker awareness.
        
        Args:
            test_name: Name of the test
            test_method: Test method to run
            params: Optional parameters for parametrized tests
            
        Returns:
            TestOutcome with result
        """
        # Check skip markers
        should_skip_test, skip_reason = should_skip(test_method)
        if should_skip_test:
            return TestOutcome(status="skipped", reason=skip_reason)
        
        # Check xfail markers
        xfail_info = get_xfail_info(test_method)
        
        # Check if xfail with run=False
        if xfail_info and not xfail_info.get("run", True):
            return TestOutcome(status="xfailed", reason=xfail_info.get("reason", ""))
        
        # Run the test
        try:
            if params:
                test_method(*params)
            else:
                test_method()
            
            # Test passed
            if xfail_info:
                # Expected to fail but passed
                if xfail_info.get("strict", False):
                    return TestOutcome(
                        status="failed",
                        reason=f"[XPASS(strict)] {xfail_info.get('reason', '')}"
                    )
                else:
                    return TestOutcome(status="xpassed", reason=xfail_info.get("reason", ""))
            
            return TestOutcome(status="passed")
        
        except Exception as e:
            if xfail_info:
                # Check if it's the expected exception type
                expected_raises = xfail_info.get("raises")
                if expected_raises is None or isinstance(e, expected_raises):
                    return TestOutcome(
                        status="xfailed",
                        reason=xfail_info.get("reason", ""),
                        exception=e,
                    )
            
            return TestOutcome(status="failed", exception=e)


def parametrize(argnames: str, argvalues: List, **kwargs) -> Callable:
    """Standalone parametrize decorator compatible with both pytest and notebook execution.
    
    This is an alias for pytest.mark.parametrize that works in notebook context.
    
    Args:
        argnames: Comma-separated parameter names
        argvalues: List of parameter value tuples
        **kwargs: Additional arguments (ids, indirect, etc.)
        
    Returns:
        Decorated function
    """
    try:
        import pytest
        return pytest.mark.parametrize(argnames, argvalues, **kwargs)
    except ImportError:
        # Fallback for pure notebook execution without pytest
        def decorator(func):
            if not hasattr(func, "pytestmark"):
                func.pytestmark = []
            
            # Create a mark-like object
            class MarkInfo:
                def __init__(self):
                    self.name = "parametrize"
                    self.args = (argnames, argvalues)
                    self.kwargs = kwargs
            
            func.pytestmark.append(MarkInfo())
            return func
        
        return decorator


def skip(reason: str = "") -> Callable:
    """Skip decorator compatible with both pytest and notebook execution.
    
    Args:
        reason: Reason for skipping
        
    Returns:
        Decorated function
    """
    try:
        import pytest
        return pytest.mark.skip(reason=reason)
    except ImportError:
        def decorator(func):
            if not hasattr(func, "pytestmark"):
                func.pytestmark = []
            
            class MarkInfo:
                def __init__(self):
                    self.name = "skip"
                    self.args = ()
                    self.kwargs = {"reason": reason}
            
            func.pytestmark.append(MarkInfo())
            return func
        
        return decorator


def skipif(condition: bool, reason: str = "") -> Callable:
    """Skipif decorator compatible with both pytest and notebook execution.
    
    Args:
        condition: Condition to evaluate
        reason: Reason for skipping
        
    Returns:
        Decorated function
    """
    try:
        import pytest
        return pytest.mark.skipif(condition, reason=reason)
    except ImportError:
        def decorator(func):
            if not hasattr(func, "pytestmark"):
                func.pytestmark = []
            
            class MarkInfo:
                def __init__(self):
                    self.name = "skipif"
                    self.args = (condition,)
                    self.kwargs = {"reason": reason}
            
            func.pytestmark.append(MarkInfo())
            return func
        
        return decorator


def xfail(reason: str = "", strict: bool = False, raises: type = None, run: bool = True) -> Callable:
    """Xfail decorator compatible with both pytest and notebook execution.
    
    Args:
        reason: Reason for expected failure
        strict: If True, passing test is a failure
        raises: Expected exception type
        run: Whether to run the test
        
    Returns:
        Decorated function
    """
    try:
        import pytest
        return pytest.mark.xfail(reason=reason, strict=strict, raises=raises, run=run)
    except ImportError:
        def decorator(func):
            if not hasattr(func, "pytestmark"):
                func.pytestmark = []
            
            class MarkInfo:
                def __init__(self):
                    self.name = "xfail"
                    self.args = (True,)  # condition
                    self.kwargs = {"reason": reason, "strict": strict, "raises": raises, "run": run}
            
            func.pytestmark.append(MarkInfo())
            return func
        
        return decorator


def mark(name: str, *args, **kwargs) -> Callable:
    """Create a custom marker compatible with both pytest and notebook execution.
    
    Args:
        name: Marker name
        *args: Marker arguments
        **kwargs: Marker keyword arguments
        
    Returns:
        Decorated function
    """
    try:
        import pytest
        return getattr(pytest.mark, name)(*args, **kwargs)
    except ImportError:
        def decorator(func):
            if not hasattr(func, "pytestmark"):
                func.pytestmark = []
            
            class MarkInfo:
                def __init__(self):
                    self.name = name
                    self.args = args
                    self.kwargs = kwargs
            
            func.pytestmark.append(MarkInfo())
            return func
        
        return decorator

