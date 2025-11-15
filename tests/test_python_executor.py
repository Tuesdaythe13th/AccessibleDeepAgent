"""
Unit tests for Python executor security
"""

import pytest
import asyncio
from src.tools.python_executor import (
    PythonExecutor,
    UnsafeCodeError,
    GenericRuntime
)


class TestPythonExecutorSecurity:
    """Tests for unsafe code detection"""

    def test_safe_code_executes(self):
        """Safe code should execute without errors"""
        runtime = GenericRuntime()
        code = "result = 2 + 2"
        # Should not raise
        runtime.exec_code(code)

    def test_blocks_os_import(self):
        """Should block os module import"""
        runtime = GenericRuntime()
        code = "import os"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_subprocess_import(self):
        """Should block subprocess module import"""
        runtime = GenericRuntime()
        code = "import subprocess"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_eval_function(self):
        """Should block eval() calls"""
        runtime = GenericRuntime()
        code = "eval('2 + 2')"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_exec_function(self):
        """Should block exec() calls in user code"""
        runtime = GenericRuntime()
        code = "exec('print(1)')"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_file_open(self):
        """Should block file operations"""
        runtime = GenericRuntime()
        code = "f = open('/etc/passwd')"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_system_calls(self):
        """Should block os.system calls"""
        runtime = GenericRuntime()
        code = "os.system('ls')"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_socket_import(self):
        """Should block socket import"""
        runtime = GenericRuntime()
        code = "import socket"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_pickle_import(self):
        """Should block pickle import (code execution risk)"""
        runtime = GenericRuntime()
        code = "import pickle"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_getattr_setattr(self):
        """Should block getattr/setattr for security bypass"""
        runtime = GenericRuntime()
        code = "getattr(obj, 'attr')"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_blocks_dunder_access(self):
        """Should block access to dunder attributes"""
        runtime = GenericRuntime()
        code = "x.__class__.__bases__"
        with pytest.raises(UnsafeCodeError):
            runtime.exec_code(code)

    def test_allows_math_operations(self):
        """Should allow safe math operations"""
        runtime = GenericRuntime()
        code = """
import math
result = math.sqrt(16)
answer = result
"""
        runtime.exec_code(code)
        assert runtime.answer == 4.0

    def test_allows_numpy_operations(self):
        """Should allow numpy operations"""
        runtime = GenericRuntime()
        code = """
import numpy as np
arr = np.array([1, 2, 3])
answer = arr.sum()
"""
        runtime.exec_code(code)
        assert runtime.answer == 6


class TestPythonExecutorAsync:
    """Tests for async execution"""

    @pytest.mark.asyncio
    async def test_async_execution_basic(self):
        """Test basic async execution"""
        code = "2 + 2"
        result, report = await PythonExecutor.execute(code)
        assert result == 4
        assert report == "Done"

    @pytest.mark.asyncio
    async def test_async_execution_timeout(self):
        """Test that timeout works"""
        code = """
import time
time.sleep(100)
"""
        # This should timeout
        result, report = await PythonExecutor.execute(
            code,
            timeout_length=1
        )
        # Should fail and return error in report
        assert result == ''
        assert report != "Done"

    @pytest.mark.asyncio
    async def test_async_execution_with_answer_symbol(self):
        """Test execution with answer symbol"""
        code = """
x = 10
y = 20
answer = x + y
"""
        result, report = await PythonExecutor.execute(
            code,
            answer_symbol="answer"
        )
        assert result == 30
        assert report == "Done"


class TestPythonExecutorEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.asyncio
    async def test_handles_list_code_input(self):
        """Should handle code as list of lines"""
        code = ["x = 5", "y = 10", "x + y"]
        result, report = await PythonExecutor.execute(code)
        assert result == 15
        assert report == "Done"

    @pytest.mark.asyncio
    async def test_handles_multiline_string(self):
        """Should handle multiline string code"""
        code = """
x = 5
y = 10
z = x + y
z * 2
"""
        result, report = await PythonExecutor.execute(code)
        assert result == 30
        assert report == "Done"
