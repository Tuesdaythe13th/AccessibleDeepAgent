# AccessibleDeepAgent Test Suite

This directory contains unit and integration tests for the AccessibleDeepAgent project.

## Running Tests

### Prerequisites

Install pytest and dependencies:

```bash
pip install pytest pytest-asyncio
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_config_validator.py
pytest tests/test_python_executor.py
pytest tests/test_validators.py
```

### Run with Coverage

```bash
pip install pytest-cov
pytest --cov=src --cov-report=html
```

### Run Only Unit Tests

```bash
pytest -m unit
```

### Run Only Async Tests

```bash
pytest -m asyncio
```

## Test Structure

- `test_config_validator.py` - Tests for API key and configuration validation
- `test_python_executor.py` - Tests for Python code execution security
- `test_validators.py` - Tests for input validation utilities
- `conftest.py` - Shared fixtures and configuration

## Writing New Tests

Follow these conventions:

1. Test files should start with `test_`
2. Test classes should start with `Test`
3. Test functions should start with `test_`
4. Use descriptive test names that explain what is being tested
5. Use async test functions with `@pytest.mark.asyncio` for async code
6. Group related tests in classes

Example:

```python
import pytest

class TestMyFeature:
    def test_basic_functionality(self):
        assert my_function() == expected_value

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        result = await my_async_function()
        assert result == expected_value
```

## Test Coverage Goals

Target coverage: 70%+

Priority areas:
- Security-critical components (Python executor, validators)
- Configuration validation
- Core agent logic
- API boundaries
