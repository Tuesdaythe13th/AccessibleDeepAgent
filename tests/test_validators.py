"""
Unit tests for input validators
"""

import pytest
import math
from src.adk.utils.validators import (
    validate_signal_value,
    validate_probability,
    validate_content_length,
    validate_file_path,
    validate_api_key,
    validate_timeout,
    ValidationException
)


class TestSignalValueValidation:
    """Tests for signal value validation"""

    def test_valid_integer(self):
        result = validate_signal_value(42, "test_signal")
        assert result == 42.0
        assert isinstance(result, float)

    def test_valid_float(self):
        result = validate_signal_value(3.14, "test_signal")
        assert result == 3.14

    def test_none_raises(self):
        with pytest.raises(ValidationException):
            validate_signal_value(None, "test_signal")

    def test_string_number_converts(self):
        result = validate_signal_value("42.5", "test_signal")
        assert result == 42.5

    def test_nan_raises(self):
        with pytest.raises(ValidationException):
            validate_signal_value(math.nan, "test_signal")

    def test_infinity_raises(self):
        with pytest.raises(ValidationException):
            validate_signal_value(math.inf, "test_signal")

    def test_invalid_string_raises(self):
        with pytest.raises(ValidationException):
            validate_signal_value("not_a_number", "test_signal")


class TestProbabilityValidation:
    """Tests for probability validation"""

    def test_valid_probability_zero(self):
        result = validate_probability(0.0, "test_prob")
        assert result == 0.0

    def test_valid_probability_one(self):
        result = validate_probability(1.0, "test_prob")
        assert result == 1.0

    def test_valid_probability_middle(self):
        result = validate_probability(0.5, "test_prob")
        assert result == 0.5

    def test_negative_raises(self):
        with pytest.raises(ValidationException):
            validate_probability(-0.1, "test_prob")

    def test_greater_than_one_raises(self):
        with pytest.raises(ValidationException):
            validate_probability(1.1, "test_prob")

    def test_non_number_raises(self):
        with pytest.raises(ValidationException):
            validate_probability("0.5", "test_prob")


class TestContentLengthValidation:
    """Tests for content length validation"""

    def test_valid_content(self):
        content = "This is a test content"
        result = validate_content_length(content)
        assert result == content

    def test_empty_content(self):
        result = validate_content_length("")
        assert result == ""

    def test_max_length_default(self):
        content = "x" * 100000
        result = validate_content_length(content)
        assert len(result) == 100000

    def test_max_length_exceeded_raises(self):
        content = "x" * 100001
        with pytest.raises(ValidationException):
            validate_content_length(content)

    def test_custom_max_length(self):
        content = "x" * 101
        with pytest.raises(ValidationException):
            validate_content_length(content, max_length=100)

    def test_non_string_raises(self):
        with pytest.raises(ValidationException):
            validate_content_length(123)


class TestFilePathValidation:
    """Tests for file path validation"""

    def test_valid_relative_path(self):
        result = validate_file_path("test/file.txt")
        assert result == "test/file.txt"

    def test_path_traversal_raises(self):
        with pytest.raises(ValidationException):
            validate_file_path("../etc/passwd")

    def test_absolute_path_raises_by_default(self):
        with pytest.raises(ValidationException):
            validate_file_path("/etc/passwd")

    def test_absolute_path_allowed_with_flag(self):
        result = validate_file_path("/home/user/file.txt", allow_absolute=True)
        assert result == "/home/user/file.txt"

    def test_non_string_raises(self):
        with pytest.raises(ValidationException):
            validate_file_path(123)


class TestAPIKeyValidation:
    """Tests for API key validation"""

    def test_valid_api_key(self):
        key = "sk-1234567890abcdef"
        result = validate_api_key(key, "test_key")
        assert result == key

    def test_placeholder_raises(self):
        with pytest.raises(ValidationException):
            validate_api_key("YOUR_API_KEY", "test_key")

    def test_empty_raises(self):
        with pytest.raises(ValidationException):
            validate_api_key("", "test_key")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationException):
            validate_api_key("   ", "test_key")

    def test_non_string_raises(self):
        with pytest.raises(ValidationException):
            validate_api_key(123, "test_key")


class TestTimeoutValidation:
    """Tests for timeout validation"""

    def test_valid_timeout_int(self):
        result = validate_timeout(10)
        assert result == 10.0
        assert isinstance(result, float)

    def test_valid_timeout_float(self):
        result = validate_timeout(5.5)
        assert result == 5.5

    def test_minimum_timeout(self):
        result = validate_timeout(0.1)
        assert result == 0.1

    def test_below_minimum_raises(self):
        with pytest.raises(ValidationException):
            validate_timeout(0.05)

    def test_maximum_timeout(self):
        result = validate_timeout(300.0)
        assert result == 300.0

    def test_above_maximum_raises(self):
        with pytest.raises(ValidationException):
            validate_timeout(301.0)

    def test_non_number_raises(self):
        with pytest.raises(ValidationException):
            validate_timeout("10")
