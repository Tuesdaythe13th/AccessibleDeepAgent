"""
Input Validation Utilities

This module provides additional validation utilities beyond Pydantic models
for edge cases, API boundaries, and runtime validation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


class ValidationException(Exception):
    """Raised when validation fails"""
    pass


def validate_signal_value(value: Any, signal_type: str) -> float:
    """
    Validate and sanitize a signal value

    Args:
        value: Raw signal value
        signal_type: Type of signal

    Returns:
        Validated float value

    Raises:
        ValidationException: If value cannot be validated
    """
    if value is None:
        raise ValidationException(f"Signal value cannot be None for {signal_type}")

    # Try to convert to float
    try:
        float_value = float(value)
    except (ValueError, TypeError) as e:
        raise ValidationException(
            f"Could not convert signal value to float for {signal_type}: {value}"
        ) from e

    # Check for NaN and inf
    import math
    if math.isnan(float_value):
        raise ValidationException(f"Signal value is NaN for {signal_type}")
    if math.isinf(float_value):
        raise ValidationException(f"Signal value is infinite for {signal_type}")

    return float_value


def validate_probability(value: float, field_name: str) -> float:
    """
    Validate that a value is a valid probability (0-1)

    Args:
        value: Value to validate
        field_name: Name of field for error messages

    Returns:
        Validated value

    Raises:
        ValidationException: If value is not in [0, 1]
    """
    if not isinstance(value, (int, float)):
        raise ValidationException(
            f"{field_name} must be a number, got {type(value).__name__}"
        )

    if not 0.0 <= value <= 1.0:
        raise ValidationException(
            f"{field_name} must be between 0 and 1, got {value}"
        )

    return float(value)


def validate_content_length(content: str, max_length: int = 100000) -> str:
    """
    Validate content string length

    Args:
        content: Content string
        max_length: Maximum allowed length

    Returns:
        Validated content

    Raises:
        ValidationException: If content exceeds max length
    """
    if not isinstance(content, str):
        raise ValidationException(
            f"Content must be a string, got {type(content).__name__}"
        )

    if len(content) > max_length:
        raise ValidationException(
            f"Content exceeds maximum length of {max_length} characters "
            f"(got {len(content)})"
        )

    return content


def validate_file_path(path: str, allow_absolute: bool = False) -> str:
    """
    Validate file path for security issues

    Args:
        path: File path to validate
        allow_absolute: Whether to allow absolute paths

    Returns:
        Validated path

    Raises:
        ValidationException: If path is unsafe
    """
    if not isinstance(path, str):
        raise ValidationException(
            f"Path must be a string, got {type(path).__name__}"
        )

    import os

    # Check for path traversal attempts
    if ".." in path or path.startswith("/"):
        if not allow_absolute:
            raise ValidationException(
                f"Path contains traversal or absolute path: {path}"
            )

    # Normalize path
    normalized = os.path.normpath(path)

    # Check if normalization changed the path (possible attack)
    if normalized != path and ".." in path:
        raise ValidationException(
            f"Path normalization detected traversal attempt: {path}"
        )

    return normalized


def validate_api_key(key: str, key_name: str) -> str:
    """
    Validate API key format

    Args:
        key: API key value
        key_name: Name of the API key

    Returns:
        Validated key

    Raises:
        ValidationException: If key is invalid
    """
    if not isinstance(key, str):
        raise ValidationException(
            f"{key_name} must be a string, got {type(key).__name__}"
        )

    if not key or key.strip() == "":
        raise ValidationException(f"{key_name} cannot be empty")

    # Check for placeholder patterns
    placeholder_indicators = ["YOUR_", "REPLACE_", "ENTER_", "xxx", "..."]
    if any(indicator in key.upper() for indicator in placeholder_indicators):
        raise ValidationException(
            f"{key_name} appears to be a placeholder value: {key}"
        )

    # Minimum length check (most API keys are at least 20 chars)
    if len(key) < 10:
        logger.warning(
            f"{key_name} seems unusually short (length: {len(key)}). "
            "This may not be a valid API key."
        )

    return key


def validate_dict_schema(
    data: Dict[str, Any],
    required_fields: List[str],
    schema_name: str
) -> Dict[str, Any]:
    """
    Validate that a dictionary contains required fields

    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        schema_name: Name of schema for error messages

    Returns:
        Validated dictionary

    Raises:
        ValidationException: If required fields are missing
    """
    if not isinstance(data, dict):
        raise ValidationException(
            f"{schema_name} must be a dictionary, got {type(data).__name__}"
        )

    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        raise ValidationException(
            f"{schema_name} missing required fields: {missing_fields}"
        )

    return data


def safe_parse_model(model_class, data: Dict[str, Any], context: str = ""):
    """
    Safely parse data into a Pydantic model with better error messages

    Args:
        model_class: Pydantic model class
        data: Data to parse
        context: Context string for error messages

    Returns:
        Parsed model instance

    Raises:
        ValidationException: If parsing fails
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        error_msg = f"Validation failed for {model_class.__name__}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {e}"
        logger.error(error_msg)
        raise ValidationException(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error parsing {model_class.__name__}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {e}"
        logger.error(error_msg)
        raise ValidationException(error_msg) from e


def validate_timeout(timeout: Union[int, float], min_timeout: float = 0.1, max_timeout: float = 300.0) -> float:
    """
    Validate timeout value

    Args:
        timeout: Timeout value in seconds
        min_timeout: Minimum allowed timeout
        max_timeout: Maximum allowed timeout

    Returns:
        Validated timeout

    Raises:
        ValidationException: If timeout is invalid
    """
    if not isinstance(timeout, (int, float)):
        raise ValidationException(
            f"Timeout must be a number, got {type(timeout).__name__}"
        )

    if timeout < min_timeout:
        raise ValidationException(
            f"Timeout must be at least {min_timeout}s, got {timeout}s"
        )

    if timeout > max_timeout:
        raise ValidationException(
            f"Timeout cannot exceed {max_timeout}s, got {timeout}s"
        )

    return float(timeout)
