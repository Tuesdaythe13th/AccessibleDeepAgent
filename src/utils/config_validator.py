"""
Configuration and API Key Validation Utilities

This module provides validation functions for configuration files and API keys
to ensure the system has required credentials before execution.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Dataset-specific required API keys mapping
DATASET_KEY_MAP = {
    "hotpotqa": ["google_serper_api", "toolbench_api"],
    "toolbench": ["toolbench_api"],
    "scienceqa": ["google_serper_api"],
    "default": ["google_serper_api"]
}


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete"""
    pass


def is_placeholder_key(key_value: str) -> bool:
    """
    Check if an API key is a placeholder value

    Args:
        key_value: The API key value to check

    Returns:
        True if the key appears to be a placeholder
    """
    if not key_value:
        return True

    placeholder_patterns = [
        "YOUR_",
        "REPLACE_",
        "ENTER_",
        "INSERT_",
        "ADD_YOUR_",
        "sk-proj-",  # OpenAI placeholder pattern
        "xxx",
        "..."
    ]

    key_upper = key_value.upper()
    return any(pattern.upper() in key_upper for pattern in placeholder_patterns)


def validate_required_keys(
    config: Dict[str, Any],
    dataset_name: str = "default",
    strict: bool = False
) -> Dict[str, List[str]]:
    """
    Validate that required API keys are present and not placeholders

    Args:
        config: Configuration dictionary containing API keys
        dataset_name: Name of dataset to determine required keys
        strict: If True, raise exception on missing keys. If False, log warnings

    Returns:
        Dictionary with 'missing' and 'placeholder' key lists

    Raises:
        ConfigurationError: If strict=True and validation fails
    """
    required_keys = DATASET_KEY_MAP.get(dataset_name, DATASET_KEY_MAP["default"])

    missing_keys = []
    placeholder_keys = []

    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
        elif is_placeholder_key(config[key]):
            placeholder_keys.append(key)

    result = {
        "missing": missing_keys,
        "placeholder": placeholder_keys
    }

    # Log issues
    if missing_keys:
        logger.error(f"Missing required API keys: {missing_keys}")
    if placeholder_keys:
        logger.warning(f"API keys appear to be placeholders: {placeholder_keys}")

    # Raise exception if strict mode
    if strict and (missing_keys or placeholder_keys):
        error_msg = []
        if missing_keys:
            error_msg.append(f"Missing keys: {missing_keys}")
        if placeholder_keys:
            error_msg.append(f"Placeholder keys: {placeholder_keys}")
        raise ConfigurationError("; ".join(error_msg))

    return result


def validate_config_safety(config: Dict[str, Any]) -> List[str]:
    """
    Check configuration for potential security issues

    Args:
        config: Configuration dictionary

    Returns:
        List of warning messages about security concerns
    """
    warnings = []

    # Check for overly permissive settings
    if config.get("allow_unsafe_code", False):
        warnings.append("Configuration allows unsafe code execution")

    # Check for debug mode in production
    if config.get("debug_mode", False):
        warnings.append("Debug mode is enabled - should be disabled in production")

    # Check for exposed secrets in logs
    if config.get("log_api_requests", False):
        warnings.append("API request logging enabled - may expose secrets in logs")

    return warnings


def get_environment_override(key_name: str) -> Optional[str]:
    """
    Check if an API key has an environment variable override

    Args:
        key_name: Name of the API key

    Returns:
        Environment variable value if set, None otherwise
    """
    import os

    # Try common environment variable patterns
    env_patterns = [
        key_name.upper(),
        f"{key_name.upper()}_KEY",
        f"API_{key_name.upper()}",
    ]

    for pattern in env_patterns:
        value = os.getenv(pattern)
        if value and not is_placeholder_key(value):
            return value

    return None


def validate_and_warn(config: Dict[str, Any], dataset_name: str = "default") -> Dict[str, Any]:
    """
    Validate configuration and warn about issues (non-strict mode)

    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset being used

    Returns:
        Updated configuration with environment variable overrides
    """
    logger.info(f"Validating configuration for dataset: {dataset_name}")

    # Check for environment variable overrides
    updated_config = config.copy()
    required_keys = DATASET_KEY_MAP.get(dataset_name, DATASET_KEY_MAP["default"])

    for key in required_keys:
        env_value = get_environment_override(key)
        if env_value:
            logger.info(f"Using environment variable override for {key}")
            updated_config[key] = env_value

    # Validate keys
    validation_result = validate_required_keys(updated_config, dataset_name, strict=False)

    # Check security settings
    security_warnings = validate_config_safety(updated_config)
    for warning in security_warnings:
        logger.warning(f"Security concern: {warning}")

    # Print summary
    if validation_result["missing"] or validation_result["placeholder"]:
        logger.warning("=" * 60)
        logger.warning("CONFIGURATION WARNINGS")
        logger.warning("=" * 60)
        if validation_result["missing"]:
            logger.warning(f"Missing API keys: {', '.join(validation_result['missing'])}")
        if validation_result["placeholder"]:
            logger.warning(f"Placeholder API keys: {', '.join(validation_result['placeholder'])}")
        logger.warning("Some features may not work correctly.")
        logger.warning("Consider setting these keys as environment variables.")
        logger.warning("=" * 60)
    else:
        logger.info("✓ All required API keys are configured")

    return updated_config
