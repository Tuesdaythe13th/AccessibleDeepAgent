"""
Unit tests for configuration validation
"""

import pytest
from src.utils.config_validator import (
    is_placeholder_key,
    validate_required_keys,
    ConfigurationError,
    get_environment_override,
    validate_and_warn
)


class TestPlaceholderDetection:
    """Tests for placeholder key detection"""

    def test_is_placeholder_key_with_your_prefix(self):
        assert is_placeholder_key("YOUR_API_KEY")
        assert is_placeholder_key("YOUR_GOOGLE_API_KEY")

    def test_is_placeholder_key_with_replace_prefix(self):
        assert is_placeholder_key("REPLACE_WITH_KEY")

    def test_is_placeholder_key_with_ellipsis(self):
        assert is_placeholder_key("...")
        assert is_placeholder_key("sk-proj-...")

    def test_is_placeholder_key_with_xxx(self):
        assert is_placeholder_key("xxx")
        assert is_placeholder_key("xxxyyyzz")

    def test_is_placeholder_key_valid_key(self):
        assert not is_placeholder_key("sk-1234567890abcdef")
        assert not is_placeholder_key("AIzaSyDFakeKey123456")

    def test_is_placeholder_key_empty(self):
        assert is_placeholder_key("")
        assert is_placeholder_key(None)


class TestRequiredKeysValidation:
    """Tests for required keys validation"""

    def test_validate_required_keys_all_present(self):
        config = {
            "google_serper_api": "valid-api-key-123",
            "toolbench_api": "another-valid-key-456"
        }
        result = validate_required_keys(config, "hotpotqa", strict=False)
        assert result["missing"] == []
        assert result["placeholder"] == []

    def test_validate_required_keys_missing(self):
        config = {
            "google_serper_api": "valid-api-key-123"
        }
        result = validate_required_keys(config, "hotpotqa", strict=False)
        assert "toolbench_api" in result["missing"]

    def test_validate_required_keys_placeholder(self):
        config = {
            "google_serper_api": "YOUR_API_KEY",
            "toolbench_api": "valid-key-123"
        }
        result = validate_required_keys(config, "hotpotqa", strict=False)
        assert "google_serper_api" in result["placeholder"]

    def test_validate_required_keys_strict_raises(self):
        config = {
            "google_serper_api": "YOUR_API_KEY"
        }
        with pytest.raises(ConfigurationError):
            validate_required_keys(config, "hotpotqa", strict=True)

    def test_validate_required_keys_default_dataset(self):
        config = {
            "google_serper_api": "valid-key-123"
        }
        result = validate_required_keys(config, "unknown_dataset", strict=False)
        # Should fall back to default requirements
        assert result["missing"] == []


class TestValidateAndWarn:
    """Integration tests for validate_and_warn"""

    def test_validate_and_warn_returns_updated_config(self):
        config = {
            "google_serper_api": "valid-key-123",
            "toolbench_api": "valid-key-456"
        }
        updated = validate_and_warn(config, "hotpotqa")
        assert "google_serper_api" in updated
        assert "toolbench_api" in updated

    def test_validate_and_warn_with_placeholders(self):
        config = {
            "google_serper_api": "YOUR_API_KEY",
            "toolbench_api": "valid-key-456"
        }
        # Should not raise, just warn
        updated = validate_and_warn(config, "hotpotqa")
        assert updated is not None
