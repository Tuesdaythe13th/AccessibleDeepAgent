"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "google_serper_api": "test-api-key-123",
        "toolbench_api": "test-toolbench-key-456",
        "jina_api_key": "test-jina-key-789",
    }


@pytest.fixture
def placeholder_config():
    """Configuration with placeholder keys"""
    return {
        "google_serper_api": "YOUR_API_KEY",
        "toolbench_api": "REPLACE_WITH_KEY",
    }
