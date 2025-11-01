"""
BrainGraphNet Test Suite
Unit tests for data processing, models, and training utilities.

Run tests with:
    pytest tests/ -v
    pytest tests/ --cov=src --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "BrainGraphNet Contributors"

# Test configuration
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
