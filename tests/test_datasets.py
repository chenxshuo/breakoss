"""Tests for datasets module."""

import unittest
from unittest.mock import patch
import pandas as pd

from breakoss.datasets.datasets import DataSample, StrongReject, HarmfulBehaviors


class TestDataSample(unittest.TestCase):
    """Test DataSample class."""

    def test_data_sample_creation(self):
        """Test creating a DataSample."""
        sample = DataSample(
            harmful_prompt="Test prompt",
            target="Test target",
            metadata={"category": "test"}
        )
        self.assertEqual(sample.harmful_prompt, "Test prompt")
        self.assertEqual(sample.target, "Test target")
        self.assertEqual(sample.metadata["category"], "test")


class TestStrongReject(unittest.TestCase):
    """Test StrongReject dataset class."""

    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'forbidden_prompt': ['prompt1', 'prompt2'],
            'category': ['cat1', 'cat2'],
            'source': ['src1', 'src2']
        })

    @patch('pandas.read_csv')
    def test_strong_reject_initialization_custom_path(self, mock_read_csv):
        """Test StrongReject initialization with custom path."""
        mock_read_csv.return_value = self.test_data
        
        dataset = StrongReject("test.csv")
        mock_read_csv.assert_called_once_with("test.csv")

    def test_strong_reject_initialization_default_path(self):
        """Test StrongReject initialization with default path."""
        dataset = StrongReject()
        samples = list(dataset)
        self.assertEqual(len(samples), 313)
        self.assertIsInstance(samples[0], DataSample)

    @patch('pandas.read_csv')
    def test_strong_reject_iteration(self, mock_read_csv):
        """Test StrongReject iteration returns DataSample objects."""
        mock_read_csv.return_value = self.test_data
        
        dataset = StrongReject()
        samples = list(dataset)
        
        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], DataSample)
        self.assertEqual(samples[0].harmful_prompt, 'prompt1')
        self.assertEqual(samples[0].target, '')


class TestHarmfulBehaviors(unittest.TestCase):
    """Test HarmfulBehaviors dataset class."""

    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'goal': ['harmful goal 1', 'harmful goal 2'],
            'target': ['target 1', 'target 2']
        })

    @patch('pandas.read_csv')
    def test_harmful_behaviors_initialization(self, mock_read_csv):
        """Test HarmfulBehaviors initialization."""
        mock_read_csv.return_value = self.test_data
        
        dataset = HarmfulBehaviors("test.csv")
        mock_read_csv.assert_called_once_with("test.csv")

    @patch('pandas.read_csv')
    def test_harmful_behaviors_iteration(self, mock_read_csv):
        """Test HarmfulBehaviors iteration returns DataSample objects."""
        mock_read_csv.return_value = self.test_data
        
        dataset = HarmfulBehaviors()
        samples = list(dataset)
        
        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], DataSample)
        self.assertEqual(samples[0].harmful_prompt, 'harmful goal 1')
        self.assertEqual(samples[0].target, 'target 1')
    
    def test_harmful_behaviors_default_csv(self):
        """Test HarmfulBehaviors initialization with default path."""
        dataset = HarmfulBehaviors()
        samples = list(dataset)
        self.assertEqual(len(samples), 520)
        self.assertIsInstance(samples[0], DataSample)

if __name__ == '__main__':
    unittest.main()