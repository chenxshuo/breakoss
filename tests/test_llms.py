"""Tests for LLM models module."""

import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from breakoss.models.llms import InferenceConfig, LLMHF, load_llm


class TestInferenceConfig(unittest.TestCase):
    """Test InferenceConfig class."""

    def test_inference_config_defaults(self):
        """Test InferenceConfig with default values."""
        config = InferenceConfig()
        self.assertEqual(config.max_new_tokens, 4096)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertTrue(config.do_sample)

    def test_inference_config_custom_values(self):
        """Test InferenceConfig with custom values."""
        config = InferenceConfig(
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.8,
            repetition_penalty=1.1,
            do_sample=False
        )
        self.assertEqual(config.max_new_tokens, 512)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.repetition_penalty, 1.1)
        self.assertFalse(config.do_sample)


class TestLLMHF(unittest.TestCase):
    """Test LLMHF class."""

    @patch('breakoss.models.llms.AutoModelForCausalLM')
    @patch('breakoss.models.llms.AutoTokenizer')
    def test_llmhf_initialization(self, mock_tokenizer, mock_model):
        """Test LLMHF initialization."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        llm = LLMHF("test-model", mock_tokenizer_instance)
        
        self.assertEqual(llm.model_name, "test-model")
        mock_model.from_pretrained.assert_called_once_with(
            "test-model", torch_dtype="auto", device_map="auto"
        )
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")

    @patch('breakoss.models.llms.AutoModelForCausalLM')
    @patch('breakoss.models.llms.AutoTokenizer')
    def test_llmhf_chat(self, mock_tokenizer, mock_model):
        """Test LLMHF chat method."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the chat template and generation
        mock_inputs = {"input_ids": "mocked_tensor"}
        mock_tokenizer_instance.apply_chat_template.return_value.to.return_value = mock_inputs
        mock_model_instance.generate.return_value = ["generated_output"]
        mock_tokenizer_instance.decode.return_value = "Generated response"
        
        llm = LLMHF("test-model", mock_tokenizer_instance)
        config = InferenceConfig(max_new_tokens=100)
        
        response = llm.chat("Test prompt", config)
        
        self.assertEqual(response, "Generated response")
        mock_tokenizer_instance.apply_chat_template.assert_called_once()
        mock_model_instance.generate.assert_called_once()


class TestLoadLLM(unittest.TestCase):
    """Test load_llm function."""

    @patch('breakoss.models.llms.LLMHF')
    @patch('breakoss.models.llms.AutoTokenizer')
    def test_load_llm_hf_provider(self, mock_tokenizer, mock_llmhf):
        """Test load_llm with Hugging Face provider."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_llm_instance = MagicMock()
        mock_llmhf.return_value = mock_llm_instance
        
        result = load_llm("test-model", "cuda:0", "hf")
        
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_llmhf.assert_called_once_with(
            model_name="test-model", 
            tokenizer=mock_tokenizer_instance
        )
        self.assertEqual(result, mock_llm_instance)

    def test_load_llm_vllm_provider(self):
        """Test load_llm with VLLM provider raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            load_llm("test-model", "cuda:0", "vllm")

    def test_load_llm_unknown_provider(self):
        """Test load_llm with unknown provider raises ValueError."""
        with self.assertRaises(ValueError):
            load_llm("test-model", "cuda:0", "unknown")


if __name__ == '__main__':
    unittest.main()