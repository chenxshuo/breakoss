"""Tests for LLM models module."""

import unittest
from unittest.mock import patch, MagicMock

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

    def test_inference_config_accepts_various_values(self):
        """Test InferenceConfig accepts various input values (no validation constraints)."""
        # The current implementation doesn't have validation constraints
        # Test that various values are accepted
        config = InferenceConfig(max_new_tokens=-1)  # Negative values accepted
        self.assertEqual(config.max_new_tokens, -1)
        
        config = InferenceConfig(temperature=-0.1)  # Negative temperature accepted
        self.assertEqual(config.temperature, -0.1)
        
        config = InferenceConfig(top_p=1.5)  # Values > 1.0 accepted
        self.assertEqual(config.top_p, 1.5)
        
        # Test edge case values
        config = InferenceConfig(
            max_new_tokens=0,
            temperature=0.0,
            top_p=0.0,
            repetition_penalty=0.0,
            do_sample=False
        )
        self.assertEqual(config.max_new_tokens, 0)
        self.assertEqual(config.temperature, 0.0)


class TestLLMHF(unittest.TestCase):
    """Test LLMHF class."""

    @patch('breakoss.models.llms.AutoModelForCausalLM')
    @patch('breakoss.models.llms.AutoTokenizer')
    def test_llmhf_initialization(self, mock_tokenizer, mock_model):
        """Test LLMHF initialization."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.device = "cuda:0"
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        llm = LLMHF(model_name="test-model", tokenizer=mock_tokenizer_instance)
        
        self.assertEqual(llm.model_name, "test-model")
        mock_model.from_pretrained.assert_called_once_with(
            "test-model", torch_dtype="auto", device_map="auto"
        )
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")

    @patch('breakoss.models.llms.AutoModelForCausalLM')
    @patch('breakoss.models.llms.AutoTokenizer')
    def test_llmhf_initialization_with_cuda_device(self, mock_tokenizer, mock_model):
        """Test LLMHF initialization with specific CUDA device."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.device = "cuda:1"
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        llm = LLMHF(model_name="test-model", tokenizer=mock_tokenizer_instance, cuda_device="cuda:1")
        
        self.assertEqual(llm.model_name, "test-model")
        mock_model.from_pretrained.assert_called_once_with(
            "test-model", torch_dtype="auto", device_map="cuda:1"
        )

    @patch('breakoss.models.llms.AutoModelForCausalLM')
    @patch('breakoss.models.llms.AutoTokenizer')
    def test_llmhf_chat(self, mock_tokenizer, mock_model):
        """Test LLMHF chat method."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.device = "cuda:0"
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the chat template and generation
        mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_inputs_tensor = MagicMock()
        mock_inputs_tensor.to.return_value = mock_inputs
        mock_tokenizer_instance.apply_chat_template.return_value = mock_inputs_tensor
        
        mock_outputs = [MagicMock()]
        mock_model_instance.generate.return_value = mock_outputs
        mock_tokenizer_instance.decode.return_value = "Generated response"
        
        llm = LLMHF(model_name="test-model", tokenizer=mock_tokenizer_instance)
        config = InferenceConfig(max_new_tokens=100)
        
        response = llm.chat(prompt="Test prompt", inference_config=config)
        
        self.assertEqual(response, "Generated response")
        mock_tokenizer_instance.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Test prompt"}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        mock_model_instance.generate.assert_called_once()
        
        # Check that generate was called with correct parameters
        call_args = mock_model_instance.generate.call_args
        self.assertEqual(call_args[1]["max_new_tokens"], 100)
        self.assertEqual(call_args[1]["temperature"], 0.7)  # default
        self.assertEqual(call_args[1]["top_p"], 0.9)  # default


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
        
        result = load_llm(model_name="test-model", cuda_device="cuda:0", provider="hf")
        
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_llmhf.assert_called_once_with(
            model_name="test-model", 
            tokenizer=mock_tokenizer_instance
        )
        self.assertEqual(result, mock_llm_instance)

    def test_load_llm_hf_provider_default_params(self):
        """Test load_llm with Hugging Face provider using default parameters."""
        with patch('breakoss.models.llms.LLMHF') as mock_llmhf, \
             patch('breakoss.models.llms.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_llm_instance = MagicMock()
            mock_llmhf.return_value = mock_llm_instance
            
            result = load_llm(model_name="test-model")
            
            mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
            mock_llmhf.assert_called_once_with(
                model_name="test-model", 
                tokenizer=mock_tokenizer_instance
            )
            self.assertEqual(result, mock_llm_instance)

    def test_load_llm_vllm_provider(self):
        """Test load_llm with VLLM provider raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as cm:
            load_llm(model_name="test-model", cuda_device="cuda:0", provider="vllm")
        self.assertIn("VLLM provider is not implemented yet", str(cm.exception))

    def test_load_llm_unknown_provider(self):
        """Test load_llm with unknown provider raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            load_llm(model_name="test-model", cuda_device="cuda:0", provider="unknown")  # type: ignore
        self.assertIn("Unknown provider: unknown", str(cm.exception))


if __name__ == '__main__':
    unittest.main()