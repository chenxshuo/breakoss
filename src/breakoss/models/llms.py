"""LLM model classes and utilities for text generation."""

from dataclasses import dataclass
from pyexpat import model
from pyexpat.errors import messages
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from loguru import logger
import os
from .openrouter import LLMOpenRouter

class InferenceConfig(BaseModel):
    """Configuration for LLM text generation inference."""

    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    reasoning_effort: Literal["low", "medium", "high"] = "medium"


class LLMHF:
    """LLM wrapper for Hugging Face Transformers models."""

    def __init__(
        self, *, model_name: str, tokenizer: AutoTokenizer, cuda_device: str = "auto"
    ):
        """Initialize the Hugging Face LLM wrapper.

        Args:
            model_name: Name of the model to load from Hugging Face.
            tokenizer: Pre-initialized tokenizer for the model.
        """
        logger.info(
            f"Loading model {model_name} on device {cuda_device} based on Hugging Face Transformers"
        )
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=cuda_device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"


    def chat(self, *, prompt: str, inference_config: InferenceConfig, apply_chat_template: bool = True) -> str:
        """Generate a chat response using the loaded model.

        Args:
            prompt: Input text prompt for the model.
            inference_config: Configuration parameters for text generation.

        Returns:
            Generated text response from the model.
        """
        messages = [{"role": "user", "content": prompt}]
        if apply_chat_template:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=inference_config.reasoning_effort
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=inference_config.max_new_tokens,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            repetition_penalty=inference_config.repetition_penalty,
            do_sample=inference_config.do_sample,
        )
        logger.info(f"Generated response for prompt: {prompt}")
        logger.info(f"*" * 20)
        logger.info(
            f"Full input text: {self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}"
        )
        logger.info(f"*" * 20)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )

    def chat_batch(self, *, prompts: list[str], inference_config: InferenceConfig, system_prompt: str = None, apply_chat_template: bool=True) -> list[str]:
        """Generate chat responses for a batch of prompts using the loaded model.

        Args:
            prompts: List of input text prompts for the model.
            inference_config: Configuration parameters for text generation.

        Returns:
            List of generated text responses from the model.
        """
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        if apply_chat_template:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True, truncation=True
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=inference_config.max_new_tokens,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            repetition_penalty=inference_config.repetition_penalty,
            do_sample=inference_config.do_sample,
        )
        logger.info(f"Generated responses for {len(prompts)} prompts.")
        responses = []
        for i, prompt in enumerate(prompts):
            response = self.tokenizer.decode(
                outputs[i][inputs["input_ids"].shape[1] :], skip_special_tokens=False
            )
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
            logger.info(f"*" * 20)
            responses.append(response)
        return responses


def load_llm(
    *,
    model_name: str,
    cuda_device: str = "auto",
    provider: Literal["hf", "vllm", "openrouter"] = "hf",
):
    """Load a language model with the specified provider.

    Args:
        model_name: Name of the model to load.
        cuda_device: CUDA device identifier (currently unused).
        provider: Model provider, either 'hf' for Hugging Face or 'vllm'.

    Returns:
        Initialized LLM model instance.

    Raises:
        NotImplementedError: If VLLM provider is requested.
        ValueError: If unknown provider is specified.
    """
    if provider == "hf":
        return LLMHF(
            model_name=model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), cuda_device=cuda_device
        )
    elif provider == "vllm":
        raise NotImplementedError("VLLM provider is not implemented yet.")
    elif provider.lower() == "openrouter":
        return LLMOpenRouter(
            model_name=model_name
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


class Translator():
    ...