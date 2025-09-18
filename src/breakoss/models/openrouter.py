from dataclasses import dataclass
from pyexpat import model
from pyexpat.errors import messages
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from loguru import logger
import os
import json
import time

import requests

class InferenceConfig(BaseModel):
    """Configuration for LLM text generation inference."""

    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True


class LLMOpenRouter:
    def __init__(self, model_name):
        logger.info(f"Initializing LLM Router {model_name}")
        self.model_name = model_name
        assert self.model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-20b:free"]
        if self.model_name == "openai/gpt-oss-20b:free":
            self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY", "")
        if self.OPEN_ROUTER_KEY == "":
            raise ValueError(f"OPEN_ROUTER_KEY must be defined")

        self.Template =  """<|start|>user<|message|>{prompt}<|end|><|start|>assistant"""

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, *, prompt: str, inference_config: InferenceConfig, **kwargs) -> str:
        num_input_tokens = len(self.tokenizer(prompt).input_ids)
        self.total_input_tokens += num_input_tokens
        response_chat = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.OPEN_ROUTER_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": inference_config.max_new_tokens,
                "temperature": inference_config.temperature,
                "top_p": inference_config.top_p,
                "n": 1,
                "presence_penalty": inference_config.repetition_penalty,
                "provider": {
                    "order": [
                        # "DeepInfra",
                        "AtlasCloud"
                    ]
                }

            })
        )
        try:
            result =  response_chat.json()['choices'][0]['message']['content']
            self.total_output_tokens += len(self.tokenizer(result).input_ids)
        except (KeyError, json.decoder.JSONDecodeError) as e :
            result = f"Error with {e} and full response is: {response_chat.json()}"
        return result

    def chat_batch(self, *, prompts: list[str], inference_config: InferenceConfig, system_prompt: str = None,
                   apply_chat_template: bool = True) -> list[str]:
        responses = []
        for prompt in prompts:
            response = self.chat(
                prompt=prompt,
                inference_config=inference_config,
            )
            responses.append(response)
            time.sleep(2)

        for i, response in enumerate(responses):
            logger.info(f"Input prompt {i}: {prompts[i]}")
            logger.info(f"Generated response: {response}.")
        return responses



