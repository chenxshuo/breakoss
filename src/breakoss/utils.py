import os
import random
import gc
from breakoss.models import LLMHF, InferenceConfig
from breakoss.methods import BaseMethod
from breakoss.evaluators import Evaluator
from datetime import datetime

from pathlib import Path
import numpy as np
import torch


def get_exp_dir(
    log_base: Path | str = None,
    model: LLMHF = None,
    method: BaseMethod = None,
    evaluator: Evaluator = None,
    seed: int = 42,
    dataset_name: str = None,
    starting_index: int = 0,
    end_index: int = -1,
) -> str:
    if log_base is None:
        log_base = os.path.join(os.path.dirname(__file__), "../../breakoss_logs")
    datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        log_base,
        model.model_name.replace("/", "_"),
        (
            f"{dataset_name}_[{starting_index}_{end_index}]"
            if dataset_name
            else "user_custom_data"
        ),
        method.name() if method else "direct",
        evaluator.name() if evaluator else "pure_inference_no_eval",
        f"seed_{seed}",
        datetime_stamp,
    )
    return log_dir


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def free_gpu_memory(tar_object=None):
    if tar_object is not None:
        # Handle evaluator objects
        if hasattr(tar_object, '__class__') and 'Evaluator' in tar_object.__class__.__name__:
            # LlamaGuardEvaluator
            if hasattr(tar_object, 'llama_model'):
                if hasattr(tar_object.llama_model, 'model') and hasattr(tar_object.llama_model.model, 'cpu'):
                    tar_object.llama_model.model.cpu()
                del tar_object.llama_model
            
            # StrongREJECTEvaluator - handle PeftModel and base model
            if hasattr(tar_object, 'base_model'):
                if hasattr(tar_object.base_model, 'cpu'):
                    tar_object.base_model.cpu()
                del tar_object.base_model
            
            # Handle any model attribute (covers StrongREJECT, HarmBench, LlamaGuard)
            if hasattr(tar_object, 'model'):
                if hasattr(tar_object.model, 'cpu'):
                    tar_object.model.cpu()
                del tar_object.model
            
            # Handle tokenizers
            if hasattr(tar_object, 'tokenizer'):
                del tar_object.tokenizer
        
        # Handle LLMHF model objects specifically
        elif hasattr(tar_object, 'model') and hasattr(tar_object, 'model_name'):
            if hasattr(tar_object.model, 'cpu'):
                tar_object.model.cpu()
            del tar_object.model
            if hasattr(tar_object, 'tokenizer'):
                del tar_object.tokenizer
        
        # Handle general PyTorch models
        elif hasattr(tar_object, 'cpu'):
            tar_object.cpu()
        
        del tar_object
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
