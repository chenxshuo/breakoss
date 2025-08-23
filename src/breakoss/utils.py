import os
import random
from breakoss.models import LLMHF, InferenceConfig
from breakoss.methods import BaseMethod
from breakoss.evaluators import Evaluator
from datetime import datetime

from pathlib import Path
import numpy as np
import torch 


def get_exp_dir(log_base: Path | str = None, model: LLMHF = None, method: BaseMethod = None, evaluator: Evaluator = None, seed: int = 42, dataset_name: str = None) -> str:
    if log_base is None:
        log_base = os.path.join(
                os.path.dirname(__file__), "../../breakoss_logs"
            )
    datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        log_base,
        model.model_name.replace("/", "_"),
        dataset_name if dataset_name else "user_custom_data",
        method.name() if method else "no_method",
        evaluator.name(),
        f"seed_{seed}",
        datetime_stamp
    )
    return log_dir

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False