"""A research toolkit providing a bag of techniques to jailbreak OpenAI's open-source models."""

from .api import hello, square
from .go_jailbreak import go_jailbreak, go_evaluate, go_evaluate_on_all_metrics
from .datasets import (
    DataSample,
    StrongReject,
    HarmfulBehaviors,
    BaseDataset,
    HarmBench,
    JBBHarmfulBehaviors,
    CatQA,
)

# being explicit about exports is important!
__all__ = [
    "hello",
    "square",
    "go_jailbreak",
    "go_evaluate",
    "go_evaluate_on_all_metrics",
    "DataSample",
    "StrongReject",
    "HarmfulBehaviors",
    "BaseDataset",
    "HarmBench",
    "JBBHarmfulBehaviors",
    "CatQA",
]
