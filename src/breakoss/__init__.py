"""A research toolkit providing a bag of techniques to jailbreak OpenAI's open-source models."""

from .api import hello, square
from .go_jailbreak import go_jailbreak, go_evaluate
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
    "DataSample",
    "StrongReject",
    "HarmfulBehaviors",
    "BaseDataset",
    "HarmBench",
    "JBBHarmfulBehaviors",
    "CatQA",
]
