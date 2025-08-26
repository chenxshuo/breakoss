"""Jailbreaking methods for bypassing language model safety mechanisms.

This package provides various techniques for transforming harmful prompts
to attempt bypassing safety filters in language models. These methods are
intended for research and evaluation of model robustness.
"""

from .cot_bypass import CoTBypass, BaseMethod
from .fake_overrefusal import FakeOverrefusal
from .intention_hijack import IntentHijack
from .plan_injection import PlanInjection

__all__ = ["CoTBypass", "BaseMethod", "FakeOverrefusal", "IntentHijack", "PlanInjection"]
