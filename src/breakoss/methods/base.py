"""Base classes for jailbreaking methods.

This module provides the abstract base class that all jailbreaking methods must implement.
Jailbreaking methods are techniques used to bypass safety mechanisms in language models
to elicit potentially harmful responses for research and evaluation purposes.
"""

from abc import ABC, abstractmethod


class BaseMethod(ABC):
    """Abstract base class for jailbreaking methods.

    This class defines the interface that all jailbreaking methods must implement.
    Methods transform harmful prompts in various ways to attempt to bypass model
    safety mechanisms while maintaining the core intent of the original prompt.
    """

    @abstractmethod
    def jailbreak_transform(self, harmful_prompt: str) -> str:
        """Transform a harmful prompt to attempt bypassing safety mechanisms.

        This method takes a harmful prompt and applies the specific jailbreaking
        technique to transform it. The goal is to maintain the harmful intent
        while potentially evading the model's safety filters.

        Args:
            harmful_prompt: The original harmful prompt to transform

        Returns:
            The transformed prompt with jailbreaking techniques applied
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return the name/identifier of this jailbreaking method.

        Returns:
            String identifier for this jailbreaking method
        """
        raise NotImplementedError
