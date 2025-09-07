"""H-CoT jailbreak method.


"""

from .base import BaseMethod

class HCoT(BaseMethod):
    """H-CoT jailbreak method.

    This method appends a chain-of-thought reasoning section that attempts to
    convince the model that the harmful request is actually allowed content
    and should be complied with. It uses special tokens and reasoning to
    bypass safety mechanisms.
    """

    def name(self) -> str:
        """Return the name of this jailbreaking method.

        Returns:
            String identifier 'HCoT'
        """
        return "HCoT"

    def jailbreak_transform(self, harmful_prompt: str) -> str:
        """Transform the harmful prompt using H-CoT bypass.

        Appends a reasoning section that attempts to convince the model the
        request is allowed content that should be complied with.

        Args:
            harmful_prompt: The original harmful prompt to transform

        Returns:
            The original prompt with H-CoT bypass reasoning appended
        """
        return ""