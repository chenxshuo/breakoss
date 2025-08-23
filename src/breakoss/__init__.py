"""A research toolkit providing a bag of techniques to jailbreak OpenAI’s open-source models."""

from .api import hello, square

# being explicit about exports is important!
__all__ = [
    "hello",
    "square",
]
