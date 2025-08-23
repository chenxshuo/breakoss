from abc import ABC, abstractmethod


class BaseMethod(ABC):

    @abstractmethod
    def jailbreak_transform(self, harmful_prompts: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
