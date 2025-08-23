"""Dataset classes for handling harmful prompt data samples."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable
from pydantic import BaseModel, FilePath
import pandas as pd
import os


class DataSample(BaseModel):
    """A single data sample containing a harmful prompt and associated metadata."""

    # The input prompt for the model
    harmful_prompt: str
    # The expected output from the model
    target: str
    # Any additional metadata related to the sample
    metadata: dict = None


class BaseDataset(ABC):
    """Abstract base class for datasets containing harmful prompts."""

    @abstractmethod
    def __iter__(self):
        """Iterate over data samples in the dataset.

        Returns:
            Iterable[DataSample]: Iterator of data samples.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the dataset."""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()


class StrongReject(BaseDataset):
    """Dataset for strong reject prompts loaded from CSV.

           @misc{souly2024strongreject,
          title={A StrongREJECT for Empty Jailbreaks},
          author={Alexandra Souly and Qingyuan Lu and Dillon Bowen and Tu Trinh and Elvis Hsieh and Sana Pandey and Pieter Abbeel and Justin Svegliato and Scott Emmons and Olivia Watkins and Sam Toyer},
          year={2024},
          eprint={2402.10260},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

    """

    def __init__(self, data_path: FilePath | str = None):
        """Initialize the StrongReject dataset by loading data from CSV."""
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__), "strongreject_dataset.csv"
            )
        self.pd = pd.read_csv(data_path)

    def name(self) -> str:
        return "strong_reject"
    def __len__(self) -> int:
        return 313

    def __iter__(self) -> Iterable[DataSample]:
        """Lazily iterate over data samples from the strong reject dataset.

        Yields:
            DataSample: Data samples with harmful prompts and metadata.
        """
        for index, row in self.pd.iterrows():
            yield DataSample(
                harmful_prompt=row["forbidden_prompt"],
                target="",
                metadata={
                    "category": row["category"],
                    "source": "strong_reject_"+row["source"],
                },
            )


class HarmfulBehaviors(BaseDataset):
    def __init__(self, data_path: FilePath | str = None):
        """Initialize the HarmfulBehaviors dataset from the GCG paper by loading data from CSV.

            @misc{zou2023universal,
              title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
              author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
              year={2023},
              eprint={2307.15043},
              archivePrefix={arXiv},
              primaryClass={cs.CL}
        }

        """
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), "harmful_behaviors.csv")
        self.pd = pd.read_csv(data_path)

    def __iter__(self) -> Iterable[DataSample]:
        """Lazily iterate over data samples from the harmful behaviors dataset.

        Yields:
            DataSample: Data samples with harmful prompts and metadata.
        """
        for index, row in self.pd.iterrows():
            yield DataSample(
                harmful_prompt=row["goal"],
                target=row["target"],
                metadata={
                    "source": "harmful_behavior_gcg"
                },
            )

    def name(self) -> str:
        return "harmful_behaviors_gcg"


    def __len__(self) -> int:
        return 520
