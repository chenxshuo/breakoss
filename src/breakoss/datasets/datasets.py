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
                    "source": "strong_reject_" + row["source"],
                },
            )


class HarmfulBehaviors(BaseDataset):
    def __init__(self, data_path: FilePath | str = None):
        """Initialize the HarmfulBehaviors/AdvBench dataset from the GCG paper by loading data from CSV.

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
                metadata={"source": "harmful_behavior_gcg"},
            )

    def name(self) -> str:
        return "harmful_behaviors_gcg"

    def __len__(self) -> int:
        return 520

class HarmBench(BaseDataset):
    def __init__(self, data_path: FilePath | str = None):
        """
        Initialize the HarmBench dataset from the HarmBench paper by loading data from CSV.
        https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv
        @article{mazeika2024harmbench,
          title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
          author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
          year={2024},
          eprint={2402.04249},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
        }
        """

        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), "harmbench_behaviors_text_all.csv")
        self.pd = pd.read_csv(data_path)

    def __iter__(self) -> Iterable[DataSample]:
        """Lazily iterate over data samples from the HarmBench dataset.

        Yields:
            DataSample: Data samples with harmful prompts and metadata.
        """
        for index, row in self.pd.iterrows():
            yield DataSample(
                harmful_prompt=row["Behavior"],
                target="",
                metadata={"category": row["SemanticCategory"], "source": "harmbench"},
            )

    def name(self) -> str:
        return "harmbench"

    def __len__(self) -> int:
        return 400

class CatQA(BaseDataset):
    """Dataset for CatQA prompts."""

    def __init__(self, data_path: FilePath | str = None):
        """Initialize the CatQA dataset by loading data from CSV."""
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), "catqa_english.json")
        self.data_items = []
        with open(data_path, "r") as f:
            for line in f.readlines():
                self.data_items.append(json.loads(line))


    def name(self) -> str:
        return "catqa"

    def __len__(self) -> int:
        return 550

    def __iter__(self) -> Iterable[DataSample]:
        """Lazily iterate over data samples from the CatQA dataset.

        Yields:
            DataSample: Data samples with harmful prompts and metadata.
        """
        for item in self.data_items:
            yield DataSample(
                harmful_prompt=item["Question"],
                target="",
                metadata={"category": item.get("Category", ""),
                          "sub-category": item.get("Subcategory", ""),
                          "source": "catqa"
                          },
            )

class JBBHarmfulBehaviors(BaseDataset):
    """Dataset for JBB Behaviors prompts."""

    def __init__(self, data_path: FilePath | str = None):
        """Initialize the JBB Behaviors dataset by loading data from CSV."""
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), "jbb-harmful-behaviors.csv")
        self.pd = pd.read_csv(data_path)

    def name(self) -> str:
        return "jbb_harmful_behaviors"

    def __len__(self) -> int:
        return 100

    def __iter__(self) -> Iterable[DataSample]:
        """Lazily iterate over data samples from the JBB Behaviors dataset.

        Yields:
            DataSample: Data samples with harmful prompts and metadata.
        """
        for index, row in self.pd.iterrows():
            yield DataSample(
                harmful_prompt=row["Goal"],
                target=row["Target"],
                metadata={"category": row.get("Category", ""),
                          "behavior": row.get("Behavior", ""),
                          "source": "jbb_behaviors"
                          },
            )