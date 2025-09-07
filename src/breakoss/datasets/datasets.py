"""Dataset classes for handling harmful prompt data samples."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Callable
from pydantic import BaseModel, FilePath
import pandas as pd
from ..methods import BaseMethod
import pathlib
import json
import os
from loguru import logger
from tqdm import tqdm

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

    @abstractmethod
    def __getitem__(self, index: int) -> DataSample:
        """Get a specific data sample by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            DataSample: The data sample at the specified index.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_transformation(self, jailbreak_method: BaseMethod, stored_path: pathlib.Path, batch_size: int = 1):
        """Apply a transformation function to each data sample in the dataset.

        Args:
            jailbreak_method (BaseMethod): Jailbreak Method Class to transform each data sample.
            stored_path (pathlib.Path): Path to store the transformed data sample.
            batch_size (int, optional): Batch size for transformation. Defaults to 1.
        Returns:
            BaseDataset: A new dataset with transformed data samples.
        """
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
            self.data_path = os.path.join(
                os.path.dirname(__file__), "strongreject_dataset.csv"
            )
        self.pd = pd.read_csv(self.data_path)

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
    def __getitem__(self, index: int) -> DataSample:
        """Get a specific data sample by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            DataSample: The data sample at the specified index.
        """
        row = self.pd.iloc[index]
        return DataSample(
            harmful_prompt=row["forbidden_prompt"],
            target="",
            metadata={
                "category": row["category"],
                "source": "strong_reject_" + row["source"],
            },
        )

    def apply_transformation(self, jailbreak_method: BaseMethod, stored_path: pathlib.Path, batch_size: int = 1):
        self.pd["forbidden_prompt_original"] = self.pd["forbidden_prompt"]
        new_prompts = [""]*len(self.pd["forbidden_prompt"])
        original_data_file_name, suffix = self.data_path.split("/")[-1].split(".")
        original_data_file_name = (
                original_data_file_name + "_" + jailbreak_method.name() + "." + suffix
        )
        for i in tqdm(range(0, len(self.pd), batch_size)):
            batch = self.pd["forbidden_prompt_original"].iloc[i:i + batch_size].tolist()
            transformed = jailbreak_method.jailbreak_transform_batch(batch)
            for j, idx in enumerate(range(i, min(i + batch_size, len(self.pd)))):
                new_prompts[idx] = transformed[j]
            self.pd["forbidden_prompt"] = pd.concat([pd.Series(new_prompts)], ignore_index=True)
            self.pd.to_csv(
                os.path.join(stored_path, original_data_file_name), index=False
            )
        logger.info(f"Transformed data saved to {os.path.join(stored_path, original_data_file_name)}")

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
            self.data_path = os.path.join(os.path.dirname(__file__), "harmful_behaviors.csv")
        self.pd = pd.read_csv(self.data_path)

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

    def __getitem__(self, index: int) -> DataSample:
        """Get a specific data sample by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            DataSample: The data sample at the specified index.
        """
        row = self.pd.iloc[index]
        return DataSample(
            harmful_prompt=row["goal"],
            target=row["target"],
            metadata={"source": "harmful_behavior_gcg"},
        )

    def apply_transformation(self, jailbreak_method: BaseMethod, stored_path: pathlib.Path, batch_size: int = 16):
        self.pd["goal_original"] = self.pd["goal"]
        new_prompts = [""]*len(self.pd["goal"])
        original_data_file_name, suffix = self.data_path.split("/")[-1].split(".")
        original_data_file_name = (
                original_data_file_name + "_" + jailbreak_method.name() + "." + suffix
        )

        for i in tqdm(range(0, len(self.pd), batch_size)):
            batch = self.pd["goal_original"].iloc[i:i + batch_size].tolist()
            transformed = jailbreak_method.jailbreak_transform_batch(batch)
            for j, idx in enumerate(range(i, min(i + batch_size, len(self.pd)))):
                new_prompts[idx] = transformed[j]
            self.pd["goal"] = pd.concat([pd.Series(new_prompts)], ignore_index=True)
            self.pd.to_csv(
                os.path.join(stored_path, original_data_file_name), index=False
            )
        logger.info(f"Transformed data saved to {os.path.join(stored_path, original_data_file_name)}")

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
            self.data_path = os.path.join(os.path.dirname(__file__), "harmbench_behaviors_text_all.csv")
        self.pd = pd.read_csv(self.data_path)

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

    def __getitem__(self, index: int) -> DataSample:
        """Get a specific data sample by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            DataSample: The data sample at the specified index.
        """
        row = self.pd.iloc[index]
        return DataSample(
            harmful_prompt=row["Behavior"],
            target="",
            metadata={"category": row["SemanticCategory"], "source": "harmbench"},
        )

    def apply_transformation(self, jailbreak_method: BaseMethod, stored_path: pathlib.Path, batch_size: int = 16):
        self.pd["Behavior_original"] = self.pd["Behavior"]
        new_prompts = [""]*len(self.pd["Behavior"])
        original_data_file_name, suffix = self.data_path.split("/")[-1].split(".")
        original_data_file_name = (
                original_data_file_name + "_" + jailbreak_method.name() + "." + suffix
        )

        for i in tqdm(range(0, len(self.pd), batch_size)):
            batch = self.pd["Behavior_original"].iloc[i:i + batch_size].tolist()
            transformed = jailbreak_method.jailbreak_transform_batch(batch)
            for j, idx in enumerate(range(i, min(i + batch_size, len(self.pd)))):
                new_prompts[idx] = transformed[j]
            self.pd["Behavior"] = pd.concat([pd.Series(new_prompts)], ignore_index=True)
            self.pd.to_csv(
                os.path.join(stored_path, original_data_file_name), index=False
            )
        logger.info(f"Transformed data saved to {os.path.join(stored_path, original_data_file_name)}")

    def name(self) -> str:
        return "harmbench"

    def __len__(self) -> int:
        return 400


class CatQA(BaseDataset):
    """Dataset for CatQA prompts."""

    def __init__(self, data_path: FilePath | str = None):
        """Initialize the CatQA dataset by loading data from CSV."""
        if data_path is None:
            self.data_path = os.path.join(os.path.dirname(__file__), "catqa_english.json")
        self.data_items = []
        with open(self.data_path, "r") as f:
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

    def __getitem__(self, index: int) -> DataSample:
        """Get a specific data sample by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            DataSample: The data sample at the specified index.
        """
        item = self.data_items[index]
        return DataSample(
            harmful_prompt=item["Question"],
            target="",
            metadata={"category": item.get("Category", ""),
                      "sub-category": item.get("Subcategory", ""),
                      "source": "catqa"
                      },
        )

    def apply_transformation(self, jailbreak_method: BaseMethod, stored_path: pathlib.Path, batch_size: int = 1):
        original_data_file_name, suffix = self.data_path.split("/")[-1].split(".")
        original_data_file_name = (
                original_data_file_name + "_" + jailbreak_method.name() + "." + suffix
        )

        for item in self.data_items:
            item["Question_original"] = item["Question"]
            item["Question"] = ""

        for i in tqdm(range(0, len(self.data_items), batch_size)):
            batch = [self.data_items[j]["Question_original"] for j in range(i, min(i + batch_size, len(self.data_items)))]
            transformed = jailbreak_method.jailbreak_transform_batch(batch)
            for j, idx in enumerate(range(i, min(i + batch_size, len(self.data_items)))):
                self.data_items[idx]["Question"] = transformed[j]

            with open(os.path.join(stored_path, original_data_file_name), "w") as f:
                for item in self.data_items:
                    f.write(json.dumps(item) + "\n")
        logger.info(f"Transformed data saved to {os.path.join(stored_path, original_data_file_name)}")

class JBBHarmfulBehaviors(BaseDataset):
    """Dataset for JBB Behaviors prompts."""

    def __init__(self, data_path: FilePath | str = None):
        """Initialize the JBB Behaviors dataset by loading data from CSV."""
        if data_path is None:
            self.data_path = os.path.join(os.path.dirname(__file__), "jbb-harmful-behaviors.csv")
        self.pd = pd.read_csv(self.data_path)

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

    def __getitem__(self, index: int) -> DataSample:
        """Get a specific data sample by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            DataSample: The data sample at the specified index.
        """
        row = self.pd.iloc[index]
        return DataSample(
            harmful_prompt=row["Goal"],
            target=row["Target"],
            metadata={"category": row.get("Category", ""),
                      "behavior": row.get("Behavior", ""),
                      "source": "jbb_behaviors"
                      },
        )

    def apply_transformation(self, jailbreak_method: BaseMethod, stored_path: pathlib.Path, batch_size: int = 16):
        self.pd["Goal_original"] = self.pd["Goal"]
        new_prompts = [""]*len(self.pd["Goal"])

        original_data_file_name, suffix = self.data_path.split("/")[-1].split(".")
        original_data_file_name = (
            original_data_file_name + "_" + jailbreak_method.name() + "." + suffix
        )
        for i in tqdm(range(0, len(self.pd), batch_size)):
            batch = self.pd["Goal_original"].iloc[i:i + batch_size].tolist()
            transformed = jailbreak_method.jailbreak_transform_batch(batch)
            for j, idx in enumerate(range(i, min(i + batch_size, len(self.pd)))):
                new_prompts[idx] = transformed[j]
            self.pd["Goal"] = pd.concat([pd.Series(new_prompts)], ignore_index=True)
            self.pd.to_csv(
                os.path.join(stored_path, original_data_file_name), index=False
            )
        logger.info(f"Transformed data saved to {os.path.join(stored_path, original_data_file_name)}")