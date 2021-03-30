import os
from typing import List

import datasets
import pandas as pd

_DESCRIPTION = "Dataset informations can be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"


class ToxicityConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for the toxicity dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ToxicityConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class ToxicityDataset(datasets.GeneratorBasedBuilder):
    """Toxicity dataset"""
    BUILDER_CONFIG_CLASS = ToxicityConfig
    BUILDER_CONFIGS = [
        ToxicityConfig(
            name="default",
            description="Toxicity dataset",
            data_dir="../data/"
        )
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "comment_text": datasets.Value("string"),
                    "toxic": datasets.ClassLabel(names=["false", "true"]),
                    "severe_toxic": datasets.ClassLabel(names=["false", "true"]),
                    "obscene": datasets.ClassLabel(names=["false", "true"]),
                    "threat": datasets.ClassLabel(names=["false", "true"]),
                    "insult": datasets.ClassLabel(names=["false", "true"]),
                    "identity_hate": datasets.ClassLabel(names=["false", "true"]),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": self.config.data_dir + "train.txt",
            "test": self.config.data_dir + "test.txt"
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"]
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"]
                }
            )
        ]

    def _generate_examples(self, filepath):
        """ Yields examples. """
        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            example = {}
            example["comment_text"] = row["comment_text"]

            for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
                example[label] = int(row[label])
            yield row["id"], example
