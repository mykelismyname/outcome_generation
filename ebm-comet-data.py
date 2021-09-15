# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import csv
import json
import os

import datasets
logger = datasets.logging.get_logger(__name__)

# meta data
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {ebm-comet},
author={Micheal Abaho.
},
year={2020}
}
"""

_DESCRIPTION = """\
A biomedical corpus containing 300 PubMed “Randomised controlled Trial” abstracts manually annotated with outcome 
classifications drawn from the taxonomy proposed by (Dodd et al., 2018). The abstracts were annotated by two experts with
 extensive experience in annotating outcomes in systematic reviews of clinical trials (Abaho et al., 2020). 
 Dodd et al. (2018)’s taxonomy hierarchically categorised 38 outcome domains into 5 outcome core areas and applied 
 this classification system to 299 published core outcome sets (COS) in the Core Outcomes Measures in Effectiveness (COMET)
database
"""
_HOMEPAGE = ""
_LICENSE = ""

#will be used once dataset is pushed to the hub
_URLs = {
    "",
}
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"

class EbmComet(datasets.GeneratorBasedBuilder):
    """Dataset of clinical outcomes annotated in randomised control trials."""

    VERSION = datasets.Version("1.1.0")

    # builder configs, none being used
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
        ]
    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=[
                            "O",
                            "B-Adverse-effects",
                            "I-Adverse-effects",
                            "B-Life-Impact",
                            "I-Life-Impact",
                            "B-Mortality",
                            "I-Mortality",
                            "B-Physiological-Clinical",
                            "I-Physiological-Clinical",
                            "B-Resource-use",
                            "I-Resource-use",
                        ]
                    )
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_urls = self.config.data_files
        assert len(data_urls) == 1, "We expected one data file to be loaded at a time, change the loading script if otherwise."
        if os.path.basename(data_urls[0]).__contains__('train'):
            my_urls = {"train": self.config.data_files}
        elif os.path.basename(data_urls[0]).__contains__('dev'):
            my_urls = {"dev": self.config.data_files}
        elif os.path.basename(data_urls[0]).__contains__('test'):
            my_urls = {"test": self.config.data_files}
        downloaded_files = dl_manager.download_and_extract(my_urls)
        file_name = os.path.basename(data_urls[0]).split('.')
        return [datasets.SplitGenerator(name=file_name[0], gen_kwargs={"filepath": downloaded_files[file_name[0]]})]

    def _generate_examples(self, filepath):
        """ Yields examples as (key, example) tuples. """
        assert len(filepath) == 1, "We expected one data file to be loaded at a time, change the loading script if otherwise."
        logger.info("generating examples from = %s", filepath[0])
        with open(filepath[0], encoding="utf-8") as f:
            current_tokens = []
            current_labels = []
            sentence_counter = 0
            for row in f:
                row = row.rstrip()
                if row:
                    token, label = row.split(" ")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    if not current_tokens:
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    sentence = (
                        sentence_counter,
                        {
                            "tokens": current_tokens,
                            "ner_tags": current_labels,
                        },
                    )
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            if current_tokens:
                yield sentence_counter, {
                    "tokens": current_tokens,
                    "ner_tags": current_labels,
                }
