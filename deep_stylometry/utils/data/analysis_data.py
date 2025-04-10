# deep_stylometry/utils/data/analysis_data.py

from typing import Optional

import lightning as L
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from deep_stylometry.utils.helpers import get_tokenizer


class DomainPerplexityDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_proc: int,
        tokenizer_name: str,
        max_length: int,
        ds_name: str = "almanach/HALvest-Contrastive",
        config_name: Optional[str] = None,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.config_name = config_name or "base-2"
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.max_length = max_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        if tokenizer_name == "openai-community/gpt2":
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self):
        load_dataset(self.ds_name, self.config_name, split="valid")

    def setup(self, stage: str):
        ds = load_dataset(self.ds_name, self.config_name, split="valid")

        def split_and_tokenize(split):
            split_ds = ds  # [split]
            texts, domains = [], []

            # Collect all texts and domains
            for ex in split_ds:
                # Process query and key texts
                for text, domain_list in zip(
                    [ex["query_text"], ex["key_text"]],
                    [ex["query_domains"], ex["key_domains"]],
                ):
                    # Explode domains into individual entries
                    for domain in domain_list:
                        texts.append(str(text))  # Ensure text is string
                        domains.append(domain)

            # Tokenize all texts at once
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            return Dataset.from_dict({**tokenized, "domains": domains})

        if stage == "test":
            self.test_dataset = split_and_tokenize("test").with_format("torch")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            shuffle=False,
        )
