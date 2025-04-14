# tests/test_halvest_datamodule.py

import pytest

from deep_stylometry.utils.data import HALvestDataModule


@pytest.fixture
def data_module():
    return HALvestDataModule(
        batch_size=2,
        num_proc=1,
        tokenizer_name="openai-community/gpt2",
        max_length=128,
        ds_name="almanach/HALvest-Contrastive",
        map_batch_size=1000,
        load_from_cache_file=False,
        cache_dir=None,  # type: ignore
    )


def test_data_module_setup(data_module):
    data_module.prepare_data()
    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    assert "q_input_ids" in batch
    assert batch["q_input_ids"].shape == (2, 128)
