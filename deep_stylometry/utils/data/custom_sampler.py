# deep_stylometry/utils/data/custom_sampler.py

from torch.utils.data import Sampler


class PadLastBatchSampler(Sampler[int]):
    """A Sampler that yields all indices 0..(dataset_size-1) exactly once, then
    repeats the first few indices (if needed) so that the total number of
    indices is a multiple of `batch_size`."""

    def __init__(self, dataset_size: int, batch_size: int):
        super().__init__(None)
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        # How many examples don't fit into full batches?
        remainder = dataset_size % batch_size
        self.num_to_pad = (batch_size - remainder) if remainder != 0 else 0

    def __iter__(self):
        # 1. Start with a list of all indices
        indices = list(range(self.dataset_size))

        # 2. If needed, append the first `num_to_pad` indices to fill the last batch
        if self.num_to_pad:
            # This repeats indices [0, 1, 2, ...] as padding
            indices += indices[: self.num_to_pad]

        return iter(indices)

    def __len__(self):
        # Total number of indices the DataLoader will see
        return self.dataset_size + self.num_to_pad
