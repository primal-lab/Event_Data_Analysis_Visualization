from torch.utils.data import Sampler

class BucketSampler(Sampler):
    def __init__(self, indices, batch_size, shuffle=True):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = self.indices.copy()
        if self.shuffle:
            import random
            random.shuffle(idxs)
        batches = [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
