import torch
import random

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while True:
            subjects = random.sample(set(self.dataset.ids), self.batch_size[0])
            sample_indices = []
            for _subject in subjects:
                indexs = self.dataset.index_dict.loc[_subject, :, :].values
                indexs = indexs[indexs > -1].flatten().tolist()
                indexs = random.choices(indexs, k=self.batch_size[1])
                sample_indices += indexs
            yield sample_indices