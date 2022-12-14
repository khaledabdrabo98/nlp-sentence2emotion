import torch
from torch.utils.data import TensorDataset
from torch.nn.functional import one_hot


class CustomTextDataset(TensorDataset):
    def __init__(self, samples, targets, vocab, labels_voc, max_sentence_length, onehot):
        self.max_len = max_sentence_length  # the maximum length for each sequence
        self.vocab = vocab
        self.labels_vocab = labels_voc
        self.samples = samples
        self.targets = targets
        self.enable_onehot = onehot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]
        
        # list of indexes
        indexes = self.vocab(sample)
        t = self.labels_vocab(target)

        # Bringing all samples to max_len
        if len(indexes) < self.max_len:
            same_size_sample = indexes + ([0] * (self.max_len-len(indexes)))
        else:
            same_size_sample = indexes[:self.max_len]

        x_batch = one_hot(torch.LongTensor(same_size_sample),
                    num_classes=len(self.vocab))
        
        if self.enable_onehot:
            t = one_hot(torch.tensor(t), num_classes=len(self.labels_vocab))
            t = torch.squeeze(t)
        
        # print(x_batch.shape)
        # print(t.shape)
        return x_batch, t
