import string
import torch
from torch.utils.data import TensorDataset
from torch.nn.functional import one_hot
from torchtext.data.utils import get_tokenizer

from utils import MAX_WORDS_PER_TRAINING, EMOTIONS


# Define tokenizer
tokenizer = get_tokenizer("basic_english")


class CustomTextDataset(TensorDataset):
    def __init__(self, vocab, sentences, emotions):
        self.max_len = MAX_WORDS_PER_TRAINING # the maximum length for each sequence 
        self.vocab = vocab
        self.sentences = sentences
        self.emotions = emotions
        self.unique_emotions = list(set(emotions))

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        emotion = self.emotions[idx]
        
        emotion_tensor = []
        for i in range(len(EMOTIONS)):
            if i == EMOTIONS.index(emotion):
                emotion_tensor.append(1)
            else:
                emotion_tensor.append(0)

        emotion_tensor = torch.tensor(emotion_tensor) 

        print(sentence)
        print(emotion)
        # Token with stop words included
        tokens = tokenizer(sentence)
        # TODO : test tokens without stopwords 
        indexes = self.vocab(tokens) # list of indexes
        
        # Bringing all samples to max_len
        if len(indexes) < self.max_len:
            same_size_sample = indexes + ([0]* (self.max_len-len(indexes)))
        else:
            same_size_sample = indexes[:self.max_len]
        
        onehot = one_hot(torch.tensor(same_size_sample), num_classes=len(self.vocab))

        return onehot, emotion_tensor
