import string
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

from utils import lineToTensor, equal_chunks
from utils import MAX_WORDS_PER_TRAINING 

EMOTIONS = ['joy', 'fear', 'surprise', 'sadness', 'anger', 'love']

# Define alphabet
all_letters = string.ascii_letters  # + " .,;'"
n_letters = len(all_letters)


class CustomTextDataset(Dataset):
    def __init__(self, vocab, sentences, emotions):
        self.vocab = vocab
        self.sentences = sentences
        self.emotions = emotions
        self.unique_emotions = list(set(emotions))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        emotion = self.emotions[idx]
        emotion_tensor = torch.tensor(
            EMOTIONS.index(emotion), dtype=torch.long)

        # List words of sentences
        one_hot_tensors = []
        words = sentence.split()
        
        # Divide list of words into equal chunks of mini lists (of size MAX_WORDS_PER_TRAINING)
        equal_words_lists = equal_chunks(words, MAX_WORDS_PER_TRAINING)
        
        for chunk in equal_words_lists:
            for word in chunk:
                one_hot_tensors.append(one_hot(torch.tensor(self.vocab[word]), num_classes=len(self.vocab)))

        return one_hot_tensors, emotion_tensor
