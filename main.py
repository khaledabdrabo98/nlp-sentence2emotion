import string
from typing import Union, Iterable

import torch
from torch.nn.functional import one_hot
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from RNN import RNN
from utils import lineToTensor

MIN_WORD_FREQUENCY = 50


def load_file(filename):
    with open(filename, 'r') as file:
        str_f = file.read()
        lines = str_f.split('\n')

    list_sentences = list([])
    list_emotions = list([])

    for line in lines:
        # TODO ignore empty lines
        sentence, emotion = line.split(';')
        list_sentences.append(sentence)
        list_emotions.append(emotion)

    return list_sentences, list_emotions


def custom_one_hot(vocab, keys: Union[str, Iterable]):
    if isinstance(keys, str):
        keys = [keys]
    return one_hot(torch.tensor(vocab(keys)), num_classes=len(vocab))


def emotion_from_output(output, emotions):
    top_n, top_i = output.topk(1)
    emotion_i = top_i[0].item()
    return emotions[emotion_i], emotion_i


def main():
    filepath = "dataset/val.txt"
    # filepath = "dataset/train.txt"
    # filepath = "dataset/test.txt"

    sentences, emotions = load_file(filepath)

    # Get total number of emotions
    unique_emotions = list(set(emotions))
    n_emotions = len(unique_emotions)

    all_letters = string.ascii_letters  # + " .,;'"
    n_letters = len(all_letters)

    ##### DEBUG #####
    # print(sentences)
    # print(emotions)
    # print(n_emotions)
    # print(unique_emotions)

    # Create correspondance between word -> word_id
    tokenizer = get_tokenizer("basic_english")
    tokens = [tokenizer(sentence) for sentence in sentences]

    vocab = build_vocab_from_iterator(tokens)

    # Create One Hot encoding for our sentences using emotions as keys
    encoding = custom_one_hot(vocab, emotions)
    # TODO use encoding !

    ##### RNN First Test #####

    n_hidden = 128
    batch_size = 1
    rnn = RNN(n_letters, n_hidden, n_hidden, n_emotions)
    line = 'im feeling rather rotten so im not very ambitious right now'
    input = lineToTensor(line, all_letters, n_letters)
    hidden = torch.zeros(batch_size, n_hidden)

    output, next_hidden = rnn(input[0], hidden)
    print(output)

    emotion, emotion_index = emotion_from_output(output, unique_emotions)
    print(emotion)

    ##### Training RNN #####

    # n_hidden = 128
    # batch_size = 1
    # rnn = RNN(n_letters, n_hidden, n_hidden, n_emotions)

    # TODO : Load data
    # TODO : Train RNN
    # TODO : Plot the results
    # TODO : Evaluate the results
    # TODO : Invent new feature (run rnn on user input?)


main()
