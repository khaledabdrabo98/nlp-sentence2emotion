import time
import math
import string
from typing import Union, Iterable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from RNN import RNN
from utils import lineToTensor

MIN_WORD_FREQUENCY = 50


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


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


def lists_to_one_hot(list_sentences, list_emotions):
    # Create correspondance between word -> word_id
    tokenizer = get_tokenizer("basic_english")
    tokens = [tokenizer(sentence) for sentence in list_sentences]
    vocab = build_vocab_from_iterator(tokens)

    # Create One Hot encoding for our sentences using emotions as keys
    encoding = custom_one_hot(vocab, list_emotions)

    return encoding


def emotion_from_output(output, emotions):
    top_n, top_i = output.topk(1)
    emotion_i = top_i[0].item()
    return emotions[emotion_i], emotion_i


def main():

    batch_size = 10  # amount of data treated each time
    n_epochs = 10000  # number of time the dataset will be read
    # learning_rate = 0.005 (already defined in RNN class)

    train_filepath = "dataset/train.txt"  # 16000 sentences
    val_filepath = "dataset/val.txt"     # 2000 sentences
    test_filepath = "dataset/test.txt"   # 2000 sentences

    val_sentences, val_emotions = load_file(val_filepath)
    train_sentences, train_emotions = load_file(train_filepath)
    test_sentences, test_emotions = load_file(test_filepath)

    # TODO : Load data
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Il est plus efficace de traiter par batch
    # préparer vos données globales (tensor(sentence_length, batch_size, vocabulary_size)),
    # pour alimenter votre réseau mot par mot (tensor(batch_size, vocabulary_size)),
    # utilisez si vous le souhaitez DataLoader de torch.utils.data

    # train_sentences_tensor = torch.Tensor(train_sentences)
    # train_emotions_tensor = torch.Tensor(train_emotions)

    # train_dataset = TensorDataset(train_sentences_tensor, train_emotions)
    # val_dataset = TensorDataset(val_sentences, val_emotions)
    # test_dataset = TensorDataset(test_sentences, test_emotions)

    # get vocab
    train_onehot = lists_to_one_hot(train_sentences, train_emotions)
    val_onehot = lists_to_one_hot(val_sentences, val_emotions)
    test_onehot = lists_to_one_hot(test_sentences, test_emotions)

    train_loader = DataLoader(train_onehot, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_onehot, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_onehot, batch_size=1, shuffle=False)

    # Get total number of emotions
    unique_emotions = list(set(train_emotions))
    n_emotions = len(unique_emotions)

    # Define alphabet
    all_letters = string.ascii_letters  # + " .,;'"
    n_letters = len(all_letters)

    ##### DEBUG #####

    # print(len(train_sentences))
    # print(val_emotions)
    # print(val_sentences)
    # print(val_emotions)
    # print(n_emotions)
    # print(unique_emotions)
    # print(encoding)

    ##### RNN First Test #####

    # n_hidden = 128
    # batch_size = 1
    # rnn = RNN(n_letters, n_hidden, n_hidden, n_emotions)
    # line = 'im feeling rather rotten so im not very ambitious right now'
    # input = lineToTensor(line, all_letters, n_letters)
    # hidden = torch.zeros(batch_size, n_hidden)

    # output, next_hidden = rnn(input[0], hidden)
    # print(output)

    # emotion, emotion_index = emotion_from_output(output, unique_emotions)
    # print(emotion)

    ##### Training RNN #####

    # TODO : Train RNN
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_hidden, n_emotions)

    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_epochs + 1):
        for emotion, sentence in train_loader:
            emotion_tensor = torch.tensor([unique_emotions.index(emotion)], dtype=torch.long)
            sentence_tensor = lineToTensor(sentence)

            output, loss = rnn.train(emotion_tensor, sentence_tensor)
            current_loss += loss

    # Print iter number, loss, name and guess
    # if iter % print_every == 0:
    #     guess, guess_i = emotion_from_output(output)
    #     correct = '✓' if guess == emotion else '✗ (%s)' % emotion
    #     print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters *
    #           100, timeSince(start), loss, sentence, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    # TODO : Plot the results
    # plt.figure()
    # plt.plot(all_losses)

    # TODO : Evaluate the results

    # TODO : Invent new feature (run rnn on user input?)


main()
