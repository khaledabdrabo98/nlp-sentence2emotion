import time
import math
import string
from typing import Union, Iterable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from RNN import RNN
from CustomDataset import CustomTextDataset, EMOTIONS

tokenizer = get_tokenizer("basic_english")


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
        # Ignore empty lines
        if line.strip() == "":
            break
        
        sentence, emotion = line.split(';')
        list_sentences.append(sentence)
        list_emotions.append(emotion)

    return list_sentences, list_emotions

def tokenize_dataset(sentences):
    for text in sentences:
        yield tokenizer(text)

def build_vocab(sentences_dataset):
    vocab = build_vocab_from_iterator(tokenize_dataset(sentences_dataset), min_freq=1, specials=["<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    return vocab

def emotion_from_output(output):
    top_n, top_i = output.topk(1)
    emotion_i = top_i[0].item()
    return EMOTIONS[emotion_i], emotion_i


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

    vocab = build_vocab(train_sentences)
    print("len", len(vocab))

    # max_tokens = 0
    # avg_tokens = 0
    # for sentence in train_sentences:
    #     tokens = tokenizer(sentence)
    #     t_size = len(tokens)
    #     avg_tokens += t_size
    #     if t_size > max_tokens:
    #         max_tokens = len(tokens)

    # print(max_tokens) 
    # print(avg_tokens/len(train_sentences))
    # # Output : Max : 66
    # #          Avg : 19.166

    # tokens_w_stopwords = tokenizer("Hello how are you?, Welcome to CoderzColumn!!")
    # tokens_without_stopwords = tokenizer("Hello how are you Welcome to CoderzColumn")
    # indexes = vocab(tokens_w_stopwords)

    # print(tokens_w_stopwords, indexes)
    # print(vocab["<UNK>"])

    # Get total number of emotions
    # unique_emotions = list(set(train_emotions))
    # n_emotions = len(unique_emotions)
    n_emotions = len(EMOTIONS)

    train_dataset = CustomTextDataset(vocab, train_sentences, train_emotions)
    # val_dataset = CustomTextDataset(vocab, val_sentences, val_emotions)
    # test_dataset = CustomTextDataset(vocab, test_sentences, test_emotions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # print(len(train_dataset))
    # for X, Y in train_loader:
    #     print(X.shape, Y.shape)
    #     break
    
    # train_dataset.__getitem__(2)

    # TODO : Load data
    # Il est plus efficace de traiter par batch
    # préparer vos données globales (tensor(sentence_length, batch_size, vocabulary_size)),
    # pour alimenter votre réseau mot par mot (tensor(batch_size, vocabulary_size)),
    # utilisez si vous le souhaitez DataLoader de torch.utils.data

    ##### Training RNN #####

    # TODO : Train RNN
    n_embedding = 128
    n_hidden = 128
    rnn = RNN(len(vocab), n_embedding, n_hidden, n_emotions, batch_size)

    print(rnn)

    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_epochs + 1):
        for x, t in train_loader:
            print("onehot", x.shape)
            print("emo", t)

            output, loss = rnn.train(x, t)
            current_loss += loss

            # Print iter number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = emotion_from_output(output)
                correct = '✓' if guess == t else '✗ (%s)' % t
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_epochs *
                    100, timeSince(start), loss, x, guess, correct))

            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

            # # TODO : Plot the results
            plt.figure()
            plt.plot(all_losses)

        # TODO : Evaluate the results
        # for x, t in test_loader:
        # acc = 0.
		# # on lit toutes les donnéees de test
		# for x,t in test_loader:
		# 	# on calcule la sortie du modèle
		# 	y = rnn(x)
		# 	# on regarde si la sortie est correcte
		# 	acc += torch.argmax(y,1) == torch.argmax(t,1)
		# # on affiche le pourcentage de bonnes réponses
		# print(acc/data_test.shape[0] * 100)

        
        # TODO : Invent new feature (run rnn on user input?)


if __name__ == "__main__":
    main()
