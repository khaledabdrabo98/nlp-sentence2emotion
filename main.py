import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from RNN import RNN
from CustomDataset import CustomTextDataset
from utils import lineToTensor

MIN_WORD_FREQUENCY = 50
FIXED_SENTENCE_LENGTH = 300

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


def get_vocab_from_sentences(list_sentences):
    # Create correspondance between word -> word_id
    tokenizer = get_tokenizer("basic_english")
    tokens = [tokenizer(sentence) for sentence in list_sentences]
    vocab = build_vocab_from_iterator(tokens, specials=['<unk>'])
    return vocab


def emotion_from_output(output, emotions):
    top_n, top_i = output.topk(1)
    emotion_i = top_i[0].item()
    return emotions[emotion_i], emotion_i


def main():
    
    # TODO
    # Stock mot-id in DataLoader
    # Get OneHot encoding each time you process a word in the loop
    # batch de phrases (same size) => n words each time

    batch_size = 10  # amount of data treated each time
    n_epochs = 1000  # number of time the dataset will be read
    # learning_rate = 0.005 (already defined in RNN class)

    train_filepath = "dataset/train.txt" # 16000 sentences
    val_filepath = "dataset/val.txt"     # 2000 sentences
    test_filepath = "dataset/test.txt"   # 2000 sentences

    train_sentences, train_emotions = load_file(train_filepath)
    test_sentences, test_emotions = load_file(test_filepath)
    val_sentences, val_emotions = load_file(val_filepath)

    # TODO : Load data

    # get vocab
    train_vocab = get_vocab_from_sentences(train_sentences)    
    train_dataset = CustomTextDataset(train_vocab, train_sentences, train_emotions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_onehot, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_onehot, batch_size=1, shuffle=False)

    # Get total number of emotions
    unique_emotions = list(set(train_emotions))
    n_emotions = len(unique_emotions)

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
    rnn = RNN(len(train_dataset.vocab), n_hidden, n_hidden, n_emotions)

    print_every = 5000
    plot_every = 1000
    
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    print(len(train_dataset))

    start = time.time()

    for n in range(n_epochs):
        for word_tensors, emotion_tensor in train_loader:
            for w_tensor in word_tensors:
                output, loss = rnn.train(emotion_tensor, w_tensor)
                print(emotion_tensor)
                print(w_tensor)
                current_loss += loss

                # # Print iter number, loss, name and guess
                # if iter % print_every == 0:
                #     guess, guess_i = emotion_from_output(output)
                #     correct = '✓' if guess == emotion else '✗ (%s)' % emotion
                #     print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_epochs *
                #         100, timeSince(start), loss, sentence, guess, correct))

                # Add current loss avg to list of losses
                if iter % plot_every == 0:
                    all_losses.append(current_loss / plot_every)
                    current_loss = 0

    # TODO : Plot the results
    plt.figure()
    plt.plot(all_losses)

    # TODO : Evaluate the results

    # TODO : Invent new feature (run rnn on user input?)


main()
