import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from RNN import RNN
from CustomDataset import CustomTextDataset
from utils import time_since, load_files, plot_confusion_matrix, plot_perf


def main():
    batch_size = 10  # amount of data treated each time
    n_epochs = 20  # number of time the dataset will be read
    max_sentence_len = 5
    learning_rate = 0.005
    n_embedding = 256
    n_hidden = 128

    # DEBUG
    print_every = 5000
    plot_every = 1000

    stopwords = ['i', 'a', 'im', 'am', 'me', 'my', 'he', 'him', 'she', 'her', 'it',
                 'us', 're', 'own', 'isn', 'isnt', 'is', 'are',  'do', 'be', 'go', 'yet'
                 'in', 'the', 'to', 'so', 'if', 'and', 'dr', 'for', 'by', 'its', 'but',
                 'm', 'this', 'up', 'yes', 'up', 'all', 'at', 'that', 'out', 'or', 'too',
                 'on', 'ive', 'of', 'as', 'bit',   'jo', 't', 'don', 's', 'oh', 'an', 'q', 
                 'we', 'they', 'dh', 'n', 'ok', 'okay' 'la']

    train_filepath = "dataset/train.txt"  # 16000 sentences
    val_filepath = "dataset/val.txt"     # 2000 sentences
    test_filepath = "dataset/test.txt"   # 2000 sentences
    files = [train_filepath, val_filepath, test_filepath]

    samples, targets, vocabulary, labels = load_files(files, stopwords)
    train_samples = samples[0]
    train_targets = targets[0]
    val_samples = samples[1]
    val_targets = targets[1]
    test_samples = samples[2]
    test_targets = targets[2]

    vocab = build_vocab_from_iterator(vocabulary, specials=["<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    # print("len", len(vocab))

    # Build emotions vocabulary and get total number of emotions
    labels_vocab = build_vocab_from_iterator(labels)
    n_emotions = len(labels_vocab)

    # Load data
    train_dataset = CustomTextDataset(
        train_samples, train_targets, vocab, labels_vocab, max_sentence_len, onehot=True)
    val_dataset = CustomTextDataset(
        val_samples, val_targets, vocab, labels_vocab, max_sentence_len, onehot=True)
    test_dataset = CustomTextDataset(
        test_samples, test_targets, vocab, labels_vocab, max_sentence_len, onehot=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # DEBUG Dataset
    # print(len(train_dataset))
    # for X, Y in train_loader:
    #     print(X.shape, Y.shape)
    #     break

    # train_dataset.__getitem__(2)

    ##### RNN Training #####
    rnn = RNN(len(vocab), n_embedding, n_hidden,
              n_emotions, batch_size, learning_rate)
    optim = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    print(rnn)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    all_perf = []
    acc = 0

    for iter in range(n_epochs):
        start_time = time.time()
        for x, t in train_loader:
            # Transpose sentence tensor to change dimension order
            # from : tensor(batch_size, sentence_length, vocabulary_size)
            # to : tensor(sentence_length, batch_size, vocabulary_size)
            x = torch.transpose(x, 0, 1)
            # Create a zeroed initial hidden state
            hidden = rnn.init_hidden()

            # Feed the rnn the batch (word by word)
            for w in x:
                output, hidden = rnn(w.type(torch.FloatTensor), hidden)

            acc += torch.argmax(output, 1) == torch.argmax(t, 1)

            # Back-propagate
            loss = criterion(t.type(torch.FloatTensor), output)
            loss.backward()

            current_loss += loss

            optim.step()
            optim.zero_grad()

            # Add parameters' gradients to their values, multiplied by learning rate
            for p in rnn.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)

            # Print iter number, loss, name and guess
            # if iter % print_every == 0:
            #     guess, guess_i = emotion_from_output(output)
            #     correct = '✓' if guess == t else '✗ (%s)' % t
            #     print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_epochs *
            #         100, timeSince(start), loss, x, guess, correct))

            # # Add current loss avg to list of losses
            # if iter % plot_every == 0:
            #     all_losses.append(current_loss / plot_every)
            #     current_loss = 0

            # # # TODO : Plot the results
            # plt.figure()
            # plt.plot(all_losses)
        print(
            f"Epoch {iter + 1:2} - Training completed in {time_since(start_time)}.")

        ##### RNN Validation #####
        # Evaluate the model's results using validation dataset
        acc = 0
        for x, t in val_loader:
            # Transpose sentence tensor to change dimension order
            # from : tensor(batch_size, sentence_length, vocabulary_size)
            # to : tensor(sentence_length, batch_size, vocabulary_size)
            x = torch.transpose(x, 0, 1)
            # Create a zeroed initial hidden state
            hidden = rnn.init_hidden()

            # Feed the rnn the batch (word by word)
            for w in x:
                output, hidden = rnn(w.type(torch.FloatTensor), hidden)

            acc += torch.argmax(output, 1) == torch.argmax(t, 1)
        total = acc.sum()
        all_perf.append(total / len(val_samples))
        print(
            f"Accuracy: {total / len(val_samples):.6f} ({total} / {len(val_samples)})")

    ##### RNN Testing #####
    # Evaluate the model's results using test dataset, plots and confusion matrix

    # Initiaisation of the confusion matrix
    confusion_matrix = []
    for i in range(len(labels_vocab)):
        confusion_matrix.append([0 for _ in range(len(labels_vocab))])

    for x, t in test_loader:
        t = t[0].tolist()
        # Transpose sentence tensor to change dimension order
        # from : tensor(batch_size, sentence_length, vocabulary_size)
        # to : tensor(sentence_length, batch_size, vocabulary_size)
        x = torch.transpose(x, 0, 1)
        # Create a zeroed initial hidden state
        hidden = rnn.init_hidden()

        # Feed the rnn the batch (word by word)
        for w in x:
            output, hidden = rnn(w.type(torch.FloatTensor), hidden)

        # Record the predicted results in the confusion matrix
        output = torch.argmax(output, dim=1).tolist()

        for i in range(len(t)):
            confusion_matrix[t[i]][output[i]] += 1

    plot_confusion_matrix(confusion_matrix)
    plot_perf(all_perf)


if __name__ == "__main__":
    main()
