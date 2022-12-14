import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from RNN import RNN
from CustomDataset import CustomTextDataset
from utils import time_since, load_files, tokenize_dataset, normalize_sample, plot_confusion_matrix, plot_perf


def main():
    batch_size = 20  # amount of data treated each time
    n_epochs = 20  # number of time the dataset will be read
    max_sentence_len = 5
    learning_rate = 0.005
    n_embedding = 256
    n_hidden = 128

    # DEBUG
    print_every = 5000
    plot_every = 1000

    stopwords = ['i', 'a', 'im', 'am', 'me', 'my', 'he', 'him', 'she', 'her', 'it',
                 'us', 're', 'own', 'isn', 'isnt', 'is', 'are', 'do', 'be', 'go', 'yet'
                 'in', 'the', 'to', 'so', 'if', 'and', 'dr', 'for', 'by', 'its', 'but',
                 'm', 'this', 'up', 'yes', 'up', 'all', 'at', 'that', 'out', 'or', 'too',
                 'on', 'ive', 'of', 'as', 'bit',  'jo', 't', 'don', 's', 'oh', 'an', 'q', 
                 'we', 'they', 'dh', 'n', 'ok', 'okay', 'la']

    train_filepath = "dataset/train.txt" # 16000 sentences
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
    train_samples = tokenize_dataset(train_samples, vocab)
    val_samples = tokenize_dataset(val_samples, vocab)
    test_samples = tokenize_dataset(test_samples, vocab)
    
    train_targets = tokenize_dataset(train_targets, labels_vocab)
    val_targets = tokenize_dataset(val_targets, labels_vocab)
    test_targets = tokenize_dataset(test_targets, labels_vocab)
    
    train_indices = np.arange(len(train_samples),step=batch_size)
    val_indices = np.arange(len(val_samples), step=batch_size)
    test_indices = np.arange(len(test_samples), step=batch_size)
    
    # train_dataset = CustomTextDataset(
    #     train_samples, train_targets, vocab, labels_vocab, max_sentence_len, onehot=True)
    # val_dataset = CustomTextDataset(
    #     val_samples, val_targets, vocab, labels_vocab, max_sentence_len, onehot=True)
    # test_dataset = CustomTextDataset(
    #     test_samples, test_targets, vocab, labels_vocab, max_sentence_len, onehot=False)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # DEBUG Dataset
    # print(len(train_dataset))
    # for X, Y in train_loader:
    #     print(X.shape, Y.shape)
    #     break

    # train_dataset.__getitem__(2)

    rnn = RNN(len(vocab), n_embedding, n_hidden,
              n_emotions, batch_size, learning_rate)
    optim = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    print(rnn)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    all_perf = []

    for iter in range(n_epochs):
        start_time = time.time()
        np.random.shuffle(train_indices)
        
        ##### RNN Training #####
        for i in train_indices:
            x = train_samples[i:i + batch_size]
            t = train_targets[i:i + batch_size]
            
            # Bringing all samples to max_sentence_len
            normalize_sample(x, max_sentence_len)
            
            # Batch dimensioning
            x = [list(l) for l in zip(*x)]
            t = [l for l in zip(*t)]
            # One-hot encode the batch
            x = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=len(vocab))
            t = torch.nn.functional.one_hot(torch.LongTensor(t).flatten(), num_classes=len(labels_vocab))
            # print(x.shape)
            # print(t.shape)
            
            # Create a zeroed initial hidden state
            hidden = rnn.init_hidden()

            # Feed the rnn the batch (word by word)
            for w in x:
                output, hidden = rnn(w.type(torch.FloatTensor), hidden)

            # Back-propagate
            loss = criterion(t.type(torch.FloatTensor), output)
            loss.backward()

            current_loss += loss

            optim.step()
            optim.zero_grad()

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
        print(f"Epoch {iter + 1:2} - Training completed in {time_since(start_time)}.")

        ##### RNN Validation #####
        # Evaluate the model's results using validation dataset
        acc = 0
        np.random.shuffle(val_indices)
        for v in val_indices:
            x = val_samples[v:v + batch_size]
            t = val_targets[v:v + batch_size]
            
            # Bringing all samples to max_sentence_len
            normalize_sample(x, max_sentence_len)
            
            # Batch dimensioning
            x = [list(l) for l in zip(*x)]
            t = [l for l in zip(*t)]
            # One-hot encode the batch
            x = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=len(vocab))
            t = torch.nn.functional.one_hot(torch.LongTensor(t).flatten(), num_classes=len(labels_vocab))
            
            # Create a zeroed initial hidden state
            hidden = rnn.init_hidden()

            # Feed the rnn the batch (word by word)
            for w in x:
                output, hidden = rnn(w.type(torch.FloatTensor), hidden)

            acc += torch.argmax(output, 1) == torch.argmax(t, 1)
        total = acc.sum()
        all_perf.append(total / len(val_samples))
        print(f"Accuracy: {total / len(val_samples):.6f} ({total} / {len(val_samples)})")

    ##### RNN Testing #####
    # Evaluate the model's results using test dataset, plots and confusion matrix

    # Initiaisation of the confusion matrix
    confusion_matrix = []
    for i in range(len(labels_vocab)):
        confusion_matrix.append([0 for _ in range(len(labels_vocab))])

    for t in test_indices:
        x = test_samples[t:t + batch_size]
        t = test_targets[t:t + batch_size]
        
        # Bringing all samples to max_sentence_len
        normalize_sample(x, max_sentence_len)
        
        # Batch dimensioning
        x = [list(l) for l in zip(*x)]
        t = [i for sub in t for i in sub]
        # One-hot encode the batch
        x = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=len(vocab))
        
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
