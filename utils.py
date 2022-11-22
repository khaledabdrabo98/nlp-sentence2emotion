import torch

MAX_WORDS_PER_TRAINING = 10


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter, all_letters):
    return all_letters.find(letter)


# Turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter, all_letters, n_letters):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter, all_letters)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
def lineToTensor(line, all_letters, n_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter, all_letters)] = 1
    return tensor


# Divide list into chunks of size n
def divide_chunks(l, n):
    return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]


# Divide list into equal chunks (by adding Mockup word) 
def equal_chunks(l, n):
    chunks = divide_chunks(l, n)
    for chunk in chunks:
        if len(chunk) != MAX_WORDS_PER_TRAINING:
            while (len(chunk) < MAX_WORDS_PER_TRAINING):
                chunk.append('<unk>')

    return chunks
