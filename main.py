from torchtext.vocab import build_vocab_from_iterator

MIN_WORD_FREQUENCY = 50


def load_file(filename):
    with open(filename, 'r') as file:
        str_f = file.read()
        lines = str_f.split('\n')

    list_sentences = list([])
    list_feelings = list([])

    for line in lines:
        # TODO ignore empty lines
        sentence, feeling = line.split(';')
        list_sentences.append(sentence)
        list_feelings.append(feeling)

    return list_sentences, list_feelings


def build_vocab(data_iter, tokenizer):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def main():
    sentences, feelings = load_file("val.txt")
    print(sentences)
    print(feelings)


main()
