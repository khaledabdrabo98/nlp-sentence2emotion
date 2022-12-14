import time
import math
import seaborn as sn
import matplotlib.pyplot as plt

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Load one file, return the list of sentences (tokenized 
# with stopwords removed) and the list of emotions
def load_file(filename, stopwords):
    with open(filename, 'r') as file:
        str_f = file.read()
        lines = str_f.split('\n')

    list_sentences = list([])
    list_emotions = list([])

    for line in lines:
        # Ignore empty lines
        if line.strip() == "":
            break
        
        # TODO : test tokens without stopwords 
        sentence, emotion = line.split(';')
        sentence = [t for t in sentence.split(' ') if t not in stopwords]
        list_sentences.append(sentence)
        list_emotions.append([emotion])
        
    return list_sentences, list_emotions

# Reads multiple files and builds the global vocabularies
def load_files(files, stopwords):
    samples = []
    targets = []
    vocabulary = []
    labels = []

    for file in files:
        sentences, emotions = load_file(file, stopwords)
        samples.append(sentences)
        targets.append(emotions)
        vocabulary += sentences
        labels += emotions

    return samples, targets, vocabulary, labels

# Function that displays the confusion matrix 
def plot_confusion_matrix(matrix):
    ax = sn.heatmap(matrix, annot=False, annot_kws={"size": 16})
    ax.set_xlabel("Model predictions")
    ax.set_ylabel("Correct labels")
    plt.show()

# Function that displays performances (accuracy) plot over time  
def plot_perf(measures):
    plt.plot(measures)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
