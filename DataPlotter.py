from GibbsLDA import simpleDataReader, removeCommonAndUniqueWords
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_word_occurrences(documents):
    """
    doesn't give the results I wanted :(
    """
    occurences = []
    wordToIndexMap = dict()
    current_index = 0
    total_words = 0
    for document in documents:
        words_in_document = set()
        for word in document:
            if word in wordToIndexMap:
                if word not in words_in_document:
                    occurences[wordToIndexMap[word]] += 1
                    words_in_document.add(word)
            else:
                occurences.append(1)
                wordToIndexMap[word] = current_index
                words_in_document.add(word)
                current_index += 1
            total_words += 1
    print(total_words)
    print(len(documents))
    print(len(occurences))
    plt.figure(figsize=(16, 7), dpi=160)
    plt.hist(occurences, bins=1000, color='navy')
    plt.gca().set(xlim=(0, 2500), ylabel='Number of Words', xlabel='Documents containing word')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 2500, 11))
    plt.title('Distribution of word occurrences after pre-processing', fontdict=dict(size=22))
    plt.show()


def plot_document_word_count_distribution(documents):
    documentLengths = [len(document) for document in documents]

    figure = plt.figure(figsize=(16, 7), dpi=160)
    plt.hist(documentLengths, bins=1000, color='navy')
    plt.gca().set(xlim=(0, 5000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 5000, 11))
    plt.title('Distribution of Document Word Counts before pre-processing', fontdict=dict(size=22))
    plt.show()


if __name__ == "__main__":
    # documents = simpleDataReader()
    # documents, removed = removeCommonAndUniqueWords(documents)
    # with open('obj/pre_documents.pkl', 'wb') as file:  # save document_topic_mixture to a file
    #     pickle.dump(documents, file)
    plt.rc('font', size=18)
    with open('obj/pre_documents.pkl', 'rb') as file:
       documents = pickle.load(file)

    plot_document_word_count_distribution(documents)
