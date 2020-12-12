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

def plot_topic_distribution(document_topic_mixture):
    topic_count = len(document_topic_mixture[0])
    topic_count_distribution = [0]*topic_count
    for document in document_topic_mixture:
        max_topic = document.index(max(document))
        topic_count_distribution[max_topic] += 1

    plt.figure(figsize=(16, 7), dpi=160)
    plt.bar(x=range(topic_count), height=topic_count_distribution)
    plt.gca().set(ylabel='Number of Documents', xlabel='Topic')
    if topic_count > 20:
        plt.xticks(np.linspace(0, topic_count, 11))
    else:
        plt.xticks(range(topic_count))
    plt.title("Number of documents per topic for %d topics"%topic_count, fontdict=dict(size=22))
    plt.show()


def printLatexTable(term_topic_mixture):
    print("\\begin{table}[h!]\n\\begin{adjustwidth}{-5cm}{-5cm}\n\\begin{center}")
    print("\\begin{tabular}{ |c|m{15cm}| } ")
    print("\\hline")
    print("topic & top 20 words \\\\")
    print("\\hline")
    for idx, topic in enumerate(term_topic_mixture):
        sorted_topics=sorted(topic.items(), key=lambda x: x[1], reverse=True)
        print("%s & " % idx, end="")
        for term in sorted_topics[:20]:
            print(term[0], end=', ')
        print("\\\\\n\\hline")
    print("\\end{tabular}")
    print("\\caption{Top 20 words for each topic with %d topics}"%len(term_topic_mixture))
    print("\\label{top_terms_%d}"%len(term_topic_mixture))
    print("\\end{center}\n\\end{adjustwidth}\n\\end{table}")


if __name__ == "__main__":
    #print("done")
    # plt.rc('font', size=18)
    with open('obj/term_topic_mixture_20.pkl', 'rb') as file:
         documents = pickle.load(file)
    printLatexTable(documents)
    # plot_topic_distribution(documents)
