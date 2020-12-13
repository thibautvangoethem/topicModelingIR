import pickle
import numpy as np
import matplotlib.pyplot as plt

# This document contains al code used to generate the figure and tables present in the rapport.

def plot_document_word_count_distribution(documents):
    """
    Creates a plot showing the distribution of document lengths.
    """
    documentLengths = [len(document) for document in documents]

    plt.figure(figsize=(16, 7), dpi=160)
    plt.hist(documentLengths, bins=1000, color='navy')
    plt.gca().set(xlim=(0, 5000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 5000, 11))
    plt.title('Distribution of Document Word Counts before pre-processing', fontdict=dict(size=22))
    plt.show()


def plot_topic_distribution(document_topic_mixture):
    """
    Creates a plot showing the distribution of documents among the topics
    """
    topic_count = len(document_topic_mixture[0])
    topic_count_distribution = [0]*topic_count
    for document in document_topic_mixture:
        max_topic = document.index(max(document))
        topic_count_distribution[max_topic] += 1

    plt.figure(figsize=(16, 7), dpi=160)
    plt.bar(x=range(1, topic_count+1), height=topic_count_distribution)
    plt.gca().set(ylabel='Number of Documents', xlabel='Topic')
    if topic_count > 20:
        plt.xticks(np.linspace(0, topic_count, 11))
    else:
        plt.xticks(range(1, topic_count+1))
    plt.title("Number of documents per topic for %d topics"%topic_count, fontdict=dict(size=22))
    plt.show()


def printLatexTable(term_topic_mixture):
    """
    Prints a table containing the top 20 words in each topic in latex formatting
    """
    print("\\begin{table}[h!]\n\\begin{adjustwidth}{-5cm}{-5cm}\n\\begin{center}")
    print("\\begin{tabular}{ |c|m{15cm}| } ")
    print("\\hline")
    print("topic & top 20 words \\\\")
    print("\\hline")
    for idx, topic in enumerate(term_topic_mixture):
        sorted_topics=sorted(topic.items(), key=lambda x: x[1], reverse=True)
        print("%s & " % (idx+1), end="")
        for term in sorted_topics[:20]:
            print(term[0], end=', ')
        print("\\\\\n\\hline")
    print("\\end{tabular}")
    print("\\caption{Top 20 words for each topic with %d topics}"%len(term_topic_mixture))
    print("\\label{top_terms_%d}"%len(term_topic_mixture))
    print("\\end{center}\n\\end{adjustwidth}\n\\end{table}")


def printLatexTableFor50(term_topic_mixture):
    """
    Prints a table containing the top 10 words for 50 topics in latex formatting
    """
    print("\\begin{table}[h!]\n\\begin{adjustwidth}{-5cm}{-5cm}\n\\begin{center}")
    print("\\begin{tabular}{ |c|m{7.5cm}|c|m{7.5cm}| }")
    print("\\hline")
    print("topic & top 10 words & topic & top 10 words\\\\")
    print("\\hline")
    for i in range(25):
        j = i+25
        topic1 = term_topic_mixture[i]
        topic2 = term_topic_mixture[j]
        sorted_topics1 = sorted(topic1.items(), key=lambda x: x[1], reverse=True)
        sorted_topics2 = sorted(topic2.items(), key=lambda x: x[1], reverse=True)
        print("%s & " % (i + 1), end="")
        for term in sorted_topics1[:10]:
            print(term[0], end=', ')
        print("& %s & " % (j + 1), end="")
        for term in sorted_topics2[:10]:
            print(term[0], end=', ')
        print("\\\\\n\\hline")
    print("\\end{tabular}")
    print("\\caption{Top 10 words for each topic with %d topics}" % len(term_topic_mixture))
    print("\\label{top_terms_%d}" % len(term_topic_mixture))
    print("\\end{center}\n\\end{adjustwidth}\n\\end{table}")


if __name__ == "__main__":
    plt.rc('font', size=18)
    with open('obj/document_topic_mixture_topics_10_topics_improved.pkl', 'rb') as file:
         documents = pickle.load(file)
    plot_topic_distribution(documents)
