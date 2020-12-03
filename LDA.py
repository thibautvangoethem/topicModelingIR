import numpy as np
from numpy.random import dirichlet
from random import choices
from scipy.sparse import csr_matrix,lil_matrix,dok_matrix


def weightsToCumWeights(weights):
    current_total = 0.0
    for i in range(len(weights)):
        current_total += weights[i]
        weights[i] = current_total


def LDA(alpha, beta, nb_documents, words_per_document):
    """
    Generates a corpus using the LDA model
    :param alpha: parameter for a diririchlet disribution that determines the topic distribution per document,
     this should be an list of size equal to the number of topics
    :param beta: parameter for a diririchlet disribution that determines the word distribution per topic,
     this should be an list of size equal to the number of words
    :param nb_documents: int specifying the number of documents the generated corpus should have
    :param words_per_document: int specifying the number of words a document contains
    (can later turned into a list to allow documents to have different amount of words)
    :return: A corpus
    """

    topic_distributions = []
    for i in range(len(alpha)):  # len(alpha) is equal to the number of topics
        topic_distributions.append(weightsToCumWeights(dirichlet(beta)))

    document_topic_distributions = []
    for i in range(nb_documents):
        document_topic_distributions.append(weightsToCumWeights(dirichlet(alpha)))

    corpus = dok_matrix((nb_documents, len(beta)), dtype=np.int)
    choiceList_topics = range(len(alpha))
    choiceList_words = range(len(beta))  # len(beta) is equal to the number of words
    for i in range(nb_documents):
        document_topic_distribution = document_topic_distributions[i]
        for j in range(words_per_document):
            topic = choices(choiceList_topics, cum_weights=document_topic_distribution)[0]
            word = choices(choiceList_words, cum_weights=topic_distributions[topic])[0]
            corpus[i, word] += 1
    print("done")
    return corpus

#LDA([1/3]*3, [1/100]*10, 20, 10)
LDA([1/10]*10, [1/100000]*10000, 1000, 10)
