import numpy as np
from scipy.sparse import csr_matrix,lil_matrix,dok_matrix
import csv
import re
from bayespy import nodes
from bayespy.inference.vmp.nodes.categorical import CategoricalMoments
from bayespy.inference import VB
import bayespy.plot as bpplt

removeNonAlphabet=re.compile('[\W_]+', re.UNICODE)
wordSet=set()


def readStopword():
    with open('data/stopwords.txt',"r") as stopFile:
        stopwords=stopFile.read().split("\n")
        return stopwords

def simpleDataReader():
    data = list()
    n_words = 0
    stopwords = readStopword()
    with open('data/news_dataset.csv', newline='\n',encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            new_document = sanitiseData(row["title"], stopwords)
            n_words += len(new_document)
            data.append(new_document)
    return data, n_words


def sanitiseData(data, stopwords):

    splitted=data.split(" ")
    removedStopWord = [removeNonAlphabet.sub('', removePossessive(word)).lower() for word in splitted if word.lower() not in stopwords]

    wordSet.update(removedStopWord)
    return removedStopWord


def removePossessive(word):
    word = word.replace("'s", '')
    word = word.replace("’s", '')
    return word


def read_data():
    data, n_words = simpleDataReader()
    # Don't know this one yet
    # Here we need to sanitize our data
    # 1)tokenizen -> unigram/bigram / ...
    # 2)remove stop words of term frequency-inverse document frequency
    # 3)stemming -> no porter stemmer because this one is too agressive (Krovetz Stemmer? ->conclussion for paper below,(also there most likely are no spellig mistakes so we don't need to look at that)
    # Actually this paper comes to the conlclussion that no stemming is better most of the time (even uses NYT articles for it)
    # https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf
    documentSize = len(data)
    vocabularySize = len(wordSet)
    wordToIndexDict = dict()
    initialCorpus = np.ndarray(shape=(n_words, ), dtype=np.int)
    word_documents = np.ndarray(shape=(n_words, ), dtype=np.int)
    currentVocabIndex = 0
    currentIndex = 0
    for DocIndex, document in enumerate(data):
        for word in document:
            if word not in wordToIndexDict:
                wordToIndexDict[word] = currentVocabIndex
                vocabIndex = currentVocabIndex
                currentVocabIndex += 1
            else:
                vocabIndex = wordToIndexDict[word]
            initialCorpus[currentIndex] = vocabIndex
            word_documents[currentIndex] = DocIndex
            currentIndex += 1

    return initialCorpus, word_documents, wordToIndexDict, documentSize, vocabularySize


if __name__ == "__main__":
    corpus, word_documents, wordToIndexDict, n_documents, n_vocabulary = read_data()
    print("data is read, starting training")
    n_words = len(corpus)
    print("total words: ", n_words)
    print("total documents: ", n_documents)
    n_topics = 10
    subset_size = 1500
    plates_multiplier = n_words / subset_size

    p_topic = nodes.Dirichlet(np.ones(n_topics), plates=(n_documents,), name='p_topic')
    print("created p_topic")
    p_word = nodes.Dirichlet(np.ones(n_vocabulary), plates=(n_topics,), name='p_word')
    print("created p_word")
    #document_indices = nodes.Constant(CategoricalMoments(n_documents), word_documents, name='document_indices')
    document_indices = nodes.Constant(CategoricalMoments(n_documents), word_documents[:subset_size], name='document_indices')
    print("created document_indices")
    #topics = nodes.Categorical(nodes.Gate(document_indices, p_topic), plates=(len(corpus),), name='topics')
    topics = nodes.Categorical(nodes.Gate(document_indices, p_topic),
                               plates = (subset_size,), plates_multiplier=(plates_multiplier,), name='topics')
    print("created topics")
    words = nodes.Categorical(nodes.Gate(topics, p_word), name='words')
    print("created words")

    #words.observe(corpus)
    p_topic.initialize_from_random()
    p_word.initialize_from_random()
    Q = VB(words, topics, p_word, p_topic, document_indices)

    Q.ignore_bound_checks = True
    delay = 1
    forgetting_rate = 0.7

    for n in range(15):
        print("iteration: ", n)
        subset = np.random.choice(n_words, subset_size)
        Q['words'].observe(corpus[subset])
        Q['document_indices'].set_value(word_documents[subset])  # Learn intermediate variables
        Q.update('topics')  # Set step length
        step = (n + delay) ** (-forgetting_rate)# Stochastic gradient for the global variables
        Q.gradient_step('p_topic', 'p_word', scale=step)
    print(p_topic)
    print(p_word)
    print(words)
    print(topics)
    Q.save(filename="models.hdf5")
    words.save("words.hdf5")
    topics.save("topics.hdf5")
    p_topic.save("p_topics.hdf5")
    p_word.save("p_word.hdf5")
    # #Q.update(repeat=10)
    # test = bpplt.pyplot.figure()
    # bpplt.hinton(Q['p_topic'])
    # bpplt.pyplot.title("Posterior topic distribution for each document")
    # bpplt.pyplot.xlabel("Topics")
    # bpplt.pyplot.ylabel("Documents")
    # bpplt.pyplot.plot(Q.L)
    # test.savefig('my_figure.png')
    print("done")
