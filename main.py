import csv
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix,dok_matrix
import re
from sklearn.decomposition import LatentDirichletAllocation
libraryLDA = LatentDirichletAllocation(n_components=5, random_state=42)
# https://stackoverflow.com/questions/1276764/stripping-everything-but-alphanumeric-chars-from-a-string-in-python
removeNonAlphabet=re.compile('[\W_]+', re.UNICODE)
wordSet=set()


def readStopword():
    with open('data/stopwords.txt',"r") as stopFile:
        stopwords=stopFile.read().split("\n")
        return stopwords

stopwords=readStopword()


def simpleDataReader():
    data=list()
    with open('data/news_dataset.csv', newline='\n',encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((sanitiseData(row["content"])))
    return data


def sanitiseData(data):

    splitted=data.split(" ")
    removedStopWord = [removeNonAlphabet.sub('', removePossessive(word)).lower()
                       for word in splitted if word.lower()
                       not in stopwords and word != "" and not any(i.isdigit() for i in word)]

    wordSet.update(removedStopWord)
    return removedStopWord

def removePossessive(word):
    word=word.replace("'s", '')
    word=word.replace("’s", '')
    return word

def removeCommonWords(documents):
    wordlist=dict()
    for document in documents:
        for word in document:
            if word in wordlist:
                wordlist[word]+=1
            else:
                wordlist[word]=1
    sorted_words = sorted(wordlist.items(), key=lambda x: x[1], reverse=True)
    toremove=list()
    # remove top 1 percent of words
    new_document=list()
    for word in range(round(len(wordlist)*0.05)):
        toremove.append(sorted_words[word][0])
    for document in documents:
        new_document.append(list(set(document).difference(toremove)))
    return new_document


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


if __name__ == "__main__":
    data = simpleDataReader()
    print("data read, start removing common words")
    data = removeCommonWords(data)
    print("common words removed, start building corpus")
    topicSize = 15
    # Don't know this one yet
    vocabularySize = "?"
    # Here we need to sanitize our data
    # 1)tokenizen -> unigram/bigram / ...
    # 2)remove stop words of term frequency-inverse document frequency
    # 3)stemming -> no porter stemmer because this one is too agressive (Krovetz Stemmer? ->conclussion for paper below,(also there most likely are no spellig mistakes so we don't need to look at that)
    # Actuammy this papre comes to the conlclussion that no stemming is better most of the time (even uses NYT articles for it)
    # https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf
    documentSize = len(data)
    amountOfWords = len(wordSet)
    docwordList = list()
    wordToMatrixColumnDict = dict()
    columnToWord = []
    initialCorpus = dok_matrix((documentSize,amountOfWords),dtype=np.int)
    currentColumn = 0
    for RowIndex, document in enumerate(data):
        for word in document:
            columnIndex = None
            if word not in wordToMatrixColumnDict:
                columnToWord.append(word)
                wordToMatrixColumnDict[word]=currentColumn
                columnIndex = currentColumn
                currentColumn += 1
            else:
                columnIndex = wordToMatrixColumnDict[word]
            initialCorpus[RowIndex, columnIndex] += 1
    print("corpus build")
    print("total distinct words: ", len(wordSet))
    print("start training")
    lda = LatentDirichletAllocation(n_components=topicSize, n_jobs=-1)
    lda.fit(initialCorpus)
    lda.transform(initialCorpus)
    print_top_words(lda, columnToWord, 20)

    # Here we need to implement the actual LDA thing
        # https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2
    # https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158
