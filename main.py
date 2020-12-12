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
                       not in stopwords and word != "" and len(word) > 2 and not any(i.isdigit() for i in word)]

    wordSet.update(removedStopWord)
    return removedStopWord


def removePossessive(word):
    word=word.replace("'s", '')
    word=word.replace("’s", '')
    return word


def removeCommonWords(documents):
    wordlist = dict()
    nb_document_per_word = dict()
    total_documents = len(documents)
    for document in documents:
        words_in_document = set()
        for word in document:
            if word in wordlist:
                wordlist[word] += 1
                if word not in words_in_document:
                    nb_document_per_word[word] += 1
            else:
                wordlist[word] = 1
                nb_document_per_word[word] = 1
            words_in_document.add(word)
    sorted_words = sorted(wordlist.items(), key=lambda x: x[1], reverse=True)
    toremove = set()
    unique_words = 0
    for word, frequentie in nb_document_per_word.items():
        if frequentie <= 10:
            toremove.add(word)
            unique_words += 1
        elif frequentie >= total_documents*0.40:
            toremove.add(word)
            unique_words += 1
    print("removed %d unique words" % unique_words)

    # remove top 1 percent of words
    new_document = list()
    for word in range(round(len(wordlist)*0.01)):
        toremove.add(sorted_words[word][0])
    for document in documents:
        new_document.append([word for word in document if word not in toremove])

    return new_document, toremove


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
    data, removed = removeCommonWords(data)
    wordSet = wordSet - removed
    print("common words removed, start building corpus")
    topicSize = 20
    # Don't know this one yet
    vocabularySize = "?"
    # Here we need to sanitize our data
    # 1)tokenizen -> unigram/bigram / ...
    # 2)remove stop words of term frequency-inverse document frequency
    # 3)stemming -> no porter stemmer because this one is too agressive
    # (Krovetz Stemmer? ->conclussion for paper below,
    # (also there most likely are no spellig mistakes so we don't need to look at that)
    # Actuammy this papre comes to the conlclussion that no stemming is better most of the time (even uses NYT articles for it)
    # https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf
    documentSize = len(data)
    amountOfWords = len(wordSet)
    print(amountOfWords)
    docwordList = list()
    wordToMatrixColumnDict = dict()
    columnToWord = []
    initialCorpus = dok_matrix((documentSize,amountOfWords), dtype=np.int)
    currentColumn = 0
    for RowIndex, document in enumerate(data):
        for word in document:
            columnIndex = None
            if word not in wordToMatrixColumnDict:
                columnToWord.append(word)
                wordToMatrixColumnDict[word] = currentColumn
                columnIndex = currentColumn
                currentColumn += 1
            else:
                columnIndex = wordToMatrixColumnDict[word]
            initialCorpus[RowIndex, columnIndex] += 1
        if RowIndex % 10000 == 0:
            print("build corpus for %f %% of the documents" % ((RowIndex/documentSize)*100))
    print("corpus build")
    print("total distinct words: ", len(wordSet))
    print("start training")
    lda = LatentDirichletAllocation(n_components=topicSize, n_jobs=-1)
    lda.fit(initialCorpus)
    lda.transform(initialCorpus)
    print_top_words(lda, columnToWord, 20)
