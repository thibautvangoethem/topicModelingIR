import csv
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix,dok_matrix
import re
from sklearn.decomposition import LatentDirichletAllocation
from LDA import LDA
libraryLDA = LatentDirichletAllocation(n_components=5, random_state=42)
# https://stackoverflow.com/questions/1276764/stripping-everything-but-alphanumeric-chars-from-a-string-in-python
removeNonAlphabet=re.compile('[\W_]+', re.UNICODE)
wordSet=set()


def readStopword():
    with open('data/stopwords.txt',"r") as stopFile:
        stopwords=stopFile.read().split("\n")
        return stopwords
    return None

stopwords=readStopword()


def simpleDataReader():
    data=list()
    with open('data/news_dataset.csv', newline='\n',encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((sanitiseData(row["title"])))
    return data


def sanitiseData(data):

    splitted=data.split(" ")
    removedStopWord = [removeNonAlphabet.sub('', removePossessive(word)).lower() for word in splitted if word.lower() not in stopwords]

    wordSet.update(removedStopWord)
    return removedStopWord

def removePossessive(word):
    word=word.replace("'s", '')
    word=word.replace("’s", '')
    return word

if __name__=="__main__":
    data = simpleDataReader()[:10]
    topicSize = 10
    temp=readStopword()
    # Don't know this one yet
    vocabularySize = "?"
    # Here we need to sanitize our data
    # 1)tokenizen -> unigram/bigram / ...
    # 2)remove stop words of term frequency-inverse document frequency
    # 3)stemming -> no porter stemmer because this one is too agressive (Krovetz Stemmer? ->conclussion for paper below,(also there most likely are no spellig mistakes so we don't need to look at that)
    # Actuammy this papre comes to the conlclussion that no stemming is better most of the time (even uses NYT articles for it)
    # https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf
    documentSize = len(data)
    amountOfWords=len(wordSet)
    docwordList=list()
    wordToMatrixColumnDict=dict()
    initialCorpus=dok_matrix((documentSize,amountOfWords),dtype=np.int)
    currentColumn=0
    for RowIndex, document in enumerate(data):
        for word in document:
            columnIndex=None
            if(word not in wordToMatrixColumnDict):
                wordToMatrixColumnDict[word]=currentColumn
                columnIndex=currentColumn
                currentColumn+=1
            else:
                columnIndex=wordToMatrixColumnDict[word]
            initialCorpus[RowIndex, columnIndex]+=1


    print(len(wordSet))
    # Here we need to implement the actual LDA thing
        # https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2
    # https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158
    tempLDA=LDA([1/10]*topicSize,[1/10000]*amountOfWords,documentSize,amountOfWords)
