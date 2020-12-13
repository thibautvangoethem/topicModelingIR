import numpy as np
from scipy.sparse import dok_matrix
from GibbsLDA import simpleDataReader, removeCommonAndUniqueWords, wordSet
from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    """
    Print the top n words for each topic in latex table format.
    """
    print("\\begin{table}[h!]\n\\begin{adjustwidth}{-5cm}{-5cm}\n\\begin{center}")
    print("\\begin{tabular}{ |c|m{15cm}| } ")
    print("\\hline")
    print("topic & top 20 words \\\\")
    print("\\hline")
    for topic_idx, topic in enumerate(model.components_):
        print("%d & " % (topic_idx+1), end="")
        words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for word in words:
            print(word, end=", ")
        print("\\\\\n\\hline")
    print("\\end{tabular}")
    print("\\caption{Top 20 words for each topic with %d topics}" % 10)
    print("\\label{top_terms_%d}" % 10)
    print("\\end{center}\n\\end{adjustwidth}\n\\end{table}")


if __name__ == "__main__":
    data = simpleDataReader()
    # Here we need to sanitize our data
    # 1)tokenizen -> unigram/bigram / ...
    # 2)remove stop words of term frequency-inverse document frequency
    # 3)stemming -> no porter stemmer because this one is too agressive
    # (Krovetz Stemmer? ->conclussion for paper below,
    # (also there most likely are no spellig mistakes so we don't need to look at that)
    # Actuammy this paper comes to the conlclussion that no stemming is better most of the time (even uses NYT articles for it)
    # https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf
    print("data read, start removing common words")
    data, removed = removeCommonAndUniqueWords(data)
    wordSet = wordSet - removed
    print("common words removed, start building corpus")
    topicSize = 10
    documentSize = len(data)
    amountOfWords = len(wordSet)
    wordToMatrixColumnDict = dict()
    columnToWord = []
    initialCorpus = dok_matrix((documentSize, amountOfWords), dtype=np.int)
    currentColumn = 0
    # build document term matrix.
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
