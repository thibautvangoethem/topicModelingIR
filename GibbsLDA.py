import csv
import re
import random
import time
import sys
import math
# http://www.arbylon.net/publications/text-est2.pdf#equation.1.5.78
removeNonAlphabet=re.compile('[\W_]+', re.UNICODE)
wordSet=set()

alpha=1/10
beta=1/10000
cut_common_word_percentage=0.10
# stops the gibs sampling after less than this amount is sampled
switched_word_cutoff=1000

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
    toremove=set()
    # remove top 1 percent of words
    new_document=list()
    for word in range(round(len(wordlist)*cut_common_word_percentage)):
        toremove.add(sorted_words[word][0])
    for document in documents:
        new_document.append([word for word in document if word not in toremove])
    return new_document

def gibbsLDA(amount_of_topics,document_list):
    # initialisation
    # Create empty count lists
    document_topic_count=[0]*len(document_list)
    document_topic_sum=[0]*len(document_list)
    topic_term_count= list()
    for doc in range(amount_of_topics):
        topic_term_count.append(dict())
    topic_term_sum = [0]*amount_of_topics
    document_word_to_topic_matrix=list()
    print("start init")
    initializeCountVariables(amount_of_topics, document_list, document_topic_count, document_topic_sum,
                             document_word_to_topic_matrix, topic_term_count, topic_term_sum)
    print("starting sampling")
    #gibbs sampling
    finished=False
    switched=100000
    while switched > switched_word_cutoff:
        switched = corpusGibbsSamplePass(amount_of_topics, document_list, document_topic_count, document_topic_sum,
                                     document_word_to_topic_matrix, topic_term_count, topic_term_sum)
        print("switched "+str(switched)+" word topics")
    print("will now calculate mixtures")
    # first document topic mixture
    document_topic_mixture = calculateDocumentTopixMixture(amount_of_topics, document_list, document_topic_count)

    print("highest topic chance per document")
    for idx, doc in enumerate(document_topic_mixture):
        print("doc %s : topic: %s"%(str(idx),str(doc.index(max(doc)))))
    # second term topic mixture
    term_topic_mixture = calculateTermTopicMixture(amount_of_topics, topic_term_count)
    print("top 20 words per topic")
    for idx, topic in enumerate(term_topic_mixture):
        sorted_topics=sorted(topic.items(), key=lambda x: x[1], reverse=True)
        print("topic %s" % idx, end=": ")
        for term in sorted_topics[:20]:
            print(term[0], end=', ')
        print("")

def initializeCountVariables(amount_of_topics, document_list, document_topic_count, document_topic_sum,
                             document_word_to_topic_matrix, topic_term_count, topic_term_sum):
    """
    Here we start the intialization of the five main variables: document_topic_count, document_topic_sum, topic_term_count, topic_term_sum and document_word_to_topic_matrix
    This function simply iterates over all documents and calls the init document function
    :param amount_of_topics: Variable K denoting the amount of topics we will sample
    :param document_list: The entire list of documents, each document is a list of terms
    :param document_topic_count: the count of how many words in a document have a certain documet = M*K matrix (M=amount of doc, K=amount of topics)
    :param document_topic_sum: The amount of words in topics a document has
    :param document_word_to_topic_matrix: A list of dictionaries, denoting which term has which topic per document
    :param topic_term_count: The amount of times a term is in a topic, an K*V matrix (k=amount of topics, V= amount of terms)
    :param topic_term_sum: The total amount of terms per topic (K size vector)
    :return:all data is in the variables
    """
    for doc_idx, document in enumerate(document_list):
        addDocumentToInitialization(amount_of_topics, doc_idx, document, document_topic_count, document_topic_sum,
                                    document_word_to_topic_matrix, topic_term_count, topic_term_sum)


def addDocumentToInitialization(amount_of_topics, doc_idx, document, document_topic_count, document_topic_sum,
                                document_word_to_topic_matrix, topic_term_count, topic_term_sum):
    """
    This function iterates over all words in a document and calls the countword for init function
    Variables are similar to 'initializeCountVariables'
    """
    document_word_to_topic_matrix.append(list())
    document_topic_count[doc_idx] = [0] * amount_of_topics
    for word_idx, word in enumerate(document):
        countWordForInitialization(amount_of_topics, doc_idx, document_topic_count, document_topic_sum,
                                   document_word_to_topic_matrix, topic_term_count, topic_term_sum, word)


def countWordForInitialization(amount_of_topics, doc_idx, document_topic_count, document_topic_sum,
                               document_word_to_topic_matrix, topic_term_count, topic_term_sum, word):
    """
    Counts a single word/term in a document the correct way for initialization
    For meaning of the variables look at "initializeCountVariables"
    """
    chosen_topic = random.randint(0, amount_of_topics - 1)
    document_topic_count[doc_idx][chosen_topic] += 1
    document_topic_sum[doc_idx] += 1
    if (word in topic_term_count[chosen_topic]):
        topic_term_count[chosen_topic][word] += 1
    else:
        topic_term_count[chosen_topic][word] = 1
    topic_term_sum[chosen_topic] += 1
    document_word_to_topic_matrix[doc_idx].append(chosen_topic)


def corpusGibbsSamplePass(amount_of_topics, document_list, document_topic_count, document_topic_sum,
                          document_word_to_topic_matrix, topic_term_count, topic_term_sum):
    """
    Does one entire pass over the entire corpus and uses the gibbs sampling process on every word of a document
    :param amount_of_topics: variable K
    :param document_list: the entire list of documents, with a documetn being a list of terms/words
    :param document_list: The entire list of documents, each document is a list of terms
    :param document_topic_count: the count of how many words in a document have a certain documet = M*K matrix (M=amount of doc, K=amount of topics)
    :param document_topic_sum: The amount of words in topics a document has
    :param document_word_to_topic_matrix: A list of dictionaries, denoting which term has which topic per document
    :param topic_term_count: The amount of times a term is in a topic, an K*V matrix (k=amount of topics, V= amount of terms)
    :param topic_term_sum: The total amount of terms per topic (K size vector)
    :return: The amount of words whos topic got switched, is used to determine whether the gibbs sampling is converging
    """
    switched = 0
    topic_denuminator_cache = [None] * amount_of_topics
    for doc_idx, document in enumerate(document_list):
        switched = gibbsSampleForWordOfDocument(amount_of_topics, doc_idx, document, document_topic_count,
                                                document_topic_sum, document_word_to_topic_matrix, switched,                                                topic_denuminator_cache, topic_term_count, topic_term_sum)
        if doc_idx % 10000 == 0:
            print("sampled %s %% of the documents" % (str((doc_idx / len(document_list)) * 100)))
    return switched


def gibbsSampleForWordOfDocument(amount_of_topics, doc_idx, document, document_topic_count, document_topic_sum,
                                 document_word_to_topic_matrix, switched, topic_denuminator_cache, topic_term_count,
                                 topic_term_sum):
    """
    Does the gibbs sampling process for a single word in a document
    This function consists of first removing this word from the current counters,
    Then sampling a new topic through the dirichlet distribution
    Then counting this new topic for this word in the corect lists
    For the variable meanings look at "corpusGibbsSamplePass"
    """
    for word_idx, word in enumerate(document):
        word_topic = document_word_to_topic_matrix[doc_idx][word_idx]
        document_topic_count[doc_idx][word_topic] -= 1
        document_topic_sum[doc_idx] -= 1
        topic_term_count[word_topic][word] -= 1
        topic_term_sum[word_topic] -= 1
        if (topic_denuminator_cache[word_topic]) != None:
            topic_denuminator_cache[word_topic] -= 1
        # Here comes LDA the formula

        new_topic = sampleNewTopicForWords(amount_of_topics, doc_idx, document_topic_count, topic_denuminator_cache,
                                           topic_term_count, word)

        document_word_to_topic_matrix[doc_idx][word_idx] = new_topic
        document_topic_count[doc_idx][new_topic] += 1
        document_topic_sum[doc_idx] += 1
        if (word in topic_term_count[new_topic]):
            topic_term_count[new_topic][word] += 1
        else:
            topic_term_count[new_topic][word] = 1
        topic_term_sum[new_topic] += 1
        if (topic_denuminator_cache[new_topic]) != None:
            topic_denuminator_cache[new_topic] += 1
        if (word_topic != new_topic):
            switched += 1
        if(topic_term_count[word_topic][word] == 0):
            del(topic_term_count[word_topic][word])
    return switched


def sampleNewTopicForWords(amount_of_topics, doc_idx, document_topic_count, topic_denuminator_cache,
                           topic_term_count, word):
    """
    Uses the dirichlet distribtion to sample a new topic
    For efficiency we do not constantly recalculate the topic denominators, but chance them according to the removed/added word to a topic
    So this full calculation is only done on the first term of a document
    :return the best topic for this word in this document as determined by the dirichlet distribution
    """
    sample_list = list()
    for topic_check in range(amount_of_topics):
        # Sum of denominator
        # this sum starts at an epsilon (extremely small amount) because the chance exist that at a certain point the topic we are looking has no terms
        # If this happens we would first of divide by zero if we initialized on 0
        # Second it is actual logical that when a topic is empty, then our term must be in that topic, because a word in a topic alone is the perfect 100% cohesion topic
        # So by intializing on epsilon the division will give an extremely large number eventually giving this topic the highest chance
        if (topic_denuminator_cache[topic_check]) == None:
            topic_sum = sys.float_info.epsilon
            for topic_word in topic_term_count[topic_check].keys():
                topic_sum += topic_term_count[topic_check][topic_word]
            topic_sum += beta
            topic_denuminator_cache[topic_check] = topic_sum
        # first fraction
        fraction = (topic_term_count[topic_check].get(word, 0) + beta) / topic_denuminator_cache[topic_check]
        val = fraction * (document_topic_count[doc_idx][topic_check] + alpha)
        sample_list.append(val)
    return sample_list.index(max(sample_list))

def calculateTermTopicMixture(amount_of_topics, topic_term_count):
    """
    calculate the term topic mixture according to http://www.arbylon.net/publications/text-est2.pdf#equation.1.5.78 formula 81
    :return: the term topic mixutre matrix (K*V) V is here actually a dict containing words to probabilities, this is doen to save memory
    """
    term_topic_mixture = list()
    for topic in range(amount_of_topics):
        term_topic_mixture.append(dict())
        denominator = 0
        for term_idx, term in enumerate(topic_term_count[topic].keys()):
            denominator += topic_term_count[topic][term]
        denominator += beta
        for term_idx, term in enumerate(topic_term_count[topic].keys()):
            term_topic_mixture[topic][term] = (topic_term_count[topic][term] + beta) / denominator
    return term_topic_mixture


def calculateDocumentTopixMixture(amount_of_topics, document_list, document_topic_count):
    """
    calculate the Document topic mixture according to http://www.arbylon.net/publications/text-est2.pdf#equation.1.5.78 formula 82
    :return: the document topic mixture matrix (M*K)
    """
    document_topic_mixture = list()
    for doc_idx, document in enumerate(document_list):
        document_topic_mixture.append([0] * amount_of_topics)
        for topic in range(amount_of_topics):
            denominator = 0
            for denominator_topic in range(amount_of_topics):
                denominator += document_topic_count[doc_idx][denominator_topic]
            denominator += alpha
            numerator = document_topic_count[doc_idx][topic] + alpha
            document_topic_mixture[doc_idx][topic] = numerator / denominator
    return document_topic_mixture


if __name__ == "__main__":
    documents = simpleDataReader()
    documents = removeCommonWords(documents)
    gibbsLDA(20, documents)