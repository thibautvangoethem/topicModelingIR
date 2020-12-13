import csv
import random
import re
import pickle
from time import perf_counter as pc
from operator import itemgetter
# Base implementations gotten from
# http://www.arbylon.net/publications/text-est2.pdf#equation.1.5.78

# Regex expression to remove all non alphabet/numbers characters
removeNonAlphabet=re.compile('[\W_]+', re.UNICODE)
# The wordset V that will contain all terms
wordSet=set()
# Seed is set so we can get reproducible results over multiple runs
random.seed(0)
# Both hyperparameters alpha and beta, these are not arrays in our implementation as we have chosen to let them be equal for each topic/term
alpha = 1/10
beta = 1/10000
# The top percentage of words we cut (1=100%, 0=0%, 0.01=1%)
cut_common_word_percentage = 0.01
# stops the gibs sampling after less than this amount of topics have been changed since last sample
switched_word_cutoff = 0.98

# This list will be later used to take a subsection from and randomly choose an index according to weights
index_list=list()
# Global list used to store preculculations of the beta sum
precalc_beta=list()

def readStopword():
    """
    Reads stopwords from the txt file in the data directory
    :return: a list of stopwords in string formate
    """
    with open('data/stopwords.txt', "r") as stopFile:
        stopwords=stopFile.read().split("\n")
        return stopwords
stopwords=readStopword()

def simpleDataReader():
    """
    Iterates over a dataset and sanitise the data
    :return: a list of documents, which are a list of words
    """
    data=list()
    with open('data/news_dataset.csv', newline='\n', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((sanitiseData(row["content"])))
    return data


def sanitiseData(data):
    """
    Removes stopwords, non alphabetic characters, digits , "'s", also puts everything in lower case
    Ends up returing the input split on whitespace
    :param data: a string containg words
    :return: a list of string
    """
    splitted=data.split(" ")
    removedStopWord = [removeNonAlphabet.sub('', removePossessive(word)).lower()
                       for word in splitted if word.lower()
                       not in stopwords and word != "" and len(word) > 2 and not any(i.isdigit() for i in word)]

    wordSet.update(removedStopWord)
    return removedStopWord


def removePossessive(word):
    '''
    Removes possessive 's
    :param word: a string
    :return: a string without possessive
    '''
    word=word.replace("'s", '')
    word=word.replace("’s", '')
    return word

wordlist=dict()


def removeCommonAndUniqueWords(documents):
    """
    Goes through the entire document and removes words based on different criteria:
    - Remove all words that only appear in less then 10 document
    - Remove all words that are in the top 1% used
    - Remove all words that are used in more than 40% of the documents
    :param documents: a list of documents
    :return: a list of documents, with certain words removed
    """
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
        elif frequentie >= total_documents * 0.40:
            toremove.add(word)
            unique_words += 1
    print("removed %d unique and common words" % unique_words)

    # remove top 1 percent of words
    new_document = list()
    for word in range(round(len(wordlist) * cut_common_word_percentage)):
        toremove.add(sorted_words[word][0])
    for document in documents:
        new_document.append([word for word in document if word not in toremove])

    return new_document, toremove

def gibbsLDA(amount_of_topics,document_list):
    """
    The base function of the LDA algorithm
    :param amount_of_topics: the K, or amount of topics that will be chosen
    :param document_list: a list of documents, which are a list of words
    :return: nothing, everything will be printed, and save in pkl files in the obj directory
    """
    # initialisation
    alpha_sum = alpha * amount_of_topics
    for i in range(amount_of_topics):
        index_list.append(i)
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
    for topic in range(amount_of_topics):
        precalc_beta.append(len(topic_term_count)*beta)
    #gibbs sampling
    finished=False
    prev_switched=0
    switched=0
    while prev_switched==0 or switched/prev_switched<switched_word_cutoff:
        prev_switched=switched
        switched = corpusGibbsSamplePass(amount_of_topics, document_list, document_topic_count, document_topic_sum,
                                     document_word_to_topic_matrix, topic_term_count, topic_term_sum,alpha_sum)
        print("switched "+str(switched)+" word topics")
        if(switched==0):
            break
    print("will now calculate mixtures")
    # first document topic mixture
    document_topic_mixture = calculateDocumentTopixMixture(amount_of_topics, document_list, document_topic_count)

    with open('obj/document_topic_mixture_topics.pkl', 'wb') as file:  # save document_topic_mixture to a file
        pickle.dump(document_topic_mixture, file)

    print("highest document chance per topic ")
    topic_chances_list=list()
    for topic in range(amount_of_topics):
        topic_chances_list.append(list())
    for idx, doc in enumerate(document_topic_mixture):
        for topic in range(amount_of_topics):
            topic_chances_list[topic].append((idx,document_topic_mixture[idx][topic]))
    for topic in range(amount_of_topics):
        sorted_list=sorted(topic_chances_list[topic],key=itemgetter(1), reverse=True)
        print("topic %s:"%(str(topic)),end='')
        for i in sorted_list[:20]:
            print(i[0],end=' ')
        print(" ")
        # print("doc %s : topic: %s"%(str(idx),str(doc.index(max(doc)))))

    term_topic_mixture = calculateTermTopicMixture(amount_of_topics, topic_term_count,topic_term_sum)

    with open('obj/term_topic_mixture_topics.pkl', 'wb') as file:  # save document_topic_mixture to a file
        pickle.dump(term_topic_mixture, file)

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
                          document_word_to_topic_matrix, topic_term_count, topic_term_sum,alpha_sum):
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
    t0 = pc()
    for doc_idx, document in enumerate(document_list):
        switched = gibbsSampleForWordOfDocument(amount_of_topics, doc_idx, document, document_topic_count,
                                                document_topic_sum, document_word_to_topic_matrix, switched,
                                                topic_term_count, topic_term_sum,alpha_sum)
        if doc_idx % 10000 == 0:
            print("sampled %s %% of the documents" % (str((doc_idx / len(document_list)) * 100)))
    print(str(pc()-t0)+"s")
    return switched

def gibbsSampleForWordOfDocument(amount_of_topics, doc_idx, document, document_topic_count, document_topic_sum,
                                 document_word_to_topic_matrix, switched, topic_term_count,
                                 topic_term_sum,alpha_sum):
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

        new_topic = sampleNewTopicForWords(amount_of_topics, doc_idx, document_topic_count,document_topic_sum,
                                           topic_term_count,topic_term_sum, word,alpha_sum)[0]

        document_word_to_topic_matrix[doc_idx][word_idx] = new_topic
        document_topic_count[doc_idx][new_topic] += 1
        document_topic_sum[doc_idx] += 1
        if (word in topic_term_count[new_topic]):
            topic_term_count[new_topic][word] += 1
        else:
            topic_term_count[new_topic][word] = 1
            precalc_beta[new_topic] += beta
        topic_term_sum[new_topic] += 1
        if (word_topic != new_topic):
            switched += 1
        if(topic_term_count[word_topic][word] == 0):
            del(topic_term_count[word_topic][word])
            precalc_beta[word_topic]-=beta
    return switched

def sampleNewTopicForWords(amount_of_topics, doc_idx, document_topic_count,document_topic_sum,
                           topic_term_count,topic_term_sum, word,alpha_sum):
    """
    Uses the dirichlet distribtion to sample a new topic
    For efficiency we do not constantly recalculate the topic denominators, but chance them according to the removed/added word to a topic
    So this full calculation is only done on the first term of a document
    :return the best topic for this word in this document as determined by the dirichlet distribution
    """
    sample_list = list()
    val = 0
    for topic_check in range(amount_of_topics):
        first_fraction=(document_topic_count[doc_idx][topic_check]+alpha)/(document_topic_sum[doc_idx]+alpha_sum)
        # second fraction
        second_fraction=0
        if(word in topic_term_count[topic_check]):
            second_fraction=(topic_term_count[topic_check][word]+beta)/(topic_term_sum[topic_check]+precalc_beta[topic_check])
        else:
            second_fraction = (beta) / (
                        topic_term_sum[topic_check] + precalc_beta[topic_check])
        val += first_fraction*second_fraction
        sample_list.append(val)
    # normalised_sample_list = [float(i) / sum(sample_list) for i in sample_list]
    return random.choices(index_list, cum_weights=sample_list)


def calculateTermTopicMixture(amount_of_topics, topic_term_count,topic_term_sum):
    """
    calculate the term topic mixture according to http://www.arbylon.net/publications/text-est2.pdf#equation.1.5.78 formula 81
    :return: the term topic mixture matrix (K*V) V is here actually a dict containing words to probabilities, this is done to save memory
    """
    term_topic_mixture = list()
    for topic in range(amount_of_topics):
        term_topic_mixture.append(dict())
        dminator=topic_term_sum[topic]+amount_of_topics*beta
        for term_idx, term in enumerate(topic_term_count[topic].keys()):
            # term_topic_mixture[topic][term] = (topic_term_count[topic][term] + beta) / denominator
            term_topic_mixture[topic][term] = topic_term_count[topic][term]
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
                denominator += (document_topic_count[doc_idx][denominator_topic]+alpha)
            numerator = document_topic_count[doc_idx][topic] + alpha
            document_topic_mixture[doc_idx][topic] = numerator / denominator
    return document_topic_mixture


if __name__ == "__main__":
    # get current time
    t0 = pc()
    # read data and sanitise to
    documents = simpleDataReader()
    # Remove certain non usefull words
    documents, removed = removeCommonAndUniqueWords(documents)
    # save pre-processed documents to file to be used by loadLDA.py
    with open('obj/documents.pkl', 'wb') as file:
        pickle.dump(documents, file)
    wordSet = wordSet - removed
    # Define amount of topics
    amount_of_topics = 50
    # Set hyperparameters to something more useful
    alpha = 1/amount_of_topics
    beta = 1/len(wordSet)

    # Run the actual algorithm
    gibbsLDA(amount_of_topics, documents)
    # Print time spent
    print(str(pc() - t0)+"s")
