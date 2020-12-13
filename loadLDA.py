import pickle
import csv
from operator import itemgetter
# This file loads in the result from GibbsLDA.py and generates a csv file


def createCSV(document_topic_mixture, blackList, nb_documents=100):
    """
    Creates a csv file as specified in the assigment.
    """
    amount_of_topics = len(document_topic_mixture[0])
    print("highest document chance per topic ")
    topic_chances_list = list()
    for topic in range(amount_of_topics):
        topic_chances_list.append(list())
    for idx, doc in enumerate(document_topic_mixture):
        for topic in range(amount_of_topics):
            # after preprocessing there are a few documents that do not contain any words.
            # These document are listed in the blackList variable and generaly contain bad results.
            # There for they should not be present in the csv file
            if idx not in blackList:
                topic_chances_list[topic].append((idx, document_topic_mixture[idx][topic]))
    for topic in range(amount_of_topics):
        sorted_list = sorted(topic_chances_list[topic], key=itemgetter(1), reverse=True)
        topic_chances_list[topic] = sorted_list[:nb_documents]
    with open('top_100_documents_per_topic.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        for i in range(100):
            row = []
            for topic in topic_chances_list:
                row.append(topic[i][0])
            writer.writerow(row)


if __name__ == "__main__":
    # load the results of GibbsLDA.py
    with open('obj/document_topic_mixture_topics_10_topics_improved.pkl', 'rb') as file:
        document_topic_mixture = pickle.load(file)

    with open('obj/term_topic_mixture_topics_10_topics_improved.pkl', 'rb') as file:
        term_topic_mixture = pickle.load(file)

    # load in all pre-processed documents
    with open('obj/documents.pkl', 'rb') as file:
        documents = pickle.load(file)

    # find all documents that should be blacklisted
    blackList = set()
    for idx, doc in enumerate(documents):
        if len(doc) <= 5:
            blackList.add(idx)

    print("top 20 words per topic")
    for idx, topic in enumerate(term_topic_mixture):
        sorted_topics = sorted(topic.items(), key=lambda x: x[1], reverse=True)
        print("topic %s" % idx, end=": ")
        for term in sorted_topics[:20]:
            print(term[0], end=', ')
        print("")
    createCSV(document_topic_mixture, blackList)
