import pickle
import csv
from operator import itemgetter


def createCSV(document_topic_mixture, blackList, nb_documents=100):
    amount_of_topics = len(document_topic_mixture[0])
    print("highest document chance per topic ")
    topic_chances_list = list()
    for topic in range(amount_of_topics):
        topic_chances_list.append(list())
    for idx, doc in enumerate(document_topic_mixture):
        for topic in range(amount_of_topics):
            if idx not in blackList:
                topic_chances_list[topic].append((idx, document_topic_mixture[idx][topic]))
    for topic in range(amount_of_topics):
        sorted_list = sorted(topic_chances_list[topic], key=itemgetter(1), reverse=True)
        print("topic %s:" % (str(topic)), end='')
        for i in sorted_list[:20]:
            print(i[0], end=' ')
        print(" ")
        topic_chances_list[topic] = sorted_list[:nb_documents]
    with open('top_100_documents_per_topic.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        for i in range(100):
            row = []
            for topic in topic_chances_list:
                row.append(topic[i][0])
            writer.writerow(row)


if __name__ == "__main__":

    with open('obj/document_topic_mixture_topics_10_topics_improved.pkl', 'rb') as file:
        document_topic_mixture = pickle.load(file)

    with open('obj/term_topic_mixture_topics_10_topics_improved.pkl', 'rb') as file:
        term_topic_mixture = pickle.load(file)

    with open('obj/documents.pkl', 'rb') as file:
        documents = pickle.load(file)

    blackList = set()
    for idx, doc in enumerate(documents):
        if len(doc) <= 5:
            blackList.add(idx)

    #print("highest topic chance per document")
    #for idx, doc in enumerate(document_topic_mixture):
    #    print("doc %s : topic: %s"%(str(idx),str(doc.index(max(doc)))))

    print("top 20 words per topic")
    for idx, topic in enumerate(term_topic_mixture):
        sorted_topics=sorted(topic.items(), key=lambda x: x[1], reverse=True)
        print("topic %s" % idx, end=": ")
        for term in sorted_topics[:20]:
            print(term[0], end=', ')
        print("")
    #createCSV(document_topic_mixture, blackList)
