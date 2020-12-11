import pickle

with open('obj/document_topic_mixture.pkl', 'rb') as file:
    document_topic_mixture = pickle.load(file)

with open('obj/term_topic_mixture.pkl', 'rb') as file:
    term_topic_mixture = pickle.load(file)

print("highest topic chance per document")
for idx, doc in enumerate(document_topic_mixture):
    print("doc %s : topic: %s"%(str(idx),str(doc.index(max(doc)))))

print("top 20 words per topic")
for idx, topic in enumerate(term_topic_mixture):
    sorted_topics=sorted(topic.items(), key=lambda x: x[1], reverse=True)
    print("topic %s" % idx, end=": ")
    for term in sorted_topics[:20]:
        print(term[0], end=', ')
    print("")