import csv
import re
import random
import time
import sys
import math
# http://www.arbylon.net/publications/text-est2.pdf#equation.1.5.78
random.seed(15)
removeNonAlphabet=re.compile('[\W_]+', re.UNICODE)
wordSet=set()

alpha=1/10
beta=1/10000

def readStopword():
    with open('data/stopwords.txt',"r") as stopFile:
        stopwords=stopFile.read().split("\n")
        return stopwords
    return None

stopwords=readStopword()


def simpleDataReader():
    data=list()
    with open('data/news_dataset.csv ', newline='\n',encoding="utf8") as csvfile:
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
    for word in range(round(len(wordlist)*0.01)):
        toremove.append(sorted_words[word][0])
    for document in documents:
        new_document.append(list(set(document).difference(toremove)))
    return new_document

def gibbsLDA(amount_of_topics,document_list):
    # initialisation
    document_topic_count=[0]*len(document_list)
    document_topic_sum=[0]*len(document_list)
    topic_term_count= list()
    for doc in range(amount_of_topics):
        topic_term_count.append(dict())
    topic_term_sum = [0]*amount_of_topics
    document_word_to_topic_matrix=list()
    wordSet=set()
    print("start init")
    for doc_idx,document in enumerate(document_list):
        document_word_to_topic_matrix.append(list())
        document_topic_count[doc_idx]=[0]*amount_of_topics
        for word_idx,word in enumerate(document):
            chosen_topic=random.randint(0,amount_of_topics-1)
            document_topic_count[doc_idx][chosen_topic]+=1
            document_topic_sum[doc_idx]+=1
            if(word in topic_term_count[chosen_topic]):
                topic_term_count[chosen_topic][word]+=1
            else:
                topic_term_count[chosen_topic][word]=1
            topic_term_sum[chosen_topic]+=1
            document_word_to_topic_matrix[doc_idx].append(chosen_topic)
            wordSet.add(word)
    print("starting sampling")
    #gibbs sampling
    finished=False
    # while not finished:
    switched=0
    topic_denuminator_cache=[None]*amount_of_topics
    for doc_idx, document in enumerate(document_list):
        for word_idx,word in enumerate(document):
            word_topic=document_word_to_topic_matrix[doc_idx][word_idx]
            document_topic_count[doc_idx][word_topic] -= 1
            document_topic_sum[doc_idx] -= 1
            topic_term_count[word_topic][word] -= 1
            topic_term_sum[word_topic] -= 1
            if(topic_denuminator_cache[word_topic])!=None:
                topic_denuminator_cache[word_topic]-=1
            #Here comes LDA the formula
            sample_list=list()
            for topic_check in range(amount_of_topics):
                # Sum of denominator
                # this sum starts at an epsilon (extremely small amount) because the chance exist that at a certain point the topic we are looking has no terms
                # If this happens we would first of divide by zero if we initialized on 0
                # Second it is actual logical that when a topic is empty, then our term must be in that topic, because a word in a topic alone is the perfect 100% cohesion topic
                # So by intializing on epsilon the division will give an extremely large number eventually giving this topic the highest chance
                if(topic_denuminator_cache[topic_check])==None:
                    topic_sum = sys.float_info.epsilon
                    for topic_word in topic_term_count[topic_check].keys():
                        topic_sum += topic_term_count[topic_check][topic_word]
                    topic_sum += beta
                    topic_denuminator_cache[topic_check]=topic_sum
                # first fraction
                fraction = (topic_term_count[topic_check].get(word,0) + beta) / topic_denuminator_cache[topic_check]
                val = fraction * (document_topic_count[doc_idx][topic_check] + alpha)
                sample_list.append(val)

            new_topic=sample_list.index(max(sample_list))
            document_word_to_topic_matrix[doc_idx][word_idx]=new_topic
            document_topic_count[doc_idx][new_topic] += 1
            document_topic_sum[doc_idx] += 1
            if(word in topic_term_count[new_topic]):
                topic_term_count[new_topic][word] += 1
            else:
                topic_term_count[new_topic][word] = 1
            topic_term_sum[new_topic] += 1
            if (topic_denuminator_cache[new_topic]) != None:
                topic_denuminator_cache[new_topic] += 1
            if(word_topic!=new_topic):
                switched+=1
        if(doc_idx%10000==0):
            print("sampled %s %% of the documents"%(str((doc_idx/len(document_list))*100)))
    print("switched "+str(switched)+" word topics")
    print("will mixtures")
    #first document topic mixture
    document_topic_mixture=list()
    for doc_idx ,document in enumerate(document_list):
        document_topic_mixture.append([0]*amount_of_topics)
        for topic in range(amount_of_topics):
            denominator=0
            for denominator_topic in range(amount_of_topics):
                denominator+=document_topic_count[doc_idx][denominator_topic]
            denominator+=alpha
            numerator=document_topic_count[doc_idx][topic]+alpha
            document_topic_mixture[doc_idx][topic]=numerator/denominator

    print("highest topic chance per document")
    for idx,doc in enumerate(document_topic_mixture):
        print("doc %s : topic: %s"%(str(idx),str(doc.index(max(doc)))))
    #second term topic mixture
    term_topic_mixture=list()
    for topic in range(amount_of_topics):
        term_topic_mixture.append(dict())
        denominator = 0
        for term_idx, term in enumerate(topic_term_count[topic].keys()):
            denominator+=topic_term_count[topic][term]
        denominator+=beta
        for term_idx,term in enumerate(topic_term_count[topic].keys()):
            term_topic_mixture[topic][term]=(topic_term_count[topic][term]+beta)/denominator
    print("top 20 words per topic")
    for idx,topic in enumerate(term_topic_mixture):
        sorted_topics=sorted(topic.items(), key=lambda x: x[1], reverse=True)
        print("topic %s"%(idx),end=": ")
        for term in sorted_topics[:20]:
            print(term[0] ,end=', ')
        print("")

if __name__=="__main__":
    documents=simpleDataReader()
    documents=removeCommonWords(documents)
    gibbsLDA(10,documents)
    print("test")