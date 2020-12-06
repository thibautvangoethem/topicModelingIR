import numpy as np
from bayespy import nodes
from bayespy.inference.vmp.nodes.categorical import CategoricalMoments
from bayespy.inference import VB
import bayespy.plot as bpplt
if __name__=="__main__":
    n_documents=100
    n_words=10000
    n_vocabulary = 5000
    n_topics = 10
    word_documents = nodes.Categorical(np.ones(n_documents) / n_documents,plates = (n_words,)).random()
    p_topic = nodes.Dirichlet(1e-1 * np.ones(n_topics),plates = (n_documents,)).random()
    p_word = nodes.Dirichlet(1e-1 * np.ones(n_vocabulary),plates = (n_topics,)).random()
    topic = nodes.Categorical(p_topic[word_documents],plates = (n_words,)).random()
    corpus = nodes.Categorical(p_word[topic],plates = (n_words,)).random()

    p_topic = nodes.Dirichlet(np.ones(n_topics),plates = (n_documents,),name = 'p_topic')
    p_word = nodes.Dirichlet(np.ones(n_vocabulary),plates = (n_topics,),name = 'p_word')
    document_indices = nodes.Constant(CategoricalMoments(n_documents), word_documents,name = 'document_indices')
    topics = nodes.Categorical(nodes.Gate(document_indices, p_topic),plates = (len(corpus),),name = 'topics')
    words = nodes.Categorical(nodes.Gate(topics, p_word),name = 'words')

    words.observe(corpus)
    p_topic.initialize_from_random()
    p_word.initialize_from_random()
    Q = VB(words, topics, p_word, p_topic, document_indices)
    Q.update(repeat=10)
    test=bpplt.pyplot.figure()
    bpplt.hinton(Q['p_topic'])
    bpplt.pyplot.title("Posterior topic distribution for each document")
    bpplt.pyplot.xlabel("Topics")
    bpplt.pyplot.ylabel("Documents")
    bpplt.pyplot.plot(Q.L)
    test.savefig('my_figure.png')
    print("test")