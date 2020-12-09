import numpy as np
from bayespy import nodes

from bayespy.inference.vmp.nodes.categorical import CategoricalMoments

from LDABayes import read_data

corpus, word_documents, indexToWord, n_documents, n_vocabulary = read_data()
print("data is read, starting training")
n_words = len(corpus)
print("total words: ", n_words)
print("total documents: ", n_documents)
n_topics = 10
subset_size = 1500
plates_multiplier = n_words / subset_size

p_topic = nodes.Dirichlet(np.ones(n_topics), plates=(n_documents,), name='p_topic')
print("created p_topic")
p_word = nodes.Dirichlet(np.ones(n_vocabulary), plates=(n_topics,), name='p_word')
print("created p_word")
document_indices = nodes.Constant(CategoricalMoments(n_documents), word_documents[:subset_size], name='document_indices')
print("created document_indices")
topics = nodes.Categorical(nodes.Gate(document_indices, p_topic),
                           plates=(subset_size,), plates_multiplier=(plates_multiplier,), name='topics')
print("created topics")
words = nodes.Categorical(nodes.Gate(topics, p_word), name='words')
print("created words")
p_topic.load("p_topics.hdf5")
p_word.load("p_word.hdf5")
topics.load("topics.hdf5")
words.load("words.hdf5")

word_topic_distribution = p_word.get_moments()[0]
total_topics, total_words = word_topic_distribution.shape
nb_best_words = 10
best_words_per_topic = []
for i in range(total_topics):
    current_best = [1000] * nb_best_words
    current_words = [None] * nb_best_words
    current_min = max(current_best)
    current_min_index = current_best.index(current_min)
    for j in range(total_words):
        if word_topic_distribution[i][j] < current_min:
            current_best[current_min_index] = word_topic_distribution[i][j]
            current_words[current_min_index] = indexToWord[j]

            current_min = max(current_best)
            current_min_index = current_best.index(current_min)
    best_words_per_topic.append(current_words)

for i in best_words_per_topic:
    print(i)

print("test")