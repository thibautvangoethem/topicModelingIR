import csv


def simpleDataReader():
    data=list()
    with open('data/news_dataset.csv', newline='\n',encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((row["title"],row["publication"],row["author"],row["date"],row["year"],row["month"],row["url"],row["content"]))
if __name__=="__main__":
    data = simpleDataReader()
    topicSize = 10
    # Don't know this one yet
    vocabularySize = "?"
    # Here we need to sanitize our data
    # 1)tokenizen -> unigram/bigram / ...
    # 2)remove stop words of term frequency-inverse document frequency
    # 3)stemming -> no porter stemmer because this one is too agressive (Krovetz Stemmer? ->conclussion for paper below,(also there most likely are no spellig mistakes so we don't need to look at that)
    # Actuammy this papre comes to the conlclussion that no stemming is better most of the time (even uses NYT articles for it)
    # https://mimno.infosci.cornell.edu/papers/schofield_tacl_2016.pdf
    documentSize = len(data)
    # Here we need to implement the actual LDA thing
    # https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2
    # https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158
