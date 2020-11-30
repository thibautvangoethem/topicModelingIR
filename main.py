import csv
from TermDocumentMatrix import TermDocumentMatrix
def simpleDataReader():
    data=list()
    with open('data/news_dataset.csv', newline='\n',encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((row["title"],row["publication"],row["author"],row["date"],row["year"],row["month"],row["url"],row["content"]))
if __name__=="__main__":
    data=simpleDataReader()
    topicSize=10
    # Don't know this one yet
    vocabularySize="?"
    documentSize=len(data)
