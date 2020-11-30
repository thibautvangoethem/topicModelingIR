class TermDocumentMatrix():
    def __init__(self):
        self.matrixDict=dict()

    def add(self,word,documentIndex):
        if(word in self.matrixDict):
            innerDict=self.matrixDict[word]
            if(documentIndex in innerDict):
                innerDict[documentIndex]=innerDict[documentIndex]+1
            else:
                innerDict[documentIndex] =1
        else:
            innerDict=dict()
            self.matrixDict[word]=innerDict
            innerDict[documentIndex] = 1