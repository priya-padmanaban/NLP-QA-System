import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import gensim
import os

stops = set(stopwords.words("english"))
punct = set(string.punctuation)

w2v = os.path.join("GoogleNews-vectors-negative300.bin")

class Word2vecExtractor:

    def __init__(self, w2vecmodel = w2v):
        self.w2vecmodel=gensim.models.Word2Vec.load_word2vec_format(w2vecmodel, binary=True)
    
    def sent2vec(self,sentence):    
        words = [word.lower() for word in nltk.word_tokenize(sentence) if word not in stops and word not in punct]
        res = np.zeros(self.w2vecmodel.vector_size)
        count = 0
        for word in words:
            if word in self.w2vecmodel:
                count += 1
                res += self.w2vecmodel[word]

        if count != 0:
            res /= count

        return res 

    def doc2vec(self, doc):
        count = 0    
        res = np.zeros(self.w2vecmodel.vector_size)
        for sentence in nltk.sent_tokenize(doc):
            for word in nltk.word_tokenize(sentence):
                if((word not in stops) and (word not in punct)):
                    if word in self.w2vecmodel:
                        count += 1
                        res += self.w2vecmodel[word]

        if count != 0:
            res /= count

        return res 
  
    def word2v(self, word):
        res = np.zeros(self.w2vecmodel.vector_size)
        if word in self.w2vecmodel:
            res += self.w2vecmodel[word]
        return res

if __name__ == "__main__":

        W2vecextractor = Word2vecExtractor(w2vecmodel)
 
        sentence = "A fisherman was catching fish by the sea."

        features = W2vecextractor.sent2vec(sentence)
        











	
        
