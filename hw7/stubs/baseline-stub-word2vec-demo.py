#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015

Modified on May 23, 2017 by geetanjali
'''

import sys, nltk, operator
from sklearn.metrics.pairwise import cosine_similarity
from word2vec_extractor import Word2vecExtractor
from dependency_demo_stub import read_dep_parses, find_main


# Read the file from disk
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()
    
    return text

# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    #sentences = [nltk.word_tokenize(sent) for sent in sentences]
    #sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences	

def get_bow(sentence, stopwords):
    return set([word.lower() for word in nltk.word_tokenize(sentence) if word not in stopwords])
	
def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]
	
def baseline(question, sentences, stopwords):
    # Collect all the candidate answers
    qbow = get_bow(get_sentences(question)[0], stopwords)
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        
        answers.append((overlap, sent))
        
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = (answers[0])[1]    
    return best_answer

def baseline_word2vec(question, sentences, stopwords, W2vecextractor):
    q_feat = W2vecextractor.sent2vec(question)
    candidate_answers = []
    
    for sent in sentences:
       a_feat = W2vecextractor.sent2vec(sent)
       dist = cosine_similarity(q_feat, a_feat)[0]
       candidate_answers.append((dist, sent))
       #print("distance: "+str(dist)+"\t sent: "+sent)

    answers = sorted(candidate_answers, key=operator.itemgetter(0), reverse=True)

    print(answers)
    
    best_answer = (answers[0])[1]    
    return best_answer

def baseline_word2vec_verb(question, sentences, stopwords, W2vecextractor, q_verb, sgraphs):
    q_feat = W2vecextractor.word2v(q_verb)
    candidate_answers = []
    print("ROOT of question: "+str(q_verb))
    for i in range(0, len(sentences)):
       sent = sentences[i]
       s_verb = find_main(sgraphs[i])['word']
       print("ROOT of sentence: "+str(s_verb))
       a_feat = W2vecextractor.word2v(s_verb)
       dist = cosine_similarity(q_feat, a_feat)[0]
       candidate_answers.append((dist, sent))
       #print("distance: "+str(dist)+"\t sent: "+sent)

    answers = sorted(candidate_answers, key=operator.itemgetter(0), reverse=True)

    print(answers)
   
    best_answer = (answers[0])[1]    
    return best_answer


if __name__ == '__main__':

    W2vecextractor = Word2vecExtractor()
    text_file = "fables-01.sch"
	
    stopwords = set(nltk.corpus.stopwords.words("english"))
    text = read_file(text_file)
    question = "Where was the crow sitting?"

    qgraphs = read_dep_parses("fables-01.questions.dep")
    qgraph = qgraphs[0]
    q_verb = find_main(qgraph)['word']
    ans_dep_file = "fables-01.sch.dep"
    sgraphs = read_dep_parses(ans_dep_file)

    sentences = get_sentences(text)
	
    #answer = baseline(question, sentences, stopwords)
    answer = baseline_word2vec(question, sentences, stopwords, W2vecextractor)
    #answer = baseline_word2vec_verb(question, sentences, stopwords, W2vecextractor, q_verb, sgraphs)

    print("The answer is: \n"+str(answer))

    #print(" ".join(t[0] for t in answer))
