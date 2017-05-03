import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import math, random

MaxQuestionLength = 40   # words
MaxMissingWords = 2*MaxQuestionLength

def normalize_question(q):
    q = q.lower()
    q = q.replace("?"," ? ")
    q = q.replace("..."," . ")
    q = q.replace(".."," . ")
    q = q.replace("."," . ")
    q = q.replace(":"," : ")
    q = q.replace(","," , ")
    q = q.replace("/"," / ")
    q = q.replace("("," ( ")
    q = q.replace(")"," ) ")
    q = q.replace("n't"," not ")    
    q = q.replace("i'm", "i am")
    q = q.replace("[math]", " [math] ")
    q = q.replace("[/math]", " [/math] ")
    q = q.replace("-"," - ")
    q = q.replace('"',' " ')
    q = q.replace('\xe2\x80\x9c', ' " ').replace('\xe2\x80\x9d', ' " ')    
    q = q.replace("'s", " 's ")
    words = q.lower().split()[:MaxQuestionLength]
    #words = map(lambda x: x.strip(), words)
    #words = filter(lambda x: len(x) > 0, words)
    #words = map(lambda w: w[:-1] + " s" if len(w)>3 and w[-1]=='s' and 
    #            not w in ("this", "does") else w, words)
    return " ".join(words)

def word_frequencies(questions):
    words = " ".join(questions).split(" ")
    counter = Counter(words)
    return sorted(counter.items(), key=lambda x:-x[1])

def encode_question(q):
    # question is already normalized
    words = q.split()
    wset = set(words)
    missing_words = wset - vocabulary_set
    missing_words_encoding = {w:VocabularySize+1+random.randint(1,MaxMissingWords-1) 
                              for i, w in enumerate(missing_words)}
    encoded = np.array([vocabulary_encoding[w] 
            if w in vocabulary_set
            else missing_words_encoding[w]           
            for w in words], 
        dtype=np.uint16)
    return encoded

def decode_question(words):
    return " ".join(map(lambda x: vocabulary_decoding[x] if x <= VocabularySize else "<%d>" % (x,), words))

def permute_encoded_question(q):
    if len(q) <= 0: return q
    q = q.copy()
    n = len(q)
    m = max(1, n/3)
    r = range(n)
    for i in random.sample(r, m):
        q[i] = random.randint(1, VocabularySize)
    for _ in xrange(n/10):
        i,j = random.sample(r, 2)
        q[i], q[j] = q[j], q[i]
    return q

