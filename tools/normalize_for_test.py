import pandas as pd
import numpy as np

import sys, math
from collections import Counter

Usage = "python normalize_for_test.py <train_file.csv> <test_file.csv> <train_out.hd5> <test_out.hd5>"

def normalize_question(q):
    q = q.lower()
    q = q.replace("?","")
    q = q.replace("..."," . ")
    q = q.replace(".."," . ")
    q = q.replace("."," . ")
    q = q.replace(":"," : ")
    q = q.replace(","," , ")
    q = q.replace("/"," / ")
    q = q.replace("("," ( ")
    q = q.replace(")"," ) ")
    q = q.replace("don't", "do not")
    q = q.replace("i'm", "i am")
    q = q.replace("[math]", " [math] ")
    q = q.replace("[/math]", " [/math] ")
    q = q.replace("-"," - ")
    q = q.replace('"',' " ')
    q = q.replace("'s", " 's ")
    words = q.lower().split()
    words = map(lambda x: x.strip(), words)
    words = filter(lambda x: len(x) > 0, words)
    return " ".join(words)

def word_frequencies(questions):
    from collections import Counter
    words = " ".join(questions).split(" ")
    counter = Counter(words)
    return sorted(counter.items(), key=lambda x:-x[1])

train_df = pd.read_csv(sys.argv[1])
print "Train set loaded:", len(train_df)
test_df = pd.read_csv(sys.argv[2])
print "Test set loaded:", len(test_df)

test_questions = set(map(normalize_question, test_df["question1"].tolist() + test_df["question2"].tolist()))
train_questions = set(map(normalize_question, train_df["question1"].tolist() + train_df["question2"].tolist()))

word_frequencies = words_frequencies(test_questions+train_questions)

NWords = 200
most_frequent_words = [w for w, n in word_frequencies[:NWords]]
print "Word frequencies calculated. Most frequent %d words: %s" % (NWords, most_frequent_words)

encoding = {}
for i, w in most_frequent_words:
    row = np.zeros((NWords,), dtype=np.uint8)
    row[i] = 1
    encoding[w] = row




