import pandas as pd
import numpy as np

import sys, math, random
from collections import Counter

Usage = "python normalize_for_test.py <train_file.csv> <test_file.csv> <train_out.hd5> <test_out.hd5>"

NFrequent = 2000
MaxInfrequentWords = 25

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
    q = q.replace("n't"," not ")    
    q = q.replace("i'm", "i am")
    q = q.replace("[math]", " [math] ")
    q = q.replace("[/math]", " [/math] ")
    q = q.replace("-"," - ")
    q = q.replace('"',' " ')
    q = q.replace("'s", " 's ")
    words = q.lower().split()
    #words = map(lambda x: x.strip(), words)
    #words = filter(lambda x: len(x) > 0, words)
    words = map(lambda w: w[:-1] + " s" if len(w)>3 and w[-1]=='s' and w != "this" else w, words)
    return " ".join(words)

def word_frequencies(questions):
    words = " ".join(questions).split(" ")
    counter = Counter(words)
    return sorted(counter.items(), key=lambda x:-x[1])

train_df=pd.read_csv(sys.argv[1])
train_df.fillna("", inplace=True)
print "Train set loaded:", len(train_df)

test_df = pd.read_csv(sys.argv[2])
test_df.fillna("", inplace=True)
print "Test set loaded:", len(test_df)

test_questions = map(normalize_question, test_df["question1"].tolist() + test_df["question2"].tolist())
train_questions = map(normalize_question, train_df["question1"].tolist() + train_df["question2"].tolist())

all_questions = set(test_questions+train_questions)
all_frequencies = word_frequencies(all_questions)
#all_words = [w for w, n in all_frequencies]

VocabularySize = 6000

encoding = { w:i+1 for i, (w, n) in enumerate(all_frequencies[:VocabularySize]) }

MaxLen = 100

def encode_pair(q1, q2):
    words1 = [encoding.get(w, VocabularySize-1) for w in normalize_question(q1).split()[:MaxLen]]
    words2 = [encoding.get(w, VocabularySize-1) for w in normalize_question(q2).split()[:MaxLen]]
    l1 = len(words1)
    l2 = len(words2)
    l = max(l1, l2)
    n1 = l-l1
    n2 = l-l2
    words1 = np.array([0]*n1 + words1, dtype=np.int32)
    words2 = np.array([0]*n2 + words2, dtype=np.int32)
    return words1, words2
    
def encode_question(q):
    return np.array([encoding.get(w, VocabularySize-1) for w in normalize_question(q).split()[:MaxLen]], dtype=np.int32)
    
def permute_encoded_question(q, vocabulary_size):
    if len(q) <= 0: return q
    q = q.copy()
    if random.random() < 0.5:
        random.shuffle(q)
    else:
        n = len(q)
        m = max(1, n/5)
        for i in random.sample(range(n), m):
            q[i] = random.randint(0, vocabulary_size-1)
    return q
        
pairs_encoded = [
    encode_pair(q1, q2)+(is_dup,)
    for i, id, qid1, qid2, q1, q2, is_dup in train_df.itertuples()
]

questions_encoded = [
    (encode_question(q),1) for q in all_questions
]

questions_encoded = questions_encoded + [(permute_encoded_question(q, VocabularySize), 0) for q, v in questions_encoded]
random.shuffle(questions_encoded)

pairs_encoded = pd.DataFrame(pairs_encoded, columns=["q1","q2","dup"])
train_store = pd.HDFStore(sys.argv[3])
train_store["pairs"] = pairs_encoded
train_store["questions"] = pd.DataFrame(questions_encoded, columns=["q", "valid"])
train_store["original"] = train_df[["question1","question2","is_duplicate"]]

words = pd.Series([w for (w, n) in enumerate(all_frequencies[:VocabularySize])])
encoding_df = pd.DataFrame(index=range(1, VocabularySize+1))
encoding_df["word"] = words
train_store["encoding"] = encoding_df
train_store.close()

print "encoded dataframe is ready"