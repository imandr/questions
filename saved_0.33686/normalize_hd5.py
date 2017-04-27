import pandas as pd
import numpy as np
from stopwords import stopwords
import sys, math
from collections import Counter
import h5py

stopwords = { w:1 for w in stopwords }

train_df = pd.read_csv(sys.argv[1])
train_df.fillna(" ", inplace=True)

def normalize_question(q):
    q = q.lower()
    q = q.replace("?","")
    q = q.replace(".","")
    q = q.replace(",","")
    q = q.replace(":"," : ")
    q = q.replace(","," , ")
    q = q.replace("/"," / ")
    q = q.replace("("," ( ")
    q = q.replace(")"," ) ")
    q = q.replace("["," [ ")
    q = q.replace("]"," ] ")
    q = q.replace("-"," - ")
    q = q.replace('"',"")
    q = q.replace("[math]", " [math] ")
    q = q.replace("[/math]", " [/math] ")
    words = q.lower().split()
    words = map(lambda x: x.strip(), words)
    words = filter(lambda x: len(x) > 0, words)
    return " ".join(words)

print normalize_question(train_df["question1"][0])

train_df["question1"] = map(normalize_question, train_df["question1"])
train_df["question2"] = map(normalize_question, train_df["question2"])

duplicates = train_df[train_df["is_duplicate"]==1]
duplicate_words = " ".join(duplicates["question1"].tolist()) + " ".join(duplicates["question2"].tolist()) 
duplicate_words = duplicate_words.split(" ")
dupcounts = Counter(duplicate_words)

nonduplicates = train_df[train_df["is_duplicate"]==0]
nonduplicate_words = " ".join(nonduplicates["question1"].tolist()) + \
                    " ".join(nonduplicates["question2"].tolist()) 
nonduplicate_words = nonduplicate_words.split(" ")
nondupcounts = Counter(nonduplicate_words)

for w in dupcounts:
    del nondupcounts[w]

ordered = sorted(dupcounts.items(), key=lambda x: -x[1]) + \
        sorted(nondupcounts.items(), key=lambda x: -x[1])
    
ordered = [w for w, n in ordered]
SingleBitEncoded = 200
print len(ordered), ordered[:SingleBitEncoded]

n = len(ordered) - SingleBitEncoded

nbits = int(1 + SingleBitEncoded + math.ceil(math.log(float(n), 2.0)))

print "nbits=", nbits

encoding = {" ":np.zeros((nbits,), dtype=np.uint16)}

for i, w in enumerate(ordered[:SingleBitEncoded]):
    bits = np.zeros((nbits,), dtype=np.uint16)
    bits[1+i] = 1
    encoding[w] = bits
    
print w, encoding[w]
    
for i, w in enumerate(ordered[SingleBitEncoded:]):
    bits = np.zeros((nbits,), dtype=np.uint16)
    bits[0] = 1
    j = SingleBitEncoded + 1
    n = i
    while n > 0:
        bits[j] = n%2
        n /= 2
        j += 1
    encoding[w] = bits
    
print w, encoding[w]
    
def encode_question(encoding, words, l=0):
    lst = map(lambda w: encoding[w], words)
    if l > len(lst):
        lst += [encoding[" "]]*(l-len(lst))
    return np.array(lst)
    
q1_encoded = [encode_question(encoding, q.split()) for q in train_df["question1"]]
q2_encoded = [encode_question(encoding, q.split()) for q in train_df["question2"]]

q1q2_encoded = zip(q1_encoded, q2_encoded)

maxl = np.array([max(len(q1), len(q2)) for q1, q2 in q1q2_encoded])

if False:
    print np.argmax(maxl)
    maxmaxl = max(maxl)

    print "grand max l=", maxmaxl

def pad_question(q, l, blank):
    if len(q) < l:
        pad = np.array([blank]*(l-len(q)))
        #print q.shape, pad.shape
        if len(q) == 0: q = pad
        else:   q = np.concatenate((q, pad), axis=0)
    return q

    def pad_questions(row, blank=None):
        q = row[0]
        l = row[1]
        return pad_question(q, l, blank)
    
    blank = encoding[" "]

    padded1 = np.array([pad_question(q, maxmaxl, blank) for q in q1_encoded])
    padded2 = np.array([pad_question(q, maxmaxl, blank) for q in q2_encoded])

    print "padded1:", padded1.shape
    print "padded2:", padded2.shape




