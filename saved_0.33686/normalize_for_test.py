import pandas as pd
import numpy as np

import sys, math
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

bits2 = [
    (-1,-1),
    (-1,1),
    (1,-1),
    (1,1)
]

def make_encoding2(N):
    rows = (N-1)*N*2
    encoding_bits = []
    encoding_index = []
    k = 0
    for i in xrange(N-1):
        for j in xrange(i+1, N):
            for dx in xrange(4):
                encoding_index.append((i,j))
                encoding_bits.append(bits2[dx])
            k += 4
    return np.array(encoding_index), np.array(encoding_bits) 


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

most_frequent_words = [w for w, n in all_frequencies[:NFrequent]]
print "most frequent words:", most_frequent_words

NBitsFrequent = int(math.ceil(math.sqrt(float(len(most_frequent_words))/2.0)))+1
print NBitsFrequent, "bits need to be allocated for",len(most_frequent_words),"words"

frequent_encoding_index, frequent_encoding_bits = make_encoding2(NBitsFrequent)

print "encoding table of size",len(frequent_encoding_index),"is generated"

frequent_encoding = {w: (frequent_encoding_index[i], frequent_encoding_bits[i]) for i, w in enumerate(most_frequent_words)}


max_infrequent_words = 25
NBitsInfrequent = int(math.ceil(math.sqrt(float(max_infrequent_words)/2.0)))+1
print NBitsInfrequent,"bits needed to encode",max_infrequent_words,"infrequent words"
infrequent_encoding_index, infrequent_encoding_bits = make_encoding2(NBitsInfrequent)


def encode_question(words, nf_bits, frequent_encoding, nextra_bits, infrequent_encoding, match_set):
    nbits = nf_bits + nextra_bits + 1
    encoded = np.zeros((len(words)+1, nbits), dtype=np.int8)
    for i, w in enumerate(words):
        if w in frequent_encoding:
            inx, bits = frequent_encoding[w]
            encoded[i,inx] = bits
        else:
            inx, bits = infrequent_encoding[w]
            encoded[i,nf_bits:][inx] = bits
        if w in match_set:   encoded[i,-1] = 1
        #print "%s -> %s" % (w, encoded[i])
    encoded[-1] = -np.ones((nbits,), dtype=np.int8)
    return encoded

def encode_pair(q1, q2, frequent_set, frequent_encoding, nf_bits, infrequent_encoding_index, infrequent_encoding_bits, nextra_bits):
    # q1 and q2 are unnormalized questions
    
    q1 = normalize_question(q1)
    q2 = normalize_question(q2)
    q1_words = q1.split()
    q2_words = q2.split()
    q1_words_set = set(q1_words)
    q2_words_set = set(q2_words)
    match = q1_words_set & q2_words_set
    union = q1_words_set | q2_words_set
    infrequent = union - frequent_set
    infrequent_table_max = len(infrequent_encoding_index)-1
    infrequent_encoding = {w: 
            (  infrequent_encoding_index[min(i, infrequent_table_max)], 
               infrequent_encoding_bits[min(i, infrequent_table_max)]) 
        for i, w in enumerate(infrequent)}
    q1_encoded = encode_question(q1_words, nf_bits, frequent_encoding, nextra_bits, infrequent_encoding, match)
    q2_encoded = encode_question(q2_words, nf_bits, frequent_encoding, nextra_bits, infrequent_encoding, match)
    return q1_encoded, q2_encoded


frequent_set = set(most_frequent_words)

train_data = [
    encode_pair(q1, q2, frequent_set, frequent_encoding, NBitsFrequent, infrequent_encoding_index, infrequent_encoding_bits, NBitsInfrequent)+
    (is_dup,)
    for i, id, qid1, qid2, q1, q2, is_dup in train_df.itertuples()
]

train_encoded = pd.DataFrame(train_data, columns=["q1","q2","dup"])
train_store = pd.HDFStore(sys.argv[3])
train_store["train"] = train_encoded
train_store["original"] = train_df[["question1","question2"]]
train_store.close()

print "encoded dataframe is ready"