import pandas as pd
import numpy as np
from stopwords import stopwords
import sys, math
from collections import Counter

stopwords = { w:1 for w in stopwords }

train_df = pd.read_csv(sys.argv[1])
train_df.fillna(" ", inplace=True)

def normalize_question(q):
    q = q.lower()
    q = q.replace("?","")
    q = q.replace("..."," ")
    q = q.replace("."," ")
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
    q = q.replace("'s", " is ")
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

encoding = {" ":np.zeros((nbits,), dtype=np.uint8),
            "?":np.ones((nbits,), dtype=np.uint8)}

for i, w in enumerate(ordered[:SingleBitEncoded]):
    bits = np.zeros((nbits,), dtype=np.uint8)
    bits[1+i] = 1
    encoding[w] = bits
    
print w, encoding[w]
    
for i, w in enumerate(ordered[SingleBitEncoded:]):
    bits = np.zeros((nbits,), dtype=np.uint8)
    bits[0] = 1
    j = SingleBitEncoded + 1
    n = i
    while n > 0:
        bits[j] = n%2
        n /= 2
        j += 1
    encoding[w] = bits
    
print w, encoding[w]
    
def encode_question(encoding, words):
    lst = map(lambda w: encoding[w], words)
    lst.append(encoding["?"])
    return np.array(lst)
    
maxqid = max(max(train_df["qid1"]), max(train_df["qid2"]))

questions_dict = dict(
    [(qid, encode_question(encoding, q.split())) for i, qid, q in train_df[["qid1","question1"]].itertuples()]
)

questions_dict.update(
    dict(
        [(qid, encode_question(encoding, q.split())) for i, qid, q in train_df[["qid2","question2"]].itertuples()]
    )
)

assert len(questions_dict) == maxqid

questions_df = pd.DataFrame(
    [questions_dict[i] for i in xrange(1, maxqid+1)],
    columns=["encoded_question"], index=range(1, maxqid+1))

pairs_df = pd.DataFrame(columns=["qid1", "qid2", "is_duplicate"])

pairs_df["is_duplicate"] = train_df["is_duplicate"]
pairs_df["qid1"] = train_df["qid1"]
pairs_df["qid2"] = train_df["qid2"]

print pairs_df.head()

out = pd.HDFStore(sys.argv[2])
out["questions"] = questions_df
out["pairs"] = pairs_df
out["original"] = train_df
out.close()

