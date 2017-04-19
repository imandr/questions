import numpy as np
import pandas as pd
import random, time
from threading import Lock

class GeneratorGuard(object):
    
    def __init__(self, gen):
        self.Lock = Lock()
        self.G = gen
        
    def __iter__(self):    return self
    
    def next(self):
        with self.Lock:
            return next(self.G)
            
class QuestionsBatchGenerator:
    
    def __init__(self, data_file, validate_size):
        self.I = 0
        self.ResetNeeded = False
        self.Store = pd.HDFStore(data_file)
        self.QuestionsDF = self.Store["questions"]
        self.OriginalDF = self.Store["original"]
        pairs_df = self.Store["pairs"]
        inx = range(len(pairs_df))
        random.shuffle(inx)
        self.ValidateInx = inx[:validate_size]
        self.TrainInx = inx[validate_size:]
        self.PairsDF = pairs_df
        sample_q = self.QuestionsDF.loc[1,"encoded_question"]
        self.RowSize = sample_q.shape[1]
        self.DType = sample_q.dtype
        print self.RowSize, self.DType

    @property
    def rowSize(self):
        return self.RowSize 

    def training_samples(self):
        return len(self.TrainInx)
        
    def pad(self, q, l):
        if len(q) < l:
            out = np.zeros((l, q.shape[1]), dtype=q.dtype)
            out[:len(q)] = q
            q = out
        return q
                
    def loadPair(self, i):
        x, y = self.slice([i])
        original = self.OriginalDF.loc[i]
        q1 = original.question1
        q2 = original.question2
        dup = original.is_duplicate
        return x, (q1, q2, dup)
        
        
    
    def batch(self, bsize, increment=True, max_samples=None):
        b = self.slice(self.TrainInx[self.I:self.I+bsize])
        if increment:   self.I += bsize
        return b
        
    def validateSet(self):
        return self.slice(self.ValidateInx)
        
    def slice(self, inx):
        pairs = self.PairsDF.loc[inx]
        n = len(pairs)
        if n == 0:   return None
        qids1 = pairs["qid1"].tolist()
        qids2 = pairs["qid2"].tolist()
        duplicate = pairs["is_duplicate"].tolist()
        qids = set(qids1 + qids2)
        
        L = max((x.shape[0] for x in self.QuestionsDF["encoded_question"][qids]))
        
        #print L
        
        x1 = np.zeros((n, L, self.rowSize), self.DType)
        x2 = np.zeros((n, L, self.rowSize), self.DType)
        x3 = np.zeros((n, L, self.rowSize*2), self.DType)
        y = np.zeros((n, 2))
        for i in xrange(n):
            #print qids1[i]
            q1 = self.QuestionsDF.loc[qids1[i]][0]
            l1 = len(q1)
            x1[i,-l1:,:] = q1
            #x3[i,-l1:,:] = q1

            q2 = self.QuestionsDF.loc[qids2[i]][0]
            l2 = len(q2)
            x2[i,-l2:,:] = q2
            
            l12 = max(l1, l2)
            
            #print l1, l2, l12
            x3[i,L-l12:L-l12+l1,:self.rowSize] = q1
            x3[i,L-l12:L-l12+l2,self.rowSize:] = q2

            dup = duplicate[i]
            y[i, dup] = 1.0
        return [x1, x2, x3], [y]

    def batches(self, bsize, randomize = False, max_samples = None):
        if randomize:
            random.shuffle(self.TrainInx)
        self.I = 0
        while True:
            b = self.batch(bsize)
            if not b:   break
            yield b
            

    def batches_infinite(self, bsize, randomize=True):
        while True:
            n = 0
            for b in self.batches(bsize, randomize=randomize):
                if self.ResetNeeded:
                    self.ResetNeeded = True
                    break
                n += len(b[0])
                yield b
            #print "---- reset batches after %d samples ----" % (n,)
            
    def reset(self):
        self.ResetNeeded = True
            
    def batches_guargded(self, batch_size, dropout=0.0):
        return GeneratorGuard(self.batches_infinite(batch_size, randomize=True))
            

if __name__ == "__main__":
    import sys
    bg = QuestionsBatchGenerator(sys.argv[1], 1000)

    i = 0
    for (x1, x2, x3), y in bg.batches(5):
        print x1.shape, x1[0]
        print x2.shape, x2[0]
        print x3.shape, x3[0]
        print y[0]
        i += 1
        if i > 0:   break
    
