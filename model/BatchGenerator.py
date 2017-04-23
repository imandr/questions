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
        self.PairsDF = self.Store["train"]
        inx = range(len(self.PairsDF))
        random.shuffle(inx)
        self.ValidateInx = inx[:validate_size]
        self.TrainInx = inx[validate_size:]
        sample_q = self.PairsDF.loc[1,"q1"]
        self.RowSize = sample_q.shape[1]
        self.DType = sample_q.dtype
        print "Row size=", self.RowSize, "    dtype=", self.DType

    @property
    def rowSize(self):
        return self.RowSize 

    def training_samples(self):
        return len(self.TrainInx)
        
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
        
        L = max(
                [max(len(q1), len(q2)) for _, q1, q2 in pairs[["q1", "q2"]].itertuples()]
                )
        
        #print L
        
        x1 = np.zeros((n, L, self.rowSize), self.DType)
        x2 = np.zeros((n, L, self.rowSize), self.DType)
        y = np.zeros((n, 2))
        for i, (_, q1, q2, dup) in enumerate(pairs.itertuples()):
            #print qids1[i]
            l1 = len(q1)
            x1[i,-l1:,:] = q1
            #x3[i,-l1:,:] = q1

            l2 = len(q2)
            x2[i,-l2:,:] = q2
            
            y[i, dup] = 1.0
        return [x1, x2], [y]

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
    for (x1, x2), y in bg.batches(5):
        print x1.shape, x1[0]
        print x2.shape, x2[0]
        print y[0]
        i += 1
        if i > 0:   break
    
