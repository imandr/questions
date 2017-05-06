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
            
class SentencesBatchGenerator:
    
    def __init__(self, data_file, validate_size):
        self.I = 0
        self.ResetNeeded = False
        print "data file: %s" % (data_file,)
        self.Store = pd.HDFStore(data_file)
        self.DF = self.Store["sentences"]
        inx = range(len(self.DF))
        random.shuffle(inx)
        self.ValidateInx = inx[:validate_size]
        self.TrainInx = inx[validate_size:]
        sample_q = self.DF.loc[1,"sentence"]
        self.DType = sample_q.dtype

    def training_samples(self):
        return len(self.TrainInx)
        
    def slice(self, inx):
        data = self.DF.loc[inx]
        n = len(data)
        if n == 0:   return None
        
        L = max(map(len, data["sentence"]))
        
        x = np.zeros((n, L), dtype=self.DType)
        y = np.zeros((n, 2), dtype=np.float32)
        
        for i, (_, s, c) in enumerate(data.itertuples()):
            if len(s):
                x[i, -len(s):] = s
                y[i, c] = 1.0
            else:
                y[i, 0] = 1.0
        return x, y

    def batch(self, bsize, increment=True, max_samples=None):
        b = self.slice(self.TrainInx[self.I:self.I+bsize])
        if increment:   self.I += bsize
        return b
        
    def validateSet(self):
        return self.slice(self.ValidateInx)
        
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
        
class PairsBatchGenerator:

    def __init__(self, data_file, validate_size):
        self.I = 0
        self.ResetNeeded = False
        print "data file: %s" % (data_file,)
        self.Store = pd.HDFStore(data_file)
        self.DF = self.Store["pairs"]
        inx = range(len(self.DF))
        random.shuffle(inx)
        self.ValidateInx = inx[:validate_size]
        self.TrainInx = inx[validate_size:]

    def training_samples(self):
        return len(self.TrainInx)
        
    def slice(self, inx):
        data = self.DF.loc[inx]
        n = len(data)
        if n == 0:   return None
        
        L = max(
            max(map(len, data["q1"])),
            max(map(len, data["q2"]))
        )

        q1_out = np.zeros((n, L), dtype=np.uint16)
        q2_out = np.zeros((n, L), dtype=np.uint16)
        match1_out = np.zeros((n, L, 1), dtype=np.float32)
        match2_out = np.zeros((n, L, 1), dtype=np.float32)
        y = np.zeros((n, 2), dtype=np.float32)
        
        for i, (_, pid, is_dup, q1, match1, q2, match2) in enumerate(data.itertuples()):
            l1 = len(q1)
            l2 = len(q2)
            if l1:
                q1_out[i,-l1:] = q1
                match1_out[i,-l1:,0] = match1
            if l2:
                q2_out[i,-l2:] = q2
                match2_out[i,-l2:,0] = match2
            y[i, is_dup] = 1
        return [q1_out, match1_out, q2_out, match2_out], y

    def batch(self, bsize, increment=True, max_samples=None):
        b = self.slice(self.TrainInx[self.I:self.I+bsize])
        if increment:   self.I += bsize
        return b
        
    def validateSet(self):
        return self.slice(self.ValidateInx)
        
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
    bg = SentencesBatchGenerator(sys.argv[1], 1000)

    i = 0
    for x, y in bg.batches(5):
        print x[0].shape, x[0]
        print y[0]
        i += 1
        if i > 0:   break
    
