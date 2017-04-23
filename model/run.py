import time, random
import numpy as np

from keras.models import Sequential, Model

from model import createModel
from BatchGenerator import QuestionsBatchGenerator


    
def run():

    import sys, getopt, os
    load_from = None

    opts, args = getopt.getopt(sys.argv[1:], "l:")
    for opt, val in opts:
        if opt == "-l": load_from = val
        
    bg = QuestionsBatchGenerator(args[0], 1000)

    trainig_set_size = bg.training_samples()

    validation_data = bg.validateSet()
    print "Validation set created"
    #print validation_data

    model = createModel(bg.rowSize, 2)
    if load_from:   model.load_weights(load_from)
    
    for i in range(100):
        pair_x, pair_truth = bg.loadPair(i)
        q1, q2, dup_ = pair_truth
        print "Question 1: ", q1
        print "Question 2: ", q2
        
        y = model.predict(pair_x)[0]
        dup = y[1] > 0.5
        
        correct = dup == dup_
        
        print "Same: %s   model: %.3f (%s)" % (dup_, y[1], "CORRECT" if correct else "incorrect")
        print
        

run()
    
