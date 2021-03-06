import time, random
import numpy as np

from keras.callbacks import TensorBoard, ProgbarLogger, ModelCheckpoint

from BatchGenerator import QuestionsBatchGenerator
from model import createModel
    
def run():

    import sys, getopt, os
    save_to = None
    load_from = None

    batch_size = 30
    nworkers = 2
    verbose = 1

    opts, args = getopt.getopt(sys.argv[1:], "s:l:n:qw:")
    for opt, val in opts:
        if opt == "-s": save_to = val
        if opt == "-l": load_from = val
        if opt == "-w": 
            load_from = val
            save_to = val
        if opt == "-n": nworkers = int(val)
        if opt == "-q": verbose = 0
        
    os.system("rm logs/*")
    
    bg = QuestionsBatchGenerator(args[0], 1000)

    trainig_set_size = bg.training_samples()

    validation_data = bg.validateSet()
    print "Validation set created"
    #print validation_data

    model = createModel(bg.rowSize, 2)
    
    if load_from:
        model.load_weights(load_from)
        print
        print "model weights loaded from %s" % (load_from,)
        print 
        
    tb = TensorBoard(write_images = False, histogram_freq=1.0)
    callbacks = [tb]
    if save_to:
	callbacks.append(ModelCheckpoint(filepath=save_to, verbose=1, save_best_only=True, save_weights_only = True, monitor="val_loss"))
    
    model.fit_generator(bg.batches_guargded(batch_size), int(trainig_set_size/batch_size/20),
            epochs=1000, verbose=verbose, workers=nworkers, callbacks=callbacks, validation_data=validation_data)

run()
    
