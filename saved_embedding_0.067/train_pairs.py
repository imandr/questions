import time, random
import numpy as np
import pandas as pd

from keras.callbacks import TensorBoard, ProgbarLogger, ModelCheckpoint

from BatchGenerator import PairsBatchGenerator
from models import createModelPairs

def run():

    import sys, getopt, os
    save_to = None
    load_from = None
    metadata = None

    batch_size = 60
    nworkers = 2
    verbose = 1

    opts, args = getopt.getopt(sys.argv[1:], "s:l:n:qw:m:")
    for opt, val in opts:
        if opt == "-s": save_to = val
        if opt == "-l": load_from = val
        if opt == "-w": 
            load_from = val
            save_to = val
        if opt == "-n": nworkers = int(val)
        if opt == "-q": verbose = 0
        if opt == "-m": metadata = val
        
    os.system("rm logs/*")
    
    print "Validation set created"
    #print validation_data

    model = createModelPairs()
    
    if load_from:
        model.load_weights(load_from, by_name=True)
        print
        print "model weights loaded from %s" % (load_from,)
        print 
        
    tb = TensorBoard(write_images = False, histogram_freq=1.0,
        embeddings_freq=1, embeddings_layer_names=["vocabulary"], embeddings_metadata=metadata,
    )
    callbacks = [tb]
    if save_to:
        callbacks.append(ModelCheckpoint(filepath=save_to, verbose=1, save_best_only=True, save_weights_only = True, 
            monitor="val_loss"))

    bg = PairsBatchGenerator(args[0], 1000)
    training_set_size = bg.training_samples()
    validation_data = bg.validateSet()
    model.fit_generator(bg.batches_guargded(batch_size), int(training_set_size/batch_size/10),
        epochs=1000, verbose=verbose, 
        workers=nworkers,
        callbacks=callbacks, validation_data=validation_data)

run()
    
