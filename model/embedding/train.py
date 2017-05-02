import time, random
import numpy as np
import pandas as pd

from keras.callbacks import TensorBoard, ProgbarLogger, ModelCheckpoint

from BatchGenerator import SentencesBatchGenerator
from model_embedding import createModel

def getData(filename, validate_size):
    df = pd.HDFStore(filename)["sentences"]
    s = df["sentence"]
    c = df["classification"]
    y = np.zeros((len(c), 2), dtype=np.float32)
    y[c==1, 1] = 1.0
    y[c==0, 0] = 1.0
    
    tx = s[validate_size:]
    ty = y[validate_size:]
    
    vx = s[:validate_size]
    vy = y[:validate_size]
    
    return tx, ty, vx, vy
    
def run():

    import sys, getopt, os
    save_to = None
    load_from = None

    batch_size = 60
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
    
    print "Validation set created"
    #print validation_data

    model = createModel(10000+100, 30, 60, 2)
    
    if load_from:
        model.load_weights(load_from, by_name=True)
        print
        print "model weights loaded from %s" % (load_from,)
        print 
        
    tb = TensorBoard(write_images = False, histogram_freq=1.0,
        embeddings_freq=1, embeddings_layer_names=["vocabulary"], embeddings_metadata={"vocabulary":"../../data/vocabulary.tsv"}
    )
    callbacks = [tb]
    if save_to:
	callbacks.append(ModelCheckpoint(filepath=save_to, verbose=1, save_best_only=True, save_weights_only = True, 
        monitor="val_loss"))

    if True:      
        bg = SentencesBatchGenerator(args[0], 2000)
        training_set_size = bg.training_samples()
        validation_data = bg.validateSet()
        model.fit_generator(bg.batches_guargded(batch_size), int(training_set_size/batch_size/200),
            epochs=1000, verbose=verbose, 
            workers=nworkers,
            callbacks=callbacks, validation_data=validation_data)
    else:
        tx, ty, vx, vy = getData(args[0], 0)
        print tx.shape, ty.shape
        training_set_size = len(tx)
        model.fit(tx, ty, batch_size = batch_size, validation_split=0.1, shuffle=True, epochs=100)

run()
    
