import time, random
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, concatenate
from keras.optimizers import Adadelta
from keras.layers.core import Flatten, Permute, Reshape, Lambda, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard, ProgbarLogger, ModelCheckpoint
import keras.backend as K
from keras import regularizers

from BatchGenerator import QuestionsBatchGenerator

def createModel(row_size, nout):
    
    in1 = Input(shape=(None, row_size))
    in2 = Input(shape=(None, row_size))
    in3 = Input(shape=(None, row_size))
    
    lstm_a = LSTM(row_size, return_sequences=True)
    lstm_a_1 = lstm_a(in1)
    lstm_a_2 = lstm_a(in2)
    
    lstm_b = LSTM(row_size, return_sequences=False)
    lstm_b_1 = lstm_b(lstm_a_1)
    lstm_b_2 = lstm_b(lstm_a_2)
    
    lstm_c_1 = LSTM(row_size, return_sequences=True)(in3)
    lstm_c_2 = LSTM(row_size, return_sequences=False)(lstm_c_1)

    merged = concatenate([lstm_b_1, lstm_b_2, lstm_c_2])
    dense = Dense(row_size)(merged)
    out = Dense(nout, activation=Activation("softmax"))(dense)
    model = Model(inputs=[in1, in2, in3], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=["accuracy"])
    return model

    
def run():

    import sys, getopt, os
    save_to = None
    load_from = None

    batch_size = 20

    opts, args = getopt.getopt(sys.argv[1:], "s:l:")
    for opt, val in opts:
        if opt == "-s": save_to = val
        if opt == "-l": load_from = val
        
    os.system("rm logs/*")
    
    bg = QuestionsBatchGenerator(args[0], 1000)

    trainig_set_size = bg.training_samples()

    validation_data = bg.validateSet()
    print "Validation set created"
    #print validation_data

    model = createModel(bg.rowSize, 2)
    
    tb = TensorBoard(write_images = False, histogram_freq=1.0)
    callbacks = [tb]
    if save_to:
	callbacks.append(ModelCheckpoint(filepath=save_to, verbose=1, save_best_only=True, save_weights_only = True, monitor="val_loss"))
    
    model.fit_generator(bg.batches_guargded(batch_size), int(trainig_set_size/batch_size*0.1),
            epochs=100, workers=4, callbacks=callbacks, validation_data=validation_data)

run()
    
