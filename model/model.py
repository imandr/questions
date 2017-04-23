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

def last(x):
    return x[:,-1,:]

def createModel(row_size, nout):
    
    inner = int(row_size * 1.5)
    inner = row_size
    
    in1 = Input(shape=(None, row_size))
    in2 = Input(shape=(None, row_size))
    
    lstm_a = LSTM(inner, return_sequences=True, implementation=1,  name="lstm_a")
    lstm_a_1 = lstm_a(in1)
    lstm_a_2 = lstm_a(in2)
    
    lstm_b = LSTM(inner, return_sequences=True, implementation=1,  name="lstm_b")
    
    merge_a_1 = concatenate([lstm_a_1, lstm_a_2], name="merge_a_1")
    merge_a_2 = concatenate([lstm_a_2, lstm_a_1], name="merge_a_2")

    lstm_b_1 = lstm_b(merge_a_1)
    lstm_b_2 = lstm_b(merge_a_2)
    
    merge_c = concatenate([lstm_b_1, lstm_b_2], name="merge_c")
    
    lstm_c = LSTM(inner, return_sequences=False, implementation=1,  name="lstm_c")(merge_c)
    
    dense = Dense(nout*10, activation="tanh", name="dense_c")(lstm_c)
    out = Dense(nout, activation=Activation("softmax"), name="output")(dense)
    model = Model(inputs=[in1, in2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=["accuracy"])
    return model

