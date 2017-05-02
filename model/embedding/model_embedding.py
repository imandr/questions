import time, random
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, concatenate, Embedding
from keras.optimizers import Adadelta
from keras.layers.core import Flatten, Permute, Reshape, Lambda, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard, ProgbarLogger, ModelCheckpoint
import keras.backend as K
from keras import regularizers

def createModel(voc_size, emb_size, inner_size, nout):
    
    inp = Input(shape=(None,))
    
    vocabulary = Embedding(voc_size, emb_size, name="vocabulary")(inp)
    word_parser = LSTM(inner_size, return_sequences=True, implementation=1,  name="word_parser")(vocabulary)
    sentence_parser = LSTM(inner_size, return_sequences=False, implementation=1,  name="sentence_parser")(word_parser)
    out = Dense(nout, activation=Activation("softmax"), name="output")(sentence_parser)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=["accuracy"])
    return model

