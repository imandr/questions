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

VocabularySize, EmbeddingSize, InnerSize, OutSize = 10000+100, 30, 60, 2

def createModelSentences():
    
    inp = Input(shape=(None,))
    
    vocabulary = Embedding(VocabularySize, EmbeddingSize, name="vocabulary")(inp)
    word_parser = LSTM(InnerSize, return_sequences=True, implementation=1,  name="word_parser")(vocabulary)
    sentence_parser = LSTM(InnerSize, return_sequences=False, implementation=1,  name="sentence_parser")(word_parser)
    out = Dense(OutSize, activation=Activation("softmax"), name="output")(sentence_parser)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=["accuracy"])
    return model
    
def createModelPairs():
    
    q1 = Input(shape=(None,))
    q2 = Input(shape=(None,))
    match1 = Input(shape=(None,1))
    match2 = Input(shape=(None,1))
    
    vocabulary = Embedding(VocabularySize, EmbeddingSize, name="vocabulary")
    word_parser = LSTM(InnerSize, return_sequences=True, implementation=1,  
	kernel_regularizer = regularizers.l2(0.0001), 
	name="word_parser")
    
    voc_1 = vocabulary(q1)
    voc_2 = vocabulary(q2)
    
    words_1 = word_parser(voc_1)
    words_2 = word_parser(voc_2)
    
    words_plus_match_1 = concatenate([words_1, match1], name="words_plus_match_1")
    words_plus_match_2 = concatenate([words_2, match2], name="words_plus_match_2")
    
    signature = LSTM(InnerSize*2, return_sequences=False, implementation=1,  
	kernel_regularizer = regularizers.l2(0.0001), 
	name="signature_xwide")
    
    signature_1 = signature(words_plus_match_1)
    signature_2 = signature(words_plus_match_2)
    
    out = Dense(OutSize, activation="softmax", 
	kernel_regularizer = regularizers.l2(0.0001), 
	name="output_xwide")(concatenate([signature_1, signature_2]))
    
    model = Model(inputs=[q1, match1, q2, match2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=["accuracy"])
    return model
    
    
    
    
    

