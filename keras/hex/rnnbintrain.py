
#coding=utf-8
#author@alingse
#2016.08.19

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation, Flatten,RepeatVector
from keras.layers import Embedding
from keras.layers import LSTM

from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils

import numpy as np
from random import shuffle
from random import choice


#0 --> 255
#0b0 --> 0b11111111
#seq
def load_XY(binlen = 8):
    maxnum = eval('0b'+'1'*binlen)
    numlen = len(str(maxnum))

    count = maxnum + 1

    X_train = np.zeros((count,numlen,1),dtype=np.float32)
    Y_train = np.zeros((count,binlen),dtype=np.float32)
    for i in range(count):

        i_str = str(i).zfill(numlen)
        x_seq = np.array(map(int,i_str))
        i_bin = bin(i)[2:].zfill(binlen)
        y_seq = np.array(map(int,i_bin))

        X_train[i,:,0] = x_seq[::-1]
        Y_train[i] = y_seq[::-1]

    X_train = np.repeat(X_train,20,axis=0)
    Y_train = np.repeat(Y_train,20,axis=0)

    index = list(range(X_train.shape[0]))
    shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    return X_train,Y_train


def train(X_train,Y_train):

    numlen = X_train.shape[1]
    binlen = Y_train.shape[1]
    print(numlen,binlen)

    #copy from keras
    model = Sequential()
    model.add(LSTM(8, return_sequences=True,
                   input_shape=(numlen,1))) 
    model.add(LSTM(8, return_sequences=True)) 
    model.add(LSTM(8))
    model.add(Dense(binlen,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print('start fit')
    model.fit(X_train, Y_train,
              batch_size=10, nb_epoch=100,
              verbose=2,shuffle=True,
              validation_split=0.2)

    return model


def dump(model,save_name):
    with open('{}.model.json'.format(save_name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(save_name))


def main(name='test'):
    
    X_train,Y_train = load_XY(binlen=4)
    
    model = train(X_train,Y_train)
    score = model.evaluate(X_train,Y_train,batch_size=20,verbose=2)
    print(score)

    x_test = np.array([3,1]).reshape(1,2,1)
    y_seq = model.predict_classes(x_test)
    print(y_seq)
    print(bin(2))
    dump(model,name)
    

if __name__ == '__main__':
    name = 'bin-rnn'
    main(name)