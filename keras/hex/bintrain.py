
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

    X_train = np.zeros((count,numlen),dtype=np.uint8)
    Y_train = np.zeros((count,binlen),dtype=np.uint8)
    for i in range(count):

        i_str = str(i).zfill(numlen)
        x_seq = np.array(map(int,i_str))
        i_bin = bin(i)[2:].zfill(binlen)
        y_seq = np.array(map(int,i_bin))

        X_train[i] = x_seq
        Y_train[i] = y_seq

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

    model.add(Dense(4*binlen,input_dim=numlen,activation='tanh',init='normal'))
    model.add(Dense(2*binlen,input_dim=4*binlen,activation='softmax'))
    model.add(Dense(binlen,input_dim=2*binlen,activation='tanh'))
    model.add(Dense(binlen,input_dim=binlen,activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print('start fit')
    model.fit(X_train, Y_train,
              batch_size=30, nb_epoch=10000,
              verbose=2,shuffle=True,
              validation_split=0.3)

    return model


def dump(model,save_name):
    with open('{}.model.json'.format(save_name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(save_name))


def main(name='test'):
    
    X_train,Y_train = load_XY(binlen=2)
    
    model = train(X_train,Y_train)
    score = model.evaluate(X_train,Y_train,batch_size=20,verbose=2)
    print(score)

    x_test = np.array([0,1]).reshape(2,1)
    y_seq = model.predict(x_test)
    print(y_seq)
    print(y_seq>0.5)
    print(bin(13))
    dump(model,name)
    

if __name__ == '__main__':
    name = 'bin'
    main(name)