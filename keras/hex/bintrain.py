
#coding=utf-8
#author@alingse
#2016.08.19

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import model_from_json

from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils

import numpy as np
from random import shuffle
from random import choice

import sys


#0 --> 255
#0b0 --> 0b11111111
#seq
def load_XY(binlen = 8):
    maxnum = eval('0b'+'1'*binlen)
    numlen = len(str(maxnum))

    count = maxnum + 1

    X_train = np.zeros((count,numlen),dtype=np.float32)
    Y_train = np.zeros((count,binlen),dtype=np.float32)
    for i in range(count):

        i_str = str(i).zfill(numlen)
        x_seq = np.array(map(int,i_str))
        i_bin = bin(i)[2:].zfill(binlen)
        y_seq = np.array(map(int,i_bin))

        X_train[i] = x_seq
        Y_train[i] = y_seq
    
    x_test = X_train
    y_test = Y_train

    X_train = np.repeat(X_train,300,axis=0)
    Y_train = np.repeat(Y_train,300,axis=0)

    index = list(range(X_train.shape[0]))
    shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    return X_train,Y_train,x_test,y_test


def train(X_train,Y_train):

    numlen = X_train.shape[1]
    binlen = Y_train.shape[1]
    print(numlen,binlen)

    #copy from keras
    model = Sequential()

    activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
    actives = [choice(activations) for i in range(3)]

    #['softplus', 'softmax', 'tanh']
    #['relu', 'sigmoid', 'linear']
    #['relu', 'sigmoid', 'hard_sigmoid']
    #['sigmoid', 'softmax', 'tanh']
    #['linear', 'softsign', 'tanh']
    #['tanh', 'sigmoid', 'softsign']
    #['hard_sigmoid', 'tanh', 'softsign']
    #['softplus', 'tanh', 'softsign']
    #['softplus', 'sigmoid', 'softplus']
    #actives = ['relu', 'tanh', 'tanh']
    actives = ['tanh', 'sigmoid', 'softsign']
    print(actives)

    model.add(Dense(2*binlen,input_dim=numlen,activation=actives[0],init='uniform'))
    model.add(Dense(4*binlen,input_dim=2*binlen,activation=actives[1],init='normal'))
    model.add(Dense(2*binlen,input_dim=4*binlen,activation=actives[2]))
    model.add(Dense(binlen,input_dim=2*binlen,activation='hard_sigmoid'))


    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

    model.compile(loss='binary_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    print('start fit')
    model.fit(X_train, Y_train,
              batch_size=600, nb_epoch=250*binlen,
              verbose=2,shuffle=True,
              validation_split=0.2)

    print(actives)
    return model


def dump(model,save_name):
    with open('{}.model.json'.format(save_name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(save_name))


def load(name):
    with open('{}.model.json'.format(name),'r') as f:
        model = model_from_json(f.read())
    model.load_weights('{}.model.weigthts.h5'.format(name))
    return model


def main(binlen=4,name='test'):

    X_train,Y_train,x_test,y_test = load_XY(binlen=binlen)
    
    model = train(X_train,Y_train)
    score = model.evaluate(X_train,Y_train,batch_size=20,verbose=2)
    print(score)
    #model = load('bin.3')

    #x_test = np.array([0,5,0,6,0,7,0,9,1,2,1,3,1,4,1,5]).reshape(6,2)
    #x_test = np.array([0,1,2,3,4,5,6,7]).reshape(8,1)
    #x_test = np.array([0,1,2,3]).reshape(4,1)
    #x_test = X_train[1:2**binlen]


    y_seq = model.predict(x_test)
    y_seq2 = np.float32(y_seq>0.5)

    if np.sum(np.abs(y_seq - y_test)) == 0.0:
        print(True)
    #exit()

    print(x_test)
    print(y_seq)
    print(y_test)
    print(y_seq2)
    print(y_seq==y_test)
    print(y_seq2==y_test)
    print('diff:sum: |y_seq - y_test|',np.sum(np.abs(y_seq - y_test)))
    print('diff:sum: |y_seq2 - y_test|',np.sum(np.abs(y_seq2 - y_test)))
    dump(model,name)

if __name__ == '__main__':
    name = 'bin'
    binlen = 4
    if len(sys.argv) == 2:
        binlen = int(sys.argv[1])
    main(binlen,name)