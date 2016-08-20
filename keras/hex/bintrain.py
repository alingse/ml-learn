
#coding=utf-8
#author@alingse
#2016.08.19

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation

from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils

from utils import activations
from utils import dump,load


import numpy as np
from random import shuffle
from random import choice

import sys

#binlen = 8 :0b1111 --> [0,0,0,0,1,1,1,1]
#numbeln = 3: 15 --> [0,1,5]
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


def train(X_train,Y_train,actives):
    numlen = X_train.shape[1]
    binlen = Y_train.shape[1]
    print(numlen,binlen)
    print(actives)


    model = Sequential()

    model.add(Dense(2*binlen,input_dim=numlen,activation=actives[0],init='uniform'))
    model.add(Dense(4*binlen,input_dim=2*binlen,activation=actives[1],init='normal'))
    model.add(Dense(2*binlen,input_dim=4*binlen,activation=actives[2]))
    model.add(Dense(binlen,input_dim=2*binlen,activation='hard_sigmoid'))

    rmsprop = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08)

    model.compile(loss='binary_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    print('start fit')
    model.fit(X_train, Y_train,
              batch_size=600, nb_epoch=max(1000,250*binlen),
              verbose=2,shuffle=True,
              validation_split=0.2)

    print(actives)
    return model


def main(binlen=4,name='test',need_train=True,test_active=True):
    print(need_train,test_active)

    X_train,Y_train,x_test,y_test = load_XY(binlen=binlen)

    if need_train:
        if test_active:
            actives = [choice(activations) for i in range(3)]
        else:
            #['relu', 'sigmoid', 'linear']
            #['relu', 'sigmoid', 'hard_sigmoid']
            #['sigmoid', 'softmax', 'tanh']
            #['sigmoid', 'tanh', 'softplus']
            #['linear', 'softsign', 'tanh']
            #['linear', 'sigmoid', 'softsign']
            #['linear', 'softmax', 'tanh']
            #['linear', 'hard_sigmoid', 'softsign']
            #['hard_sigmoid', 'tanh', 'softsign']        
            #['softplus', 'sigmoid', 'softplus']
            #['softplus', 'softsign', 'relu']
            #['softsign', 'softmax', 'softplus']
            #['tanh', 'relu', 'relu']
            #['tanh', 'softmax', 'softmax']

            #['softplus', 'tanh', 'softsign']
            #['softplus', 'softmax', 'tanh']
            #['relu', 'softmax', 'tanh']
            #['relu', 'tanh', 'softsign']
            #['linear', 'tanh', 'linear']
            #['tanh', 'sigmoid', 'softsign']
            #actives = ['tanh', 'sigmoid', 'softsign']
            #actives = ['linear', 'tanh', 'linear']
            #actives = ['relu', 'tanh', 'softsign']
            #actives = ['relu', 'softmax', 'tanh']
            #actives = ['softplus', 'tanh', 'softsign']
            actives = ['softplus', 'softmax', 'tanh']

        #for train
        model = train(X_train,Y_train,actives)
        #score
        score = model.evaluate(X_train,Y_train,batch_size=20,verbose=2)
        print(score)
    else:
        model = load(name)

    y_seq = model.predict(x_test)
    y_seq2 = np.float32(y_seq>0.5)

    if test_active:
        if np.sum(np.abs(y_seq - y_test)) == 0.0:
            print(True)
        #not test
        exit()

    print(x_test)
    print(y_seq)
    print(y_test)
    print(y_seq2)
    print(y_seq==y_test)
    print(y_seq2==y_test)
    print('diff:sum: |y_seq - y_test|',np.sum(np.abs(y_seq - y_test)))
    print('diff:sum: |y_seq2 - y_test|',np.sum(np.abs(y_seq2 - y_test)))

    if need_train:
        dump(model,name)

if __name__ == '__main__':
    need_train = True
    test_active = True
    test_active = False

    #binlen = 4
    binlen = int(sys.argv[1])
    
    name = 'bin.test.'+str(binlen)
    
    main(binlen,name=name,need_train=need_train,test_active=test_active)