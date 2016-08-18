#coding=utf-8
#author@alingse
#2016.08.17


from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,RepeatVector
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils

import numpy as np

from random import shuffle

from itertools import product
from operator import itemgetter


def load_data():
    #xor
    _data = np.array([[0,0],[0,1],[1,0],[1,1]])
    _label = np.array([0,1,1,0])
    
    #解决数量太少训练不出来
    data = np.repeat(_data,2,axis=0)
    label = np.repeat(_label,2,axis=0)

    #shuffle 避免repeat 数据集中重复
    index = list(range(len(label)))
    shuffle(index)
    data = data[index]
    label = label[index]

    return data,label


def train(data,label,activeA='tanh',activeB='softmax'):
    model = Sequential()
    #因为 A ^ B == (~ A & B)|(A & ~ B)
    #神经元个数自然是越多越好，不能太少。
    #这里 32 就比 4 收敛的快
    #好的 init 也很重要
    #有些是永远不会收敛的，要有智慧
    #tanh--softmax
    model.add(Dense(4,input_dim=2,init='normal'))
    model.add(Activation(activeA))
    model.add(Dense(2,input_dim=4))
    model.add(Activation(activeB))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    #loss 要有针对性，能衡量东西才行。
    model.compile(loss='binary_crossentropy',optimizer=sgd)
    #model.compile(loss='mse', optimizer='adam')
    #batch_size 不能占数据量太大比例。
    #次数太少，不容易迭代进去
    #validation_split 要合理，不能减少训练量。
    #虽然如此，但是每次训练还是有随机性。
    model.fit(data, label, batch_size=1,
                        nb_epoch=1000,shuffle=True,
                        verbose=1,
                        validation_split=0.0)

    return model


def dump(model,save_name):
    with open('{}.model.json'.format(save_name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(save_name))


def main(name='test'):
    #data
    _data = np.array([[0,0],[0,1],[1,0],[1,1]])

    data,label = load_data()
    label = np_utils.to_categorical(label)
    
    model = train(data,label)
    score = model.evaluate(data,label,batch_size=4,verbose=2)
    print(score)
    classes = model.predict_classes(_data)
    print(classes)
    #
    '''
    actives = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
    result = []

    for A,B in product(actives,actives):
        s = 0
        for i in range(5):
            model = train(data,label,A,B)
            score = model.evaluate(data,label,batch_size=10,verbose=0)
            print(A,B,score)
            #test
            classes = model.predict_classes(_data)
            print(classes)
            s += score
        result.append((s,A,B))
    
    result = sorted(result,key=itemgetter(0))
    print(result)
    '''


if __name__ == '__main__':
    name = 'xor'
    main(name)