#coding=utf-8
#author@alingse
#2016.08.19

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,RepeatVector
from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils

import numpy as np
from random import shuffle


#0 --> 255
#0b0 --> 0b11111111
def load_data(binlen = 8):
    maxnum = eval('0b'+'1'*binlen)
    numlen = len(str(maxnum))

    count = maxnum + 1

    data = np.zeros((count,numlen),dtype=np.uint8)
    label = np.zeros((count,binlen),dtype=np.uint8)
    for i in range(count):

        _i_str = str(i).zfill(numlen)
        _i_data = np.array(map(int,_i_str))
        _i_bin = bin(i)[2:].zfill(binlen)
        _i_label = np.array(map(int,_i_bin))

        data[i,:] = _i_data
        label[i,:] = _i_label


    #解决数量太少训练不出来
    data = np.repeat(data,20,axis=0)
    label = np.repeat(label,20,axis=0)

    #太少shuffle 也不能解决数据训练不足的
    #shuffle 避免repeat 数据集中重复
    index = list(range(len(label)))
    shuffle(index)
    data = data[index]
    label = label[index]

    return data,label


def train(data,label):
    
    numlen = data.shape[1]
    #3
    binlen = label.shape[2]
    #8

    model = Sequential()

    model.add(Dense(binlen,input_dim=numlen,init='normal'))
    model.add(Activation('tanh'))
    model.add(Dense(binlen,input_dim=binlen))
    model.add(Activation('softmax'))
    for i in range(binlen-2):
        model.add(Dense(binlen,input_dim=binlen))
        model.add(Activation('tanh'))
        model.add(Dense(binlen,input_dim=binlen))
        model.add(Activation('softmax'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    #损失函数还不知道选啥
    model.compile(loss='binary_crossentropy',optimizer=sgd)
    #model.compile(loss='mse', optimizer='adam')

    model.fit(data, label, batch_size=10,
                        nb_epoch=500,shuffle=True,
                        verbose=1,
                        validation_split=0.2)

    return model


def dump(model,save_name):
    with open('{}.model.json'.format(save_name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(save_name))


def main(name='test'):
    
    data,label = load_data()
    label = np_utils.to_categorical(label)
    
    model = train(data,label)
    score = model.evaluate(data,label,batch_size=4,verbose=2)
    print(score)

    _data = np.array([[1,6,5]])
    classes = model.predict_classes(_data)
    print(classes)
    print(bin(165))
    dump(model,name)
    
    

if __name__ == '__main__':
    name = 'bin'
    main(name)