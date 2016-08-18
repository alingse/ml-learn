#coding=utf-8
#author@alingse
#2016.08.17

#learn form https://github.com/wepe

from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

from PIL import Image
import numpy as np
from random import shuffle
import os

def load_data(image_path):
    files = os.listdir(image_path)
    length = len(files)
    data = np.empty((length,1,60,150),dtype="float32")
    label = np.empty((length,),dtype="uint8")

    
    for i,file in enumerate(files):
        img = Image.open('{}/{}'.format(lablepath,file))
        data[i,:,:,:] = np.asarray(img)
        p = int(file.split('.')[-2])
        p = 0 if p == 98 else 1
        label[i] = p
    return data,label


def train(image_path,save_name='test'):
    data, _label = load_data(image_path)

    label = np_utils.to_categorical(_label,2)

    #for model
    model = Sequential()
    #卷积层
    model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape = (1,60,150)))
    #激活
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    #池
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 3, 3, border_mode='valid')) 
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, init='normal'))
    model.add(Activation('tanh'))

    model.add(Dense(2, init='normal'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                 optimizer=sgd)


    model.fit(data, label, batch_size=10,
                        nb_epoch=10,shuffle=True,
                        verbose=1,
                        validation_split=0.2)

    loss_and_metrics = model.evaluate(data,label,batch_size=10)
    

    with open('{}.model.json'.format(save_name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(save_name))



if __name__ == '__main__':
    lablepath='./label_image/'
    name = 'classimagetype'
    #load_data(lablepath)
    train(lablepath,save_name=name)