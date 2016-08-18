#coding=utf-8
#author@alingse
#2016.08.17

from keras.models import model_from_json
from PIL import Image
import numpy as np
import sys

def load_model(name):
    with open('{}.model.json'.format(name),'r') as f:
        model = model_from_json(f.read())
    model.load_weights('{}.model.weigthts.h5'.format(name))
    return model


def mk_test(img):
    content = img.tobytes()
    img = Image.frombytes('L',img.size, content)
    data = np.asarray(img)
    X_test = np.asarray([[data]])
    return X_test


if __name__ == '__main__':
    name = 'classimagetype'
    imgpath = sys.argv[1]

    model = load_model(name)
    img = Image.open(imgpath)
    X_test = mk_test(img)
    classes = model.predict_classes(X_test)
    print(classes[0])
