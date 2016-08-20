#coding=utf-8
#author@alingse
#2016.08.20

from keras.models import model_from_json

activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']

def dump(model,name):
    with open('{}.model.json'.format(name),'w') as f:
        f.write(model.to_json())
    model.save_weights('{}.model.weigthts.h5'.format(name))


def load(name):
    with open('{}.model.json'.format(name),'r') as f:
        model = model_from_json(f.read())
    model.load_weights('{}.model.weigthts.h5'.format(name))
    return model
