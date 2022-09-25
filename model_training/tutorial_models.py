import sys
sys.path.append("../")

from model_training.network import *
from model_training.layer import *

def dnn(input_shape=(None, 10), nb_classes=2):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    # layers = [Linear(64),
    #           activation(),
    #           Linear(32),
    #           activation(),
    #           Linear(16),
    #           activation(),
    #           Linear(8),
    #           activation(),
    #           Linear(4),
    #           activation(),
    #           Linear(nb_classes),
    #           Softmax()]

    layers = [Linear(64),
              activation(),
              Linear(32),
              activation(),
              Linear(16),
              activation(),
              Linear(8),
              activation(),
              Linear(4),
              activation(),
              Flatten(),
              Linear(nb_classes),
              Softmax()
              ]

    model = MLP(layers, input_shape)
    return model
