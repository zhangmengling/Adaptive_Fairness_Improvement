import numpy as np
import sys
sys.path.append("../")

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from tensorflow.python.platform import flags
from data.census import census_data
from data.bank import bank_data
from data.credit import credit_data
from data.compas_two_year import compas_data
from data.law_school import law_school_data
from data.communities import communities_data
from data.segment import segment_data
from data.adult import adult_data
from utils.utils_tf import model_train, model_eval
from model_training.tutorial_models import dnn

FLAGS = flags.FLAGS

def training(dataset, model_path):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {"adult": adult_data, "census": census_data}

    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    print("-->x, y, input_shape, nb_classes", X, Y, input_shape, nb_classes)
    tf.set_random_seed(999)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    print("-->preds:", preds)

    # training parameters
    train_params = {
        'nb_epochs': 1000,
        'batch_size': 128,
        'learning_rate': 0.01,
        'train_dir': model_path,
        'filename': 'test.model'
    }

    # training procedure
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2021, 8, 19])
    model_train(sess, x, y, preds, X, Y, args=train_params, save=True)  # rng=rng

    # evaluate the accuracy of trained model
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

def main(argv=None):
    training(dataset = FLAGS.dataset,
             model_path = FLAGS.model_path)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_string("model_path", "../models/census_income_prefrentialsampling/", "the name of path for saving model") #census-->credit

    tf.app.run()