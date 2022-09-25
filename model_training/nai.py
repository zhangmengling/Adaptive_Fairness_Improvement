from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")
sys.path.append("../../")

# from adf_model.model_operation import model_load, model_eval
from utils.config import census, credit, bank
from utils.utils_tf import model_train, model_eval
from baseline.lime import lime_tabular
from model.tutorial_models import dnn
from data.census import census_data
from data.credit import credit_data
from data.bank import bank_data
from utils.utils_tf import model_argmax
from tutorial.utils import cluster
# from data.data import get_data, get_shape
# from guardai_util.configs import path

FLAGS = flags.FLAGS


def nai(dataset, sens_param, ration=0.1, threshold=0.9, batch_size=256, epoch=9):
    tf.reset_default_graph()
    data = {"census":census_data, "credit":credit_data, "bank":bank_data}
    data_config = {"census":census, "credit":credit, "bank":bank}
    # data preprocessing
    X, Y, input_shape, nb_classes = data[dataset](sens_param)
    X_original = np.array(X)
    Y_original = np.array(Y)

    # model structure
    model = dnn(input_shape, nb_classes)

    # tf operation
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    preds = model(x)

    saver = tf.train.Saver()
    saver.restore(sess, '../models/' + dataset + '/999/test.model')

    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples for original model: {0}'.format(accuracy))

    unique_neurons = 0
    for layer in model.layers:
        if "Conv2D" in layer.__class__.__name__:
            unique_neurons += layer.output_channels
        elif "Linear" in layer.__class__.__name__:
            unique_neurons += layer.num_hid
            # as for BN, it changes when Conv2D changes, so would make sure to invert the activation
    indices = np.random.choice(unique_neurons, int(unique_neurons * ration), replace=False)

    neurons_count = 0
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if "Conv2D" in layer.__class__.__name__:
            unique_neurons_layer = layer.output_channels
            mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
            if mutated_neurons:
                mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                kernel_shape = layer.kernel_shape
                mutated_metrix = np.asarray([1.0] * unique_neurons_layer)
                mutated_metrix[mutated_neurons] = -1.0
                mutated_kernel = np.asarray([[[list(mutated_metrix)]] * kernel_shape[1]] * kernel_shape[0])
                update_kernel = tf.assign(layer.kernels, mutated_kernel * sess.run(layer.kernels))
                update_bias = tf.assign(layer.b, mutated_metrix * sess.run(layer.b))
                sess.run(update_kernel)
                sess.run(update_bias)
                if "BN" in model.layers[i + 1].__class__.__name__:
                    layer = model.layers[i + 1]
                    update_beta = tf.assign(layer.beta, mutated_metrix * sess.run(layer.beta))
                    update_moving_mean = tf.assign(layer.moving_mean, mutated_metrix * sess.run(layer.moving_mean))
                    sess.run(update_beta)
                    sess.run(update_moving_mean)
            neurons_count += unique_neurons_layer
        elif "Linear" in layer.__class__.__name__:
            unique_neurons_layer = layer.num_hid
            mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
            if mutated_neurons:
                mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                input_shape = layer.input_shape[1]
                mutated_metrix = np.asarray([1.0] * unique_neurons_layer)
                mutated_metrix[mutated_neurons] = -1.0
                mutated_weight = np.asarray([list(mutated_metrix)] * input_shape)
                weight = sess.run(layer.W)
                update_weight = tf.assign(layer.W, mutated_weight * weight)
                update_bias = tf.assign(layer.b, mutated_metrix * sess.run(layer.b))
                sess.run(update_weight)
                sess.run(update_bias)
            neurons_count += unique_neurons_layer

    mutated_accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples for mutated model: {0}'.format(mutated_accuracy))

    # if mutated_accuracy >= threshold * accuracy:
    #     train_dir = os.path.join(path.mu_model_path, 'nai', dataset + '_' + model_name, '0')
    #     if not os.path.exists(train_dir):
    #         os.makedirs(train_dir)
    #     save_path = os.path.join(train_dir, datasets + '_' + model_name + '.model')
    #     saver = tf.train.Saver()
    #     saver.save(sess, save_path)

    sess.close()

def main(argv=None):
    nai(dataset=FLAGS.dataset,
       sens_param=FLAGS.sens_param,
        ration=FLAGS.ration,
        threshold=FLAGS.threshold)


if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'The target datasets.')
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race.')
    flags.DEFINE_float('ration', 0.1, 'The ration of mutated neurons.')
    flags.DEFINE_float('threshold', 0.9, 'The threshold of accuacy compared with original.')

    tf.app.run()