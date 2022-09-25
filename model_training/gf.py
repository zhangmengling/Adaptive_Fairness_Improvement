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
from model_training.tutorial_models import dnn
from data.census import census_data
from data.credit import credit_data
from data.bank import bank_data
from utils.utils_tf import model_argmax
from tutorial.utils import cluster
# from adf_data.data import get_data, get_shape
# from guardai_util.configs import path

FLAGS = flags.FLAGS


def gf(dataset, sens_param, ration=0.1, threshold=0.9, batch_size=256, epoch=9):
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

    num_weights = 0
    for layer in model.layers:
        if "Conv2D" in layer.__class__.__name__:
            shape = layer.kernels.shape
            num_weights += int(shape[0] * shape[1] * shape[2] * shape[3])
        elif "BN" in layer.__class__.__name__:
            shape = layer.gamma.shape
            num_weights += int(shape[0])
        elif "Linear" in layer.__class__.__name__:
            shape = layer.W.shape
            num_weights += int(shape[0] * shape[1])
    indices = np.random.choice(num_weights, int(num_weights * ration), replace=False)

    weights_count = 0
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if "Conv2D" in layer.__class__.__name__:
            shape = layer.kernels.shape
            num_weights_layer = int(shape[0] * shape[1] * shape[2] * shape[3])
            mutated_indices = set(indices) & set(np.arange(weights_count, weights_count + num_weights_layer))
            if mutated_indices:
                mutated_indices = np.array(list(mutated_indices)) - weights_count
                current_weights = sess.run(layer.kernels).reshape(-1)
                avg_weights = np.mean(current_weights)
                std_weights = np.std(current_weights)
                mutated_weights = np.random.normal(avg_weights, std_weights, mutated_indices.size)
                current_weights[mutated_indices] = mutated_weights
                update_weights = tf.assign(layer.kernels, current_weights.reshape(shape))
                sess.run(update_weights)
            weights_count += num_weights_layer
        elif "BN" in layer.__class__.__name__:
            shape = layer.gamma.shape
            num_weights_layer = int(shape[0])
            mutated_indices = set(indices) & set(np.arange(weights_count, weights_count + num_weights_layer))
            if mutated_indices:
                mutated_indices = np.array(list(mutated_indices)) - weights_count
                current_weights = sess.run(layer.gamma).reshape(-1)
                avg_weights = np.mean(current_weights)
                std_weights = np.std(current_weights)
                mutated_weights = np.random.normal(avg_weights, std_weights, mutated_indices.size)
                current_weights[mutated_indices] = mutated_weights
                update_weights = tf.assign(layer.gamma, current_weights.reshape(shape))
                sess.run(update_weights)
            weights_count += num_weights_layer
        elif "Linear" in layer.__class__.__name__:
            shape = layer.W.shape
            num_weights_layer = int(shape[0] * shape[1])
            mutated_indices = set(indices) & set(np.arange(weights_count, weights_count + num_weights_layer))
            if mutated_indices:
                mutated_indices = np.array(list(mutated_indices)) - weights_count
                current_weights = sess.run(layer.W).reshape(-1)
                avg_weights = np.mean(current_weights)
                std_weights = np.std(current_weights)
                mutated_weights = np.random.normal(avg_weights, std_weights, mutated_indices.size)
                current_weights[mutated_indices] = mutated_weights
                update_weights = tf.assign(layer.W, current_weights.reshape(shape))
                sess.run(update_weights)
            weights_count += num_weights_layer

    mutated_accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples for mutated model: {0}'.format(mutated_accuracy))

    # if mutated_accuracy >= threshold * accuracy:
    #     train_dir = os.path.join(path.mu_model_path, 'gf', datasets + '_' + model_name, '0')
    #     if not os.path.exists(train_dir):
    #         os.makedirs(train_dir)
    #     save_path = os.path.join(train_dir, datasets + '_' + model_name + '.model')
    #     saver = tf.train.Saver()
    #     saver.save(sess, save_path)

    sess.close()

def main(argv=None):
    gf(dataset=FLAGS.dataset,
       sens_param=FLAGS.sens_param,
        ration=FLAGS.ration,
        threshold=FLAGS.threshold)


if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'The target datasets.')
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race.')
    flags.DEFINE_float('ration', 0.1, 'The ration of mutated neurons.')
    flags.DEFINE_float('threshold', 0.9, 'The threshold of accuacy compared with original.')

    tf.app.run()