# code for IDS with deterministic local search
# requires installation of python package apyori: https://pypi.org/project/apyori/

import numpy as np
import pandas as pd
import math
from fairness_test.apyori import apriori

import itertools

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import random
from random import choice
import matplotlib.pyplot as plt

from tensorflow.python.platform import flags
from data.census import census_data, census_reweighting_data
from data.adult import adult_data
from data.bank import bank_data
from data.credit import credit_data
from data.compas_two_year import compas_data
from data.law_school import law_school_data
from data.communities import communities_data
from utils.utils_tf import model_train, model_eval
from model_training.tutorial_models import dnn

from utils.config import census, census_income, credit, bank, compas, law_school, communities

from pandas.core.frame import DataFrame

from utils.utils_tf import model_prediction, model_argmax
from csv import reader

from itertools import combinations, permutations

FLAGS = flags.FLAGS

def calculate_probability(label_list):
    positive_num = 0
    for label in label_list:
        if label == 1:
            positive_num += 1
    return positive_num/len(label_list)



data = {"census_income": census_reweighting_data, "credit": credit_data, "bank": bank_data, "compas": compas_data,
        "law_school": law_school_data, "communities": communities_data}
data_config = {"census": census, "census_income": census_income, "credit": credit, "bank": bank, "compas": compas,
               "law_school": law_school, "communities": communities}

dataset = "census_income"
X, Y, input_shape, nb_classes = data[dataset]()

config = tf.ConfigProto()
conf = data_config[dataset]
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
x = tf.placeholder(tf.float32, shape=input_shape)
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
model = dnn(input_shape, nb_classes)
preds = model(x)
# print("-->preds ", preds)
saver = tf.train.Saver()
# original model
# saver.restore(sess, "../models/census/test.model")
# retrained model
saver.restore(sess, "../models/census_income_uniformsampling/999/test.model")
# grad_0 = gradient_graph(x, preds)
# tfops = tf.sign(grad_0)

eval_params = {'batch_size': 128}
accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
print("-->accuracy:", accuracy)

labels = []
for sample in X:
    probs = model_prediction(sess, x, preds, np.array([sample]))[0]  # n_probs: prediction vector
    model_label = np.argmax(probs)  # GET index of max value in n_probs
    labels.append(model_label)

print("-->labels", labels)
print(calculate_probability(labels))
