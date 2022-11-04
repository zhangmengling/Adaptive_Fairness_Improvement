import autograd.numpy as np
import argparse
import json
import ast

import json_parser
# from json_parser import parse
from utils import *


def add_assertion(args, spec):
    assertion = dict()

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = '1e9' # eps is not necessary in this experiment

    spec['assert'].update(assertion)


def add_solver(args, spec):
    solver = dict()

    solver['algorithm'] = args.algorithm
    print("-->solver.algorithm:", args.algorithm)
    if args.algorithm == 'sprt':
        solver['threshold'] = str(args.threshold)
        solver['alpha'] = '0.05'
        solver['beta'] = '0.05'
        solver['delta'] = '0.005'

    spec['solver'] = solver


def main():
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str,
                        help='the chosen algorithm')
    parser.add_argument('--threshold', type=float,
                        help='the threshold in sprt')
    parser.add_argument('--eps', type=float,
                        help='the distance value')
    parser.add_argument('--dataset', type=str,
                        help='the data set for fairness experiments')

    parser.add_argument('--solve_option', type=str, default="solve_fairness", 
                        help='the specification file')
    parser.add_argument('--num_tests', type=int, default=100,
                        help='maximum number of tests')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = json_parser.parse(spec)

    print("-->model", model)

    print("-->assertion", assertion)

    if args.dataset == 'bank':
        pathX = 'benchmark/causal/bank/data/'
        pathY = 'benchmark/causal/bank/data/labels.txt'
    elif args.dataset == 'census':
        pathX = 'benchmark/causal/census/data/'
        pathY = 'benchmark/causal/census/data/labels.txt'
    elif args.dataset == 'credit':
        pathX = 'benchmark/causal/credit/data/'
        pathY = 'benchmark/causal/credit/data/labels.txt'
    elif args.dataset == 'compas':
        pathX = 'benchmark/causal/compas/data/'
        pathY = 'benchmark/causal/compas/data/labels.txt'
    elif args.dataset == 'FairSquare':
        pathX = 'benchmark/causal/FairSquare/data/'
        pathY = 'benchmark/causal/FairSquare/data/labels.txt'
    elif args.dataset == 'wiki':
        pathX = 'benchmark/rnn/data/wiki/'
        pathY = 'benchmark/rnn/data/wiki/labels.txt'
    elif args.dataset == 'imdb_train':
        pathX = 'benchmark/rnn_fairness/data/imdb_train/'
        pathY = "benchmark/rnn_fairness/data/imdb_train/labels.txt"
    elif args.dataset == 'imdb_test':
        pathX = 'benchmark/rnn_fairness/data/imdb_test/'
        pathY = "benchmark/rnn_fairness/data/imdb_test/labels.txt"


    for i in range(100):
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        x0 = np.array(ast.literal_eval(read(assertion['x0'])))

        shape_x0 = (int(x0.size / 50), 50)

        model.shape = shape_x0
        # model.lower = np.full(x0.size, lower)
        # model.upper = np.full(x0.size, upper)

        # print("-->model type", type(model))
        # model.get_layer_info()
        # print("-->model.shape", model.shape)

        output_x0 = model.apply(x0)
        lbl_x0 = np.argmax(output_x0, axis=1)[0]
        print("-->lbl_x0", lbl_x0)


    y0s = np.array(ast.literal_eval(read(pathY)))

    assertion['x0'] = pathX + 'data' + str(0) + '.txt'
    print("-->assertion", assertion)
    solver.solve(model, assertion)

    print('\n============================\n')

if __name__ == '__main__':
    main()
