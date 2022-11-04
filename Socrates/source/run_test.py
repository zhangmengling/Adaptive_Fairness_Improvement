import autograd.numpy as np
import argparse
import json
import ast

from json_parser import parse
from utils import *

Threshold = 0.5


def add_assertion(args, spec):
    assertion = dict()

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = str(args.eps)

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    solver['algorithm'] = args.algorithm
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
                        help='the data set for rnn experiments')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    # model, assertion = parse(spec)

    model, assertion, solver, display = parse(spec)
    print("-->spec")
    print(model)
    print(assertion)
    print(solver)
    print(display)
    # print("-->get_layer_info")
    # print(model.get_layer_info())

    lower = model.lower[0]
    upper = model.upper[0]
    print("lower", lower)
    print("upper", upper)

    if args.dataset == 'jigsaw':
        pathX = 'benchmark/rnn/data/jigsaw/'
        pathY = 'benchmark/rnn/data/jigsaw/labels.txt'
    elif args.dataset == 'wiki':
        pathX = 'benchmark/rnn/data/wiki/'
        pathY = 'benchmark/rnn/data/wiki/labels.txt'
    elif args.dataset == 'imdb_train':
        pathX = 'benchmark/rnn_fairness/data/imdb_train/'
        pathY = "benchmark/rnn_fairness/data/imdb_train/labels.txt"
    elif args.dataset == 'imdb_test':
        pathX = 'benchmark/rnn_fairness/data/imdb_test/'
        pathY = "benchmark/rnn_fairness/data/imdb_test/labels.txt"

    sensitive_path = "benchmark/rnn_fairness/data/sensitive_terms.txt"
    f = open(sensitive_path, "r")
    terms = []
    for line in f.readlines():
        line = line.strip('\n')
        terms.append(line)
    print("-->terms", terms)

    terms = ['lesbian', 'gay', 'bisexual', 'transgender', 'trans', 'queer', 'lgbt', 'lgbtq', 'homosexual', 'straight',
             'heterosexual', 'male', 'female', 'nonbinary', 'african', 'african american', 'black', 'white', 'european',
             'hispanic', 'latino', 'latina', 'latinx', 'mexican', 'canadian', 'american', 'asian', 'indian', 'middle eastern',
             'chinese', 'japanese', 'christian', 'muslim', 'jewish', 'buddhist', 'catholic', 'protestant', 'sikh', 'taoist',
             'atheist', 'old', 'older', 'young', 'younger', 'teenage', 'millenial', 'middle aged', 'elderly', 'blind', 'deaf', 'paralyzed']
    terms = ['gay']
    for term in terms:
        pathX = 'benchmark/rnn_fairness/data/imdb_term/' + term + "/"
        pathY = 'benchmark/rnn_fairness/data/imdb_term/' + term + "/labels.txt"


    y0s = np.array(ast.literal_eval(read(pathY)))

    l_pass = 0
    l_fail = 0

    length = len(y0s)

    all_lbl_x0 = []

    for i in range(1):  # 100
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        print("-->assertion['x0']", assertion['x0'])
        x0 = np.array(eval(read(assertion['x0'])))

        shape_x0 = x0.shape

        model.shape = shape_x0
        model.lower = np.full(shape_x0, lower)
        model.upper = np.full(shape_x0, upper)

        output_x0 = model.apply_1(x0)
        print("-->output_x0", output_x0)
        print(output_x0.shape)
        if output_x0[0][0] >= Threshold:
            lbl_x0 = 1
        else:
            lbl_x0 = 0
        # lbl_x0 = np.argmax(output_x0, axis=1)[0]

        print('Data {}\n'.format(i))
        # print('x0 = {}'.format(x0))
        # print(len(x0))
        print('output_x0 = {}'.format(output_x0))
        print('lbl_x0 = {}'.format(lbl_x0))
        print('y0 = {}\n'.format(y0s[i]))

        if lbl_x0 == y0s[i]:
            print('Run at data {}\n'.format(i))
            # solver.solve(model, assertion)
            l_pass = l_pass + 1
        else:
            print('Skip at data {}'.format(i))
            l_fail = l_fail + 1

        all_lbl_x0.append(lbl_x0)

        print('\n============================\n')

    print("Accuracy of ori network: %f.\n" % (l_pass / (l_pass + l_fail)))

    print("Count 1:", all_lbl_x0.count(1))

    solver.solve(model, assertion, term)

if __name__ == '__main__':
    main()
