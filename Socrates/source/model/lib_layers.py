import autograd.numpy as np

import sys
import os
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

import tensorflow as tf
from solver.refinement_impl import Poly
from utils import *
from poly_utils import *


class Layer:
    def apply(self, x):
        return x

    def reset(self):
        pass

    def copy(self):
        pass


class Function(Layer):
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.func = get_func(name, params)
        self.keep_prob = 0.7

    def copy(self):
        new_layer = Function(self.name, self.params)
        return new_layer

    def apply(self, x):
        return self.func(x)

    def apply_poly(self, x_poly, lst_poly):
        res = Poly()

        no_neurons = len(x_poly.lw)

        res.lw = np.zeros(no_neurons)
        res.up = np.zeros(no_neurons)

        res.le = np.zeros([no_neurons, no_neurons + 1])
        res.ge = np.zeros([no_neurons, no_neurons + 1])

        res.shape = x_poly.shape
        res.is_activation = True

        if self.name == 'relu':
            for i in range(no_neurons):
                if x_poly.up[i] <= 0:
                    pass
                elif x_poly.lw[i] >= 0:
                    res.le[i,i] = 1
                    res.ge[i,i] = 1

                    res.lw[i] = x_poly.lw[i]
                    res.up[i] = x_poly.up[i]
                else:
                    res.le[i,i] = x_poly.up[i] / (x_poly.up[i] - x_poly.lw[i])
                    res.le[i,-1] = - x_poly.up[i] * x_poly.lw[i] / (x_poly.up[i] - x_poly.lw[i])

                    lam = 0 if x_poly.up[i] <= -x_poly.lw[i] else 1

                    res.ge[i,i] = lam
                    res.lw[i] = 0 # it seems safe to set lw = 0 anyway
                    # res.lw[i] = lam * x_poly.lw[i] # notice: mnist_relu_5_10.tf
                    res.up[i] = x_poly.up[i]

        elif self.name == 'sigmoid':
            res.lw = sigmoid(x_poly.lw)
            res.up = sigmoid(x_poly.up)

            for i in range(no_neurons):
                if x_poly.lw[i] == x_poly.up[i]:
                    res.le[i][-1] = res.lw[i]
                    res.ge[i][-1] = res.lw[i]
                else:
                    if x_poly.lw[i] > 0:
                        lam1 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        if x_poly.up[i] <= 0:
                            lam2 = lam1
                        else:
                            ll = sigmoid(x_poly.lw[i]) * (1 - sigmoid(x_poly.lw[i]))
                            uu = sigmoid(x_poly.up[i]) * (1 - sigmoid(x_poly.up[i]))
                            lam2 = min(ll, uu)
                    else:
                        ll = sigmoid(x_poly.lw[i]) * (1 - sigmoid(x_poly.lw[i]))
                        uu = sigmoid(x_poly.up[i]) * (1 - sigmoid(x_poly.up[i]))
                        lam1 = min(ll, uu)
                        if x_poly.up[i] <= 0:
                            lam2 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        else:
                            lam2 = lam1

                    res.ge[i,i] = lam1
                    res.ge[i,-1] = res.lw[i] - lam1 * x_poly.lw[i]

                    res.le[i,i] = lam2
                    res.le[i,-1] = res.up[i] - lam2 * x_poly.up[i]

        elif self.name == 'softmax':
            res.lw = softmax(x_poly.lw)
            res.up = softmax(x_poly.up)

            for i in range(no_neurons):
                if x_poly.lw[i] == x_poly.up[i]:
                    res.le[i][-1] = res.lw[i]
                    res.ge[i][-1] = res.lw[i]
                else:
                    if x_poly.lw[i] > 0:
                        lam1 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        if x_poly.up[i] <= 0:
                            lam2 = lam1
                        else:
                            ll = softmax(x_poly.lw[i]) * (1 - softmax(x_poly.lw[i]))
                            uu = softmax(x_poly.up[i]) * (1 - softmax(x_poly.up[i]))
                            lam2 = min(ll, uu)
                    else:
                        ll = softmax(x_poly.lw[i]) * (1 - softmax(x_poly.lw[i]))
                        uu = softmax(x_poly.up[i]) * (1 - softmax(x_poly.up[i]))
                        lam1 = min(ll, uu)
                        if x_poly.up[i] <= 0:
                            lam2 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        else:
                            lam2 = lam1

                    res.ge[i,i] = lam1
                    res.ge[i,-1] = res.lw[i] - lam1 * x_poly.lw[i]

                    res.le[i,i] = lam2
                    res.le[i,-1] = res.up[i] - lam2 * x_poly.up[i]

        elif self.name == 'tanh':
            res.lw = tanh(x_poly.lw)
            res.up = tanh(x_poly.up)

            for i in range(no_neurons):
                if x_poly.lw[i] == x_poly.up[i]:
                    res.le[i][-1] = res.lw[i]
                    res.ge[i][-1] = res.lw[i]
                else:
                    if x_poly.lw[i] > 0:
                        lam1 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        if x_poly.up[i] <= 0:
                            lam2 = lam1
                        else:
                            ll = 1 - pow(tanh(x_poly.lw[i]), 2)
                            uu = 1 - pow(tanh(x_poly.up[i]), 2)
                            lam2 = min(ll, uu)
                    else:
                        ll = 1 - pow(tanh(x_poly.lw[i]), 2)
                        uu = 1 - pow(tanh(x_poly.up[i]), 2)
                        lam1 = min(ll, uu)
                        if x_poly.up[i] <= 0:
                            lam2 = (res.up[i] - res.lw[i]) / (x_poly.up[i] - x_poly.lw[i])
                        else:
                            lam2 = lam1

                    res.ge[i,i] = lam1
                    res.ge[i,-1] = res.lw[i] - lam1 * x_poly.lw[i]

                    res.le[i,i] = lam2
                    res.le[i,-1] = res.up[i] - lam2 * x_poly.up[i]
        
        elif self.name == 'reshape':
            res.lw = x_poly.lw.copy()
            res.up = x_poly.up.copy()

            res.le = np.eye(no_neurons + 1)[:-1]
            res.ge = np.eye(no_neurons + 1)[:-1]

            res.shape = self.params[0]

        return res

    def is_poly_exact(self):
        if self.func == relu or self.func == sigmoid or self.func == tanh:
            return False
        else:
            return True

    def is_activation_layer(self):
        return True

    def is_linear_layer(self):
        return False

    def get_number_neurons(self):
        return None


class Linear(Layer):
    def __init__(self, weights, bias, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)
        self.func = get_func(name, None)

    def copy(self):
        new_layer = Linear(np.zeros((1,1)), np.zeros(1), None)

        new_layer.weights = self.weights.copy()
        new_layer.bias = self.bias.copy()
        new_layer.func = self.func

        return new_layer

    def apply(self, x):
        if self.func == None:
            return x @ self.weights + self.bias
        else:
            return self.func(x @ self.weights + self.bias)


    def get_weight(self):
        return self.weights


    def apply_poly(self, x_poly, lst_poly):
        assert self.func == None, "self.func should be None"

        weights = self.weights.transpose(1, 0)
        bias = self.bias.transpose(1, 0)

        no_neurons = len(bias)

        res = Poly()

        res.lw = np.zeros(no_neurons)
        res.up = np.zeros(no_neurons)

        res.le = np.concatenate([weights, bias], axis=1)
        res.ge = np.concatenate([weights, bias], axis=1)

        res.shape = (1, no_neurons)

        res.back_substitute(lst_poly)

        return res

    def is_poly_exact(self):
        return True

    def is_activaiton_layer(self):
        return False

    def is_linear_layer(self):
        return True

    def get_number_neurons(self):
        return len(self.bias[0])


class BasicRNN(Layer):
    def __init__(self, weights, bias, h0, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

        self.func = get_func(name, None)

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        if self.func == None:
            self.h_t = x @ self.weights + self.bias
        else:
            self.h_t = self.func(x @ self.weights + self.bias)

        return self.h_t

    def reset(self):
        self.h_t = self.h_0


class LSTM(Layer):
    def __init__(self, weights, bias, h0, c0):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.c_0 = c0.reshape(-1, c0.size)

        self.h_t = h0.reshape(-1, h0.size)
        self.c_t = c0.reshape(-1, c0.size)

    def apply(self, x):
        x = np.concatenate((x, self.h_t), axis=1)

        gates = x @ self.weights + self.bias

        i, j, f, o = np.split(gates, 4, axis=1)

        self.c_t = self.c_t * sigmoid(f) + sigmoid(i) * tanh(j)
        self.h_t = sigmoid(o) * tanh(self.c_t)

        return self.h_t

    def reset(self):
        self.h_t = self.h_0
        self.c_t = self.c_0

    # def get_weight(self):
    #     return self.weights


class LSTM_1(Layer):
    def __init__(self, units, weights_0, weights_1, bias, return_sequences):
        self.unit = units
        self.W_x = weights_0
        self.W_h = weights_1
        self.bias = bias
        self.return_sequences = return_sequences

        # print("-->W_x", self.W_x.shape)
        # print("-->W_h", self.W_h.shape)

    def activation_sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def activation_tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def apply(self, x):
        # input gate
        w_ix = self.W_x[:, :self.unit]
        w_ih = self.W_h[:, :self.unit]
        b_i = self.bias[:self.unit]

        # forget gate
        w_fx = self.W_x[:, self.unit:self.unit * 2]
        w_fh = self.W_h[:, self.unit:self.unit * 2]
        b_f = self.bias[self.unit:self.unit * 2]

        # current state
        w_cx = self.W_x[:, self.unit * 2:self.unit * 3]
        w_ch = self.W_h[:, self.unit * 2:self.unit * 3]
        b_c = self.bias[self.unit * 2:self.unit * 3]

        # output gate
        w_ox = self.W_x[:, self.unit * 3:self.unit * 4]
        w_oh = self.W_h[:, self.unit * 3:self.unit * 4]
        b_o = self.bias[self.unit * 3:self.unit * 4]

        if self.return_sequences == True:
            # print(self.return_sequences)
            # f_t = self.activation_sigmoid(np.dot(x, w_fx) + b_f)
            # i_t = self.activation_sigmoid(np.dot(x, w_ix) + b_i)
            # c_hat_t = self.activation_tanh(np.dot(x, w_cx) + b_c)
            # o_t = self.activation_sigmoid(np.dot(x, w_ox) + b_o)
            # c_t = np.multiply(i_t, c_hat_t)
            # self.h_t = np.multiply(o_t, self.activation_tanh(c_t))

            # test2_t1 = test2[:, 0, :]
            #
            # f_t1 = activation_relu(np.dot(test2_t1, w_fx) + b_f)
            # i_t1 = activation_relu(np.dot(test2_t1, w_ix) + b_i)
            # c_hat_t1 = activation_relu(np.dot(test2_t1, w_cx) + b_c)
            # o_t1 = activation_relu(np.dot(test2_t1, w_ox) + b_o)
            # c_t1 = np.multiply(i_t1, c_hat_t1)
            # h_t1 = np.multiply(o_t1, activation_relu(c_t1))
            #
            # test2_t2 = test2[:, 1, :]
            #
            # f_t2 = activation_relu(np.dot(test2_t2, w_fx) + np.dot(h_t1, w_fh) + b_f)
            # i_t2 = activation_relu(np.dot(test2_t2, w_ix) + np.dot(h_t1, w_ih) + b_i)
            # c_hat_t2 = activation_relu(np.dot(test2_t2, w_cx) + np.dot(h_t1, w_ch) + b_c)
            # o_t2 = activation_relu(np.dot(test2_t2, w_ox) + np.dot(h_t1, w_oh) + b_o)
            # c_t2 = np.multiply(i_t2, c_hat_t2) + np.multiply(f_t2, c_t1)
            # h_t2 = np.multiply(o_t2, activation_relu(c_t2))

            h_sequences = []
            first_x = np.array(x[:, 0, :])
            f_t = self.activation_sigmoid(np.dot(first_x, w_fx) + b_f)
            i_t = self.activation_sigmoid(np.dot(first_x, w_ix) + b_i)
            c_hat_t = self.activation_tanh(np.dot(first_x, w_cx) + b_c)
            o_t = self.activation_sigmoid(np.dot(first_x, w_ox) + b_o)
            c_t0 = np.multiply(i_t, c_hat_t)
            h_t0 = np.multiply(o_t, self.activation_tanh(c_t0))
            h_sequences.append(h_t0)

            for i in range(1, x.shape[1]):
                one_x = np.array(x[:, i, :])
                f_t = self.activation_sigmoid(np.dot(one_x, w_fx) + np.dot(h_t0, w_fh) + b_f)
                i_t = self.activation_sigmoid(np.dot(one_x, w_ix) + np.dot(h_t0, w_ih) + b_i)
                c_hat_t = self.activation_tanh(np.dot(one_x, w_cx) + np.dot(h_t0, w_ch) + b_c)
                o_t = self.activation_sigmoid(np.dot(one_x, w_ox) + np.dot(h_t0, w_oh) + b_o)
                c_t = np.multiply(i_t, c_hat_t) + np.multiply(f_t, c_t0)
                h_t = np.multiply(o_t, self.activation_tanh(c_t))
                h_t0 = h_t
                c_t0 = c_t
                h_sequences.append(h_t0)

            self.h_t = np.array(h_sequences)
            return np.array(self.h_t)
        else:
            # print(self.return_sequences)

            first_x = np.array(x[:, 0, :])
            f_t = self.activation_sigmoid(np.dot(first_x, w_fx) + b_f)
            i_t = self.activation_sigmoid(np.dot(first_x, w_ix) + b_i)
            c_hat_t = self.activation_tanh(np.dot(first_x, w_cx) + b_c)
            o_t = self.activation_sigmoid(np.dot(first_x, w_ox) + b_o)
            c_t0 = np.multiply(i_t, c_hat_t)
            h_t0 = np.multiply(o_t, self.activation_tanh(c_t0))


            for i in range(1, x.shape[1]):
                one_x = np.array(x[:, i, :])
                f_t = self.activation_sigmoid(np.dot(one_x, w_fx) + np.dot(h_t0, w_fh) + b_f)
                i_t = self.activation_sigmoid(np.dot(one_x, w_ix) + np.dot(h_t0, w_ih) + b_i)
                c_hat_t = self.activation_tanh(np.dot(one_x, w_cx) + np.dot(h_t0, w_ch) + b_c)
                o_t = self.activation_sigmoid(np.dot(one_x, w_ox) + np.dot(h_t0, w_oh) + b_o)
                c_t = np.multiply(i_t, c_hat_t) + np.multiply(f_t, c_t0)
                h_t = np.multiply(o_t, self.activation_tanh(c_t))
                h_t0 = h_t
                c_t0 = c_t

            self.h_t = h_t
            # print("-->self.h_t.shape", self.h_t.shape)

            # print("-->o_t", o_t, o_t.shape)
            # print("-->c_hat_t", c_hat_t, c_hat_t.shape)
            # print("-->c_t", c_t, c_t.shape)
            # print("-->self.activation_tanh(c_t)", self.activation_tanh(c_t), self.activation_tanh(c_t).shape)
            return np.array(self.h_t)


class GRU(Layer):
    def __init__(self, gate_weights, candidate_weights,
            gate_bias, candidate_bias, h0):
        self.gate_weights = gate_weights.transpose(1, 0)
        self.gate_bias = gate_bias.reshape(-1, gate_bias.size)

        self.candidate_weights = candidate_weights.transpose(1, 0)
        self.candidate_bias = candidate_bias.reshape(-1, candidate_bias.size)

        self.h_0 = h0.reshape(-1, h0.size)
        self.h_t = h0.reshape(-1, h0.size)

    def apply(self, x):
        gx = np.concatenate((x, self.h_t), axis=1)

        gates = sigmoid(gx @ self.gate_weights + self.gate_bias)

        r, u = np.split(gates, 2, axis=1)
        r = r * self.h_t

        cx = np.concatenate((x, r), axis=1)
        c = tanh(cx @ self.candidate_weights + self.candidate_bias)

        self.h_t = (1 - u) * c + u * self.h_t

        return self.h_t

    def reset(self):
        self.h_t = self.h_0


class Conv1d(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_l = self.filters.shape # 2, 3, 4
        f = self.filters.reshape(f_n, -1)  # 2, 12

        b = self.bias.reshape(f_n, -1)  # 2, 1

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p)), mode='constant')
        x_n, x_c, x_l = x_pad.shape  # 1, 3, 10

        res_l = int((x_l - f_l) / self.stride) + 1
        size = f_c * f_l

        c_idx, l_idx = index1d(x_c, self.stride, (f_l), (x_l))

        res = x_pad[:, c_idx, l_idx] # 1, 12, 10
        res = res.reshape(size, -1) # 12, 10

        res = f @ res + b # 2, 10
        # print(res.shape)
        res = res.reshape(1, f_n, res_l) # 1, 2, 10

        return res

    # def get_weight(self):
    #     return self.weights


class Conv1d_1(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_l = self.filters.shape # 128, 100, 5
        # print("-->f_n, f_c, f_l", f_n, f_c, f_l)
        f = self.filters.reshape(f_n, -1)  # 2, 12
        # f = self.filters

        b = self.bias.reshape(f_n, -1)  # 2, 1
        # b = self.bias

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p)), mode='constant')
        x_n, x_c, x_l = x_pad.shape  # 1, 128, 14
        # print("-->x_n, x_c, x_l", x_n, x_c, x_l)

        res_l = int((x_l - f_l) / self.stride) + 1
        size = f_c * f_l

        c_idx, l_idx = index1d(x_c, self.stride, (f_l), (x_l))
        # print("-->c_idx, l_idx", c_idx, l_idx)

        res = x_pad[:, c_idx, l_idx] # 1, 12, 10
        res = res.reshape(size, -1) # 12, 10

        res = f @ res + b # 2, 10
        # print("-->res", res)
        # print(res.shape)
        # res = res.reshape(1, f_n, res_l) # 1, 2, 10

        # res = res.transpose()

        return res

    # def get_weight(self):
    #     return self.weights


class Conv2d(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_h, f_w = self.filters.shape
        f = self.filters.reshape(f_n, -1)

        b = self.bias.reshape(f_n, -1)

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_h, x_w = x_pad.shape

        res_h = int((x_h - f_h) / self.stride) + 1
        res_w = int((x_w - f_w) / self.stride) + 1
        size = f_c * f_h * f_w

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, (f_h, f_w), (x_h, x_w))

        res = x_pad[:, c_idx, h_idx, w_idx]
        res = res.reshape(size, -1)

        res = f @ res + b
        res = res.reshape(1, f_n, res_h, res_w)

        return res

    def apply_poly(self, x_poly, lst_poly):
        res = Poly()

        f_n, f_c, f_h, f_w = self.filters.shape
        x_n, x_c, x_h, x_w = x_poly.shape

        x_w += 2 * self.padding
        x_h += 2 * self.padding        

        res_h = int((x_h - f_h) / self.stride) + 1
        res_w = int((x_w - f_w) / self.stride) + 1

        len_pad = x_c * x_h * x_w
        len_res = f_n * res_h * res_w

        res.lw = np.zeros(len_res)
        res.up = np.zeros(len_res)

        res.le = np.zeros([len_res, len_pad + 1])
        res.ge = np.zeros([len_res, len_pad + 1])

        res.shape = (1, f_n, res_h, res_w)

        for i in range(f_n):
            base = np.zeros([x_c, x_h, x_w])
            base[:f_c, :f_h, :f_w] = self.filters[i]
            base = np.reshape(base, -1)
            w_idx = f_w

            for j in range(res_h * res_w):
                res.le[i * res_h * res_w + j] = np.append(base, [self.bias[i]])
                res.ge[i * res_h * res_w + j] = np.append(base, [self.bias[i]])

                if w_idx + self.stride <= x_w:
                    base = np.roll(base, self.stride)
                    w_idx += self.stride
                else:
                    base = np.roll(base, self.stride * x_w - w_idx + f_w)
                    w_idx = f_w

        del_idx = []
        if self.padding > 0:
            del_idx = del_idx + list(range(self.padding * (x_w + 1)))
            mx = x_h - self.padding
            for i in range(self.padding + 1, mx):
                tmp = i * x_w
                del_idx = del_idx + list(range(tmp - self.padding, tmp + self.padding))
            del_idx = del_idx + list(range(mx * x_w - self.padding, x_h * x_w))

            tmp = np.array(del_idx)

            for i in range(1, x_c):
                offset = i * x_h * x_w
                del_idx = del_idx + list((tmp + offset).copy())

        res.le = np.delete(res.le, del_idx, 1)
        res.ge = np.delete(res.ge, del_idx, 1)

        res.le = np.ascontiguousarray(res.le)
        res.ge = np.ascontiguousarray(res.ge)

        res.back_substitute(lst_poly)

        return res

    def is_poly_exact(self):
        return True


class Conv3d(Layer):
    def __init__(self, filters, bias, stride, padding):
        self.filters = filters
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        f_n, f_c, f_d, f_h, f_w = self.filters.shape
        f = self.filters.reshape(f_n, -1)

        b = self.bias.reshape(f_n, -1)

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_d, x_h, x_w = x_pad.shape

        res_d = int((x_d - f_d) / self.stride) + 1
        res_h = int((x_h - f_h) / self.stride) + 1
        res_w = int((x_w - f_w) / self.stride) + 1
        size = f_c * f_d * f_h * f_w

        c_idx, d_idx, h_idx, w_idx = index3d(x_c, self.stride, (f_d, f_h, f_w), (x_d, x_h, x_w))

        res = x_pad[:, c_idx, d_idx, h_idx, w_idx]
        res = res.reshape(size, -1)

        res = f @ res + b
        res = res.reshape(1, f_n, res_d, res_h, res_w)

        return res


class MaxPool1d(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        k_l = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p)), mode='constant')
        x_n, x_c, x_l = x_pad.shape

        res_l = int((x_l - k_l) / self.stride) + 1

        c_idx, l_idx = index1d(x_c, self.stride, self.kernel, (x_l))

        res = x_pad[:, c_idx, l_idx]
        res = res.reshape(x_c, k_l, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_l)

        return res

    # def get_weight(self):
    #     return self.weights

class MaxPool1d_1(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel  # 5
        self.stride = stride  # 1
        self.padding = padding  # 2

    def apply(self, x):
        if len(list(x.shape)) != 3:
            x = np.array([x])

        k_l = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p)), mode='constant')
        x_n, x_c, x_l = x_pad.shape

        res_l = int((x_l - k_l) / self.stride) + 1

        c_idx, l_idx = index1d(x_c, self.stride, self.kernel, (x_l))

        res = x_pad[:, c_idx, l_idx]
        res = res.reshape(x_c, k_l, -1)

        res = np.max(res, axis=1)
        # print("-->res", res)
        # print(res.shape)
        res = res.reshape(1, x_c, res_l)
        # print("-->res", res)
        # print(res.shape)

        return res


class MaxPool2d(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        k_h, k_w = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_h, x_w = x_pad.shape

        res_h = int((x_h - k_h) / self.stride) + 1
        res_w = int((x_w - k_w) / self.stride) + 1

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, self.kernel, (x_h, x_w))

        res = x_pad[:, c_idx, h_idx, w_idx]

        res = res.reshape(x_c, k_h * k_w, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_h, res_w)

        return res

    def apply_poly(self, x_poly, lst_poly):
        res = Poly()

        k_h, k_w = self.kernel

        lw, up = x_poly.lw.copy(), x_poly.up.copy()

        lw = lw.reshape(x_poly.shape)
        up = up.reshape(x_poly.shape)

        p = self.padding
        lw_pad = np.pad(lw, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        up_pad = np.pad(up, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_h, x_w = lw_pad.shape

        res_h = int((x_h - k_h) / self.stride) + 1
        res_w = int((x_w - k_w) / self.stride) + 1

        len_pad = x_c * x_h * x_w
        len_res = x_c * res_h * res_w

        c_idx, h_idx, w_idx = index2d(x_c, self.stride, self.kernel, (x_h, x_w))

        res_lw, res_up = [], []
        mx_lw_idx_lst, mx_up_idx_lst = [], []

        for c in range(x_c):
            for i in range(res_h * res_w):

                mx_lw_val, mx_lw_idx = -1e9, None
                
                for k in range(k_h * k_w):

                    h, w = h_idx[k, i], w_idx[k, i]
                    val = lw_pad[0, c, h, w]
                    
                    if val > mx_lw_val:
                        mx_lw_val, mx_lw_idx = val, (c, h, w)

                mx_up_val, cnt = -1e9, 0

                for k in range(k_h * k_w):

                    h, w = h_idx[k, i], w_idx[k, i]
                    val = up_pad[0, c, h, w]

                    if val > mx_up_val:
                        mx_up_val = val

                    if mx_lw_idx != (c, h, w) and val > mx_lw_val:
                        cnt += 1

                res_lw.append(mx_lw_val)
                res_up.append(mx_up_val)

                mx_lw_idx_lst.append(mx_lw_idx)
                if cnt > 0: mx_up_idx_lst.append(None)
                else: mx_up_idx_lst.append(mx_lw_idx)


        res.lw = np.array(res_lw)
        res.up = np.array(res_up)

        res.le = np.zeros([len_res, len_pad + 1])
        res.ge = np.zeros([len_res, len_pad + 1])

        res.shape = (1, x_c, res_h, res_w)

        for i in range(len_res):
            c = mx_lw_idx_lst[i][0]
            h = mx_lw_idx_lst[i][1]
            w = mx_lw_idx_lst[i][2]

            idx = c * x_h * x_w + h * x_w + w
            res.ge[i, idx] = 1

            if mx_up_idx_lst[i] is None:
                res.le[i, -1] = res.up[i]
            else:
                res.le[i, idx] = 1 

        del_idx = []
        if self.padding > 0:
            del_idx = del_idx + list(range(self.padding * (x_w + 1)))
            mx = x_h - self.padding
            for i in range(self.padding + 1, mx):
                tmp = i * x_w
                del_idx = del_idx + list(range(tmp - self.padding, tmp + self.padding))
            del_idx = del_idx + list(range(mx * x_h - self.padding, x_h * x_w))

            tmp = np.array(del_idx)

            for i in range(1, x_c):
                offset = i * x_h * x_w
                del_idx = del_idx + list((tmp + offset).copy())

        res.le = np.delete(res.le, del_idx, 1)
        res.ge = np.delete(res.ge, del_idx, 1)

        return res

    def is_poly_exact(self):
        # may be should be False
        return True


class MaxPool3d(Layer):
    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def apply(self, x):
        k_d, k_h, k_w = self.kernel

        p = self.padding
        x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p), (p,p)), mode='constant')
        x_n, x_c, x_d, x_h, x_w = x_pad.shape

        res_d = int((x_d - k_d) / self.stride) + 1
        res_h = int((x_h - k_h) / self.stride) + 1
        res_w = int((x_w - k_w) / self.stride) + 1

        c_idx, d_idx, h_idx, w_idx = index3d(x_c, self.stride, self.kernel, (x_d, x_h, x_w))

        res = x_pad[:, c_idx, d_idx, h_idx, w_idx]
        res = res.reshape(x_c, k_d * k_h * k_w, -1)

        res = np.max(res, axis=1)
        res = res.reshape(1, x_c, res_d, res_h, res_w)

        return res


class ResNet2l(Layer):
    def __init__(self, filters1, bias1, stride1, padding1,
        filters2, bias2, stride2, padding2,
        filtersX=None, biasX=None, strideX=None, paddingX=None):

        self.filters1 = filters1
        self.bias1 = bias1
        self.stride1 = stride1
        self.padding1 = padding1

        self.filters2 = filters2
        self.bias2 = bias2
        self.stride2 = stride2
        self.padding2 = padding2

        self.filtersX = filtersX
        self.biasX = biasX
        self.strideX = strideX
        self.paddingX = paddingX

    def apply(self, x):
        if len(self.filters1.shape) == 3:
            conv1 = Conv1d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv1d(self.filter2, self.bias2, self.stride2, self.padding2)
        elif len(self.filters1.shape) == 4:
            conv1 = Conv2d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv2d(self.filter2, self.bias2, self.stride2, self.padding2)
        elif len(self.filters1.shape) == 5:
            conv1 = Conv3d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv3d(self.filter2, self.bias2, self.stride2, self.padding2)

        res = conv1.apply(x)
        res = relu(res)
        res = conv2.apply(res)

        if self.filterX:
            if len(self.filtersX.shape) == 3:
                convX = Conv1d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 4:
                convX = Conv2d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 5:
                convX = Conv3d(self.filterX, self.biasX, self.strideX, self.paddingX)

            x = convX.apply(x)

        res = res + x

        return res


class ResNet3l(Layer):
    def __init__(self, filters1, bias1, stride1, padding1,
        filters2, bias2, stride2, padding2,
        filters3, bias3, stride3, paddind3,
        filtersX=None, biasX=None, strideX=None, paddingX=None):

        self.filters1 = filters1
        self.bias1 = bias1
        self.stride1 = stride1
        self.padding1 = padding1

        self.filters2 = filters2
        self.bias2 = bias2
        self.stride2 = stride2
        self.padding2 = padding2

        self.filters3 = filters3
        self.bias3 = bias3
        self.stride3 = stride3
        self.padding3 = padding3

        self.filtersX = filtersX
        self.biasX = biasX
        self.strideX = strideX
        self.paddingX = paddingX

    def apply(self, x):
        if len(self.filters1.shape) == 3:
            conv1 = Conv1d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv1d(self.filter2, self.bias2, self.stride2, self.padding2)
            conv3 = Conv1d(self.filter3, self.bias3, self.stride3, self.padding3)
        elif len(self.filters1.shape) == 4:
            conv1 = Conv2d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv2d(self.filter2, self.bias2, self.stride2, self.padding2)
            conv3 = Conv2d(self.filter3, self.bias3, self.stride3, self.padding3)
        elif len(self.filters1.shape) == 5:
            conv1 = Conv3d(self.filter1, self.bias1, self.stride1, self.padding1)
            conv2 = Conv3d(self.filter2, self.bias2, self.stride2, self.padding2)
            conv3 = Conv3d(self.filter3, self.bias3, self.stride3, self.padding3)

        res = conv1.apply(x)
        res = relu(res)
        res = conv2.apply(res)
        res = relu(res)
        res = conv3.apply(res)

        if self.filterX:
            if len(self.filtersX.shape) == 3:
                convX = Conv1d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 4:
                convX = Conv2d(self.filterX, self.biasX, self.strideX, self.paddingX)
            elif len(self.filters1.shape) == 5:
                convX = Conv3d(self.filterX, self.biasX, self.strideX, self.paddingX)

            x = convX.apply(x)

        res = res + x

        return res

# class Flatten(Layer):
#     """ Turns a multidimensional matrix into two-dimensional """
#     def __init__(self, input_shape=None):
#         self.prev_shape = None
#         self.trainable = True
#         self.input_shape = input_shape
#
#     def apply(self, x):
#         self.prev_shape = x.shape
#         return X.reshape((x.shape[0], -1))
#
#     def forward_pass(self, X, training=True):
#         self.prev_shape = X.shape
#         return X.reshape((X.shape[0], -1))
#
#     def backward_pass(self, accum_grad):
#         return accum_grad.reshape(self.prev_shape)
#
#     def output_shape(self):
#         return (np.prod(self.input_shape),)
#
# class Dropout(Layer):
#     """
#     A layer that randomly sets a fraction p of the output units of the previous layer
#     to zero.
#     Parameters:
#     -----------
#     p: float
#         The probability that unit x is set to zero.
#     """
#     def __init__(self, p=0.2):
#         self.p = p
#         self._mask = None
#         self.input_shape = None
#         self.n_units = None
#         self.pass_through = True
#         self.trainable = True
#
#     def apply(self, x):
#         self._mask = np.random.uniform(size=x.shape) > self.p
#         c = self._mask
#         return X * c
#
#     def forward_pass(self, X, training=True):
#         c = (1 - self.p)
#         if training:
#             self._mask = np.random.uniform(size=X.shape) > self.p
#             c = self._mask
#         return X * c
#
#     def backward_pass(self, accum_grad):
#         return accum_grad * self._mask
#
#     def output_shape(self):
#         return self.input_shape

class Dropout(Layer):
    """
    Wrapper of dropout operation
    """
    def __init__(self, keep_prob):
        self.__dict__.update(locals())
        self.keep_prob = keep_prob
        del self.self

    def apply(self, x):
        x = np.array(tf.nn.dropout(x, self.keep_prob))
        return x

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.dropout(x, self.keep_prob)

class Flatten(Layer):
    """
    Wrapper of reshape operation
    """
    def __init__(self):
        pass

    def apply(self, x):
        # x = tf.keras.layers.Flatten(x)
        # x = np.array(x)
        x = np.array([x.flatten()])
        return x

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [shape[0], output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    output = activation(dot(input, kernel) + bias)
    """
    def __init__(self, n_units, weights, bias, input_shape=None):
        self.n_units = n_units
        self.weights = np.array(weights)
        self.bias = np.array(bias)
        self.layer_input = None
        self.input_shape = input_shape
        self.trainable = True
        self.W = None
        self.w0 = None

    def apply(self, x):
        '''
        output = activation（dot（input，kernel）+ bias
        '''
        # print("-->x", x)
        # print(x.shape)
        # weights = self.weights.transpose()
        # print("-->weights", weights)
        # print(weights.shape)
        x = tf.add(tf.matmul(x, self.weights), self.bias)
        # x = np.dot(x, self.weights) + self.bias
        return x

    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units, )

    # def get_weight(self):
    #     return self.weights


class Dense_1(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)
    def forward(self,input):
        return np.dot(input,self.weights)+self.biases
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T,grad_output)/input.shape[0]
        grad_biases = grad_output.mean(axis=0)
        self.weights = self.weights - self.learning_rate*grad_weights
        self.biases = self.biases - self.learning_rate*grad_biases
        return grad_input

