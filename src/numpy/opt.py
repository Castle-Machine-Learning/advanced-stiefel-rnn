import numpy as np
from src.numpy.numpy_cells import StiefelArray

class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, net, grads):
        param_keys = net.weights.keys()
        for weight_key in param_keys:
            cgrads = np.expand_dims(np.mean(grads[weight_key], 0), 0)
            net.weights[weight_key] += -self.lr*cgrads


class RMSprop(SGD):
    def __init__(self, lr=0.01, rho=0.99):
        super().__init__(lr)
        self.delta = 1e-6
        self.r = None
        self.rho = rho

    def step(self, net, grads):
        param_keys = net.weights.keys()

        if self.r is None:
            self.r = {}
            for weight_key in param_keys:
                grad = np.expand_dims(np.mean(grads[weight_key], 0), 0)
                self.r[weight_key] = np.zeros_like(grad)

        for weight_key in param_keys:
            cgrads = np.expand_dims(np.mean(grads[weight_key], 0), 0)
            self.r[weight_key] = self.rho*self.r[weight_key] \
                                + (1 - self.rho)*cgrads*cgrads
            step_size = -self.lr / (np.sqrt(self.delta + self.r[weight_key]))
            net.weights[weight_key] += step_size*cgrads


class StiefelRMSopt(RMSprop):

    def step(self, net, grads):
        param_keys = net.weights.keys()

        if self.r is None:
            self.r = {}
            for weight_key in param_keys:
                grad = np.expand_dims(np.mean(grads[weight_key], 0), 0)
                self.r[weight_key] = np.zeros_like(grad)

        for weight_key in param_keys:
            weight_type = type(net.weights[weight_key])
            if weight_type is StiefelArray:
                G = np.expand_dims(np.mean(grads[weight_key], 0), 0)
                W = net.weights[weight_key]
                eye = np.expand_dims(np.identity(W.shape[-1]), 0)
                A = np.matmul(G, np.transpose(W, [0, 2, 1])) \
                              - np.matmul(W, np.transpose(G, [0, 2, 1]))
                cayleyDenom = eye + (self.lr/2.0) * A
                cayleyNumer = eye - (self.lr/2.0) * A
                C = np.matmul(np.linalg.inv(cayleyDenom), cayleyNumer)
                W_new = np.matmul(C, W)
                net.weights[weight_key] = W_new
            else:
                cgrads = np.expand_dims(np.mean(grads[weight_key], 0), 0)
                self.r[weight_key] = self.rho*self.r[weight_key] \
                                    + (1 - self.rho)*cgrads*cgrads
                step_size = -self.lr / (np.sqrt(self.delta + self.r[weight_key]))
                net.weights[weight_key] += step_size*cgrads    