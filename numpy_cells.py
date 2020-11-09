# Written by moritz (wolter@cs.uni-bonn.de)

# based on https://gist.github.com/karpathy/d4dee566867f8291f086,
# see also https://arxiv.org/pdf/1503.04069.pdf
# and https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py

import numpy as np
import sys
import pdb


class CrossEntropyCost(object):

    def forward(self, label, out):
        # np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        return np.mean(np.nan_to_num(
            -label*np.log(out) - (1-label)*np.log(1-out)))

    def backward(self, label, out):
        """ Assuming a sigmoidal netwok output."""
        return (out-label)


class MSELoss(object):
    ''' Mean squared error loss function. '''
    def forward(self, label, out):
        diff = out - label
        return np.mean(diff*diff)

    def backward(self, label, out):
        return out - label


class DenseLayer(object):
    def __init__(self, in_shape, out_shape):
        self.weights = {}
        W = np.zeros([1, out_shape, in_shape])
        W = W + np.random.randn(1, out_shape, in_shape)
        W = W / np.sqrt(in_shape)
        self.weights['W'] = W
        b = np.random.randn(1, out_shape, 1)
        self.weights['b'] = b


    def forward(self, inputs):
        return np.matmul(self.weights['W'], inputs) + self.weights['b']

    def backward(self, inputs, delta) -> {}:
        """Backward pass through a dense layer.
        Args:
            inputs: [batch_size, input_dim, 1]
            delta: [batch_size, out_dim, 1]
        Returns:
            A dictionary containing:
            'W' - dW (np.array): Weight gradients
            'b' - db (np.array): bias gradients
            'x' - dx (np.array): input gradients for lower layers.
        """
        dx = np.matmul(np.transpose(self.weights['W'], [0, 2, 1]), delta)
        dw = np.matmul(delta, np.transpose(inputs, [0, 2, 1]))
        db = 1*delta
        return {'W': dw, 'b': db, 'x': dx}


class ReLu(object):

    def forward(self, inputs):
        inputs[inputs <= 0] = 0
        return inputs

    def backward(self, inputs, delta):
        delta[delta <= 0] = 0
        return delta

class Sigmoid(object):
    """ Sigmoid activation function. """
    def sigmoid(self, inputs):
        # sig = np.exp(inputs)/(1 + np.exp(inputs))
        # return np.nan_to_num(sig)
        return np.where(inputs >= 0, 
                        1 / (1 + np.exp(-inputs)), 
                        np.exp(inputs) / (1 + np.exp(inputs)))

    def forward(self, inputs):
        return self.sigmoid(inputs)

    def backward(self, inputs, delta):
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))*delta

    def prime(self, inputs):
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))


class Tanh(object):
    """ Hyperbolic tangent activation function. """
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, inputs, delta):
        return (1. - np.tanh(inputs)*np.tanh(inputs))*delta

    def prime(self, inputs):
        return (1. - np.tanh(inputs)*np.tanh(inputs))


class LSTMcell(object):
    def __init__(self, hidden_size=64,
                 input_size=1, output_size=1):
        """ Instantiate a Long Short Term Memory Cell.

        Args:
            hidden_size (int, optional): The cell size. Defaults to 64.
            input_size (int, optional): Input size. Defaults to 1.
            output_size (int, optional): The size of the output. Defaults to 1.
        """
        self.hidden_size = hidden_size
        # create the weights
        s = 1./np.sqrt(hidden_size)
        self.weights = {}
        self.weights['Wz'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Wi'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Wf'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Wo'] = np.random.randn(1, hidden_size, input_size)*s

        self.weights['Rz'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Ri'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Rf'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Ro'] = np.random.randn(1, hidden_size, hidden_size)*s

        self.weights['bz'] = np.zeros((1, hidden_size, 1))*s
        self.weights['bi'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['bf'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['bo'] = np.random.randn(1, hidden_size, 1)*s

        self.weights['pi'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['pf'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['po'] = np.random.randn(1, hidden_size, 1)*s

        self.block_act = Tanh()
        self.out_activation = Tanh()
        self.gate_i_act = Sigmoid()
        self.gate_f_act = Sigmoid()
        self.gate_o_act = Sigmoid()

        self.weights['Wout'] = np.random.randn(1, output_size, hidden_size)*s
        self.weights['bout'] = np.random.randn(1, output_size, 1)

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size, 1))

    def forward(self, x, h, c) -> {}:
        """ Implementation of the LSTM forward pass.

        Args:
            x (np.array): Array containing the current inputs..
            h (np.array): Pre-projection cell output vector.
            c (np.array): Cell state.

        Returns:
            A dictionary containing:
                y (np.array): Projected cell output.
                c (np.array): Cell memory state
                h (np.array): Gated output vector.
                zbar (np.array): Pre-activation block input.
                z    (np.array): Block input vector.
                ibar (np.array): Pre-activation input gate vector.
                i    (np.array): Input gate vector.
                fbar (np.array): Pre-activation forget gate vector.
                f    (np.array): Forget gate vector.
                obar (np.array): Pre-activation output gate vector.
                o    (np.array): Output gate vector.
                x    (np.array): Input used to evaluate the cell.
        """
        # block input
        zbar = np.matmul(self.weights['Wz'], x) \
            + np.matmul(self.weights['Rz'], h) \
            + self.weights['bz']
        z = self.block_act.forward(zbar)
        # input gate
        ibar = np.matmul(self.weights['Wi'], x) \
            + np.matmul(self.weights['Ri'], h) \
            + self.weights['pi']*c \
            + self.weights['bi']
        i = self.gate_i_act.forward(ibar)
        # forget gate
        fbar = np.matmul(self.weights['Wf'], x) \
            + np.matmul(self.weights['Rf'], h) \
            + self.weights['pf']*c \
            + self.weights['bf']
        f = self.gate_f_act.forward(fbar)
        # cell
        c = z * i + c * f
        # output gate
        obar = np.matmul(self.weights['Wo'], x) \
            + np.matmul(self.weights['Ro'], h) \
            + self.weights['po']*c \
            + self.weights['bo']
        o = self.gate_o_act.forward(obar)
        # block output
        h = self.out_activation.forward(c)*o
        # linear projection
        y = np.matmul(self.weights['Wout'], h) + self.weights['bout']
        return {'y': y, 'c': c, 'h': h, 'zbar': zbar, 'z': z,
                'ibar': ibar, 'i': i,  'fbar': fbar, 'f': f,
                'obar': obar, 'o': o, 'x': x}

    def backward(self, deltay, fd, prev_fd, next_fd, next_gd) -> {}:
        """ The LSTM backward pass, as described in
            https://arxiv.org/pdf/1503.04069.pdf section B.

        Args:
            deltay (np.array): Gradients from the layer above.
            fd (dict): Forward dictionary recording the forward pass values.
            prev_fd (dict): Dictionary at time t-1.
            next_ft (dict): Dictionary at time t+1.
            next_gd (dict): Gradients at time t+1.

        Returns:
            A dictionary with the gradients at time t.
        """
        # projection backward
        dWout = np.matmul(deltay, np.transpose(fd['h'], [0, 2, 1]))
        dbout = 1*deltay
        deltay = np.matmul(np.transpose(self.weights['Wout'], [0, 2, 1]),
                           deltay)

        # block backward
        deltah = deltay \
            + np.matmul(np.transpose(self.weights['Rz'], [0, 2, 1]),
                        next_gd['deltaz']) \
            + np.matmul(np.transpose(self.weights['Ri'], [0, 2, 1]),
                        next_gd['deltai']) \
            + np.matmul(np.transpose(self.weights['Rf'], [0, 2, 1]),
                        next_gd['deltaf']) \
            + np.matmul(np.transpose(self.weights['Ro'], [0, 2, 1]),
                        next_gd['deltao'])

        deltao = deltah * self.out_activation.forward(fd['c']) \
            * self.gate_o_act.prime(fd['obar'])
        deltac = deltah * fd['o'] * self.block_act.prime(fd['c'])\
            + self.weights['po']*deltao \
            + self.weights['pi']*next_gd['deltai'] \
            + self.weights['pf']*next_gd['deltaf'] \
            + next_gd['deltac']*next_fd['f']
        deltaf = deltac * prev_fd['c'] * self.gate_f_act.prime(fd['fbar'])
        deltai = deltac * fd['z'] * self.gate_i_act.prime(fd['ibar'])
        deltaz = deltac * fd['i'] * self.block_act.prime(fd['zbar'])

        # input weight backward
        dWz = np.matmul(deltaz, np.transpose(fd['x'], [0, 2, 1]))
        dWi = np.matmul(deltai, np.transpose(fd['x'], [0, 2, 1]))
        dWf = np.matmul(deltaf, np.transpose(fd['x'], [0, 2, 1]))
        dWo = np.matmul(deltao, np.transpose(fd['x'], [0, 2, 1]))

        # Compute recurrent weight gradients.
        dRz = np.matmul(next_gd['deltaz'], np.transpose(fd['h'], [0, 2, 1]))
        dRi = np.matmul(next_gd['deltai'], np.transpose(fd['h'], [0, 2, 1]))
        dRf = np.matmul(next_gd['deltaf'], np.transpose(fd['h'], [0, 2, 1]))
        dRo = np.matmul(next_gd['deltao'], np.transpose(fd['h'], [0, 2, 1]))

        # bias weights
        dbz = deltaz
        dbi = deltai
        dbf = deltaf
        dbo = deltao

        # peephole connection gradient.
        dpi = fd['c']*next_gd['deltai']
        dpf = fd['c']*next_gd['deltaf']
        dpo = fd['c']*deltao

        return {'deltac': deltac, 'deltaz': deltaz, 'deltao': deltao,
                'deltai': deltai, 'deltaf': deltaf,
                'dWout': dWout, 'dbout': dbout,
                'dWz': dWz, 'dWi': dWi, 'dWf': dWf, 'dWo': dWo,
                'dRz': dRz, 'dRi': dRi, 'dRf': dRf, 'dRo': dRo,
                'dbz': dbz, 'dbi': dbi, 'dbf': dbf, 'dbo': dbo,
                'dpi': dpi, 'dpf': dpf, 'dpo': dpo}

    def update(self):
        """ Compute a SGD update step. """
        # TODO: Write me!
        pass



class BasicCell(object):
    """Basic (Elman) rnn cell."""

    def __init__(self, hidden_size=250, input_size=1, output_size=1,
                 activation=Tanh()):
        self.hidden_size = hidden_size
        # input to hidden
        self.Wxh = np.random.randn(1, hidden_size, input_size)
        self.Wxh = self.Wxh / np.sqrt(hidden_size)
        # hidden to hidden
        self.Whh = np.random.randn(1, hidden_size, hidden_size)
        self.Whh = self.Whh / np.sqrt(hidden_size)
        # hidden to output
        self.Why = np.random.randn(1, output_size, hidden_size)
        self.Why = self.Why / np.sqrt(hidden_size)
        # hidden bias
        self.bh = np.zeros((1, hidden_size, 1))
        # output bias
        self.by = np.random.randn(1, output_size, 1)
        self.activation = activation

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size, 1))

    def get_state_transition_norm(self):
        return np.linalg.norm(np.squeeze(self.Whh), ord=2)

    def forward(self, x, h):
        """Basic Cell forward pass.

        Args:
            x (np.array): The input at the current time step.
            h (np.array): The cell-state at the current time step.

        Returns:
            y (np.array): Cell output.
            h (np.array): Updated cell state.
        """
        h = np.matmul(self.Whh, h) + np.matmul(self.Wxh, x) + self.bh
        h = self.activation.forward(h)
        y = np.matmul(self.Why, h) + self.by
        return y, h

    def backward(self, deltay, deltah, x, h, hm1):
        """The backward pass of the Basic-RNN cell.

        Args:
            deltay (np.array): Deltas from the layer above.
                               [batch_size, output_size, 1].
            deltah (np.array): Cell state deltas.
            x (np.array): Input at current time step.
            h (np.array): State at current time step.
            hm1 (np.array): State at previous time step.

        Returns:
            deltah (np.array): Updated block deltas.
            dWhh (np.array): Recurrent weight matrix gradients.
            dWxh (np.array): Input weight matrix gradients
            dbh (np.array): Bias gradients.
            dWhy (np.array):  Output projection matrix gradients.
            dby (np.array): Ouput bias gradients.
        """
        # output backprop
        dydh = np.matmul(np.transpose(self.Why, [0, 2, 1]), deltay)
        dWhy = np.matmul(deltay, np.transpose(h, [0, 2, 1]))
        dby = 1*deltay

        delta = self.activation.backward(inputs=h, delta=dydh) + deltah
        # recurrent backprop
        dWxh = np.matmul(delta, np.transpose(x, [0, 2, 1]))
        dWhh = np.matmul(delta, np.transpose(hm1, [0, 2, 1]))
        dbh = 1*delta
        deltah = np.matmul(np.transpose(self.Whh, [0, 2, 1]), delta)
        # deltah, dWhh, dWxh, dbh, dWhy, dby
        return deltah, dWhh, dWxh, dbh, dWhy, dby

    def update(self, grad_lst, lr, clip=1.):
        """ Do a basic SGD update step.

        Args:
            grad_lst: A list with the numerical gradients over 
                      time.
        """
        ldWhh, ldWxh, ldbh, ldWhy, ldby = zip(*grad_lst)
        dWhh = np.stack(ldWhh, axis=0)
        dWxh = np.stack(ldWxh, axis=0)
        dbh = np.stack(ldbh, axis=0)
        dWhy = np.stack(ldWhy, axis=0)
        dby = np.stack(ldby, axis=0)
        # backprop in time requires us to sum the gradients at each
        # point in time.

        # clipping.
        dWhh = np.clip(np.sum(dWhh, axis=0), -clip, clip)
        dWxh = np.clip(np.sum(dWxh, axis=0), -clip, clip)
        dbh = np.clip(np.sum(dbh, axis=0), -clip, clip)
        dWhy = np.clip(np.sum(dWhy, axis=0), -clip, clip)
        dby = np.clip(np.sum(dby, axis=0), -clip, clip)

        # update
        self.Whh += -lr*np.expand_dims(np.mean(dWhh, 0), 0)
        self.Wxh += -lr*np.expand_dims(np.mean(dWxh, 0), 0)
        self.bh += -lr*np.expand_dims(np.mean(dbh, 0), 0)
        self.Why += -lr*np.expand_dims(np.mean(dWhy, 0), 0)
        self.by += -lr*np.expand_dims(np.mean(dby, 0), 0)


class StiefelCell(BasicCell):
    """ Definition of a stiefel cell. """
    def __init__(self, hidden_size=250, input_size=1, output_size=1,
                 activation=ReLu()):
        super().__init__(hidden_size=hidden_size,
                       input_size=input_size,
                       output_size=output_size,
                       activation=activation)
        u, s, vh = np.linalg.svd(np.squeeze(self.Whh, axis=0))        
        Whh = np.matmul(u, vh.T)
        self.Whh = np.expand_dims(Whh, axis=0)
        print('Whh norm', self.get_state_transition_norm())


    def update(self, grad_lst, lr, clip=1.):

        ldWhh, ldWxh, ldbh, ldWhy, ldby = zip(*grad_lst)
        dWhh = np.stack(ldWhh, axis=0)
        dWxh = np.stack(ldWxh, axis=0)
        dbh = np.stack(ldbh, axis=0)
        dWhy = np.stack(ldWhy, axis=0)
        dby = np.stack(ldby, axis=0)
        # backprop in time requires us to sum the gradients at each
        # point in time.

        # clipping.
        dWhh = np.clip(np.sum(dWhh, axis=0), -clip, clip)
        dWxh = np.clip(np.sum(dWxh, axis=0), -clip, clip)
        dbh = np.clip(np.sum(dbh, axis=0), -clip, clip)
        dWhy = np.clip(np.sum(dWhy, axis=0), -clip, clip)
        dby = np.clip(np.sum(dby, axis=0), -clip, clip)

        
        # stiefel-update
        G = np.mean(dWhh, 0)
        W = np.squeeze(self.Whh, axis=0)

        # pdb.set_trace()
        eye = np.eye(self.hidden_size)
        A = np.matmul(G, np.transpose(W)) - np.matmul(W, np.transpose(G))
        cayleyDenom = eye + (lr/2.0) * A
        cayleyNumer = eye - (lr/2.0) * A
        C = np.matmul(np.linalg.inv(cayleyDenom), cayleyNumer)
        # import pdb;pdb.set_trace()
        self.Whh = np.expand_dims(C, axis=0)

        # SGD updates.
        self.Wxh += -lr*np.expand_dims(np.mean(dWxh, 0), 0)
        self.bh += -lr*np.expand_dims(np.mean(dbh, 0), 0)
        self.Why += -lr*np.expand_dims(np.mean(dWhy, 0), 0)
        self.by += -lr*np.expand_dims(np.mean(dby, 0), 0)
