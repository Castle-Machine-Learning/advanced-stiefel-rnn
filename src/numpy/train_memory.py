# Created by moritz (wolter@cs.uni-bonn.de)
# This script trains rnn cells on the memory problem.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_memory
from numpy_cells import BasicCell, CrossEntropyCost, MSELoss, Sigmoid
from numpy_cells import StiefelCell

if __name__ == '__main__':
    n_train = int(10e5)
    n_test = int(1e4)
    time_steps = 1
    n_sequence = 10
    train_data = generate_data_memory(time_steps, n_train, n_sequence)
    test_data = generate_data_memory(time_steps, n_test, n_sequence)
    # --- baseline ----------------------
    baseline = np.log(8) * 10/(time_steps + 20)
    print("Baseline is " + str(baseline))
    batch_size = 50
    lr = 0.1
    cell = BasicCell(hidden_size=128, input_size=10, output_size=10)
    sigmoid = Sigmoid()

    cost = CrossEntropyCost()

    train_x, train_y = generate_data_memory(time_steps, n_train, n_sequence)

    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=0)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    h = cell.zero_state(batch_size)
    loss_lst = []
    # train cell
    for i in range(iterations):
        xx = train_x_lst[i]
        yy = train_y_lst[i]

        x_one_hot = np.zeros([batch_size, 20+time_steps, n_sequence])
        y_one_hot = np.zeros([batch_size, 20+time_steps, n_sequence])
        # one hote encode the inputs.
        for b in range(batch_size):
            for t in range(20+time_steps):
                x_one_hot[b, t, xx[b, t]] = 1
                y_one_hot[b, t, yy[b, t]] = 1
                

        x = np.expand_dims(x_one_hot, -1)
        y = np.expand_dims(y_one_hot, -1)

        out_lst = []
        h_lst = []
        # forward
        for t in range(time_steps+20):
            out, h = cell.forward(x=x[:, t, :, :], h=h)
            out = sigmoid.forward(out)
            out_lst.append(out)
            h_lst.append(h)
        out_array = np.stack(out_lst, 1)
        # loss = cost.forward(label=y[:, -10:],
        #                     out=out_array[:, -10:])
        loss = cost.forward(label=y, out=out_array)


        # compute accuracy
        y_net = np.squeeze(np.argmax(out_array, axis=2))
        mem_net = y_net[:, -10:]
        mem_y = yy[:, -10:]
        acc = np.sum((mem_y == mem_net).astype(np.float32))
        acc = acc/(batch_size * 10.)

        deltay = np.zeros([batch_size, time_steps+20, n_sequence, 1])
        # deltay[:, -10:, :, :] = cost.backward(label=y[:, -10:, :, :],
        #                                       out=out_array[:, -10:, :, :])
        deltay = cost.backward(label=y, out=out_array)

        deltah = cell.zero_state(batch_size)
        grad_lst = []
        # backward
        for t in reversed(range(time_steps+20)):
            deltah, dWhh, dWxh, dWhy, dby = \
                cell.backward(deltay=deltay[:, t, :, :],
                              deltah=deltah,
                              x=x[:, t, :, :],
                              h=h_lst[t],
                              hm1=h_lst[t-1])
            grad_lst.append([dWhh, dWxh, dWhy, dby])


        # update
        cell.update(grad_lst=grad_lst, lr=lr)

        if i % 10 == 0:
            print(i, 'loss', "%.4f" % loss, 'baseline', baseline,
                  'acc', "%.4f" % acc, 'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 500 == 0 and i > 0:
            lr = lr * 0.95
            
            #res = np.squeeze(np.argmax(out_array, axis=2))
            #import pdb;pdb.set_trace()
            print('state transition norm:', cell.get_state_transition_norm())
            print('net', y_net[0, -10:])
            print('gt ', yy[0, -10:])

    print(y[:, 0, 0])
    print(out[:, 0, 0])
    plt.semilogy(loss_lst)
    plt.show()
