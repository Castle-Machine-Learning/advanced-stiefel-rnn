import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init


class CustomGRU(torch.nn.Module):
    def __init__(self, hidden_size=250,
                 input_size=1, output_size=1):
        """Create a Gated Recurrent unit.
        Args:
            hidden_size (int, optional): The cell size. Defaults to 250.
            input_size (int, optional): The number of input dimensions.
                                        Defaults to 1.
            output_size (int, optional): Output dimension number.
                                         Defaults to 1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        # create the weights
        self.Vr = Parameter(torch.Tensor(input_size, hidden_size))
        self.Vu = Parameter(torch.Tensor(input_size, hidden_size))
        self.V = Parameter(torch.Tensor(input_size, hidden_size))

        self.Wr = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Wu = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W = Parameter(torch.Tensor(hidden_size, hidden_size))

        self.br = torch.Tensor(hidden_size)
        self.bu = Parameter(torch.Tensor(hidden_size))
        self.b = Parameter(torch.Tensor(hidden_size))

        self.state_activation = torch.nn.Tanh()
        self.gate_r_act = torch.nn.Sigmoid()
        self.gate_u_act = torch.nn.Sigmoid()

    def reset_parameters(self) -> None:
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


    def forward(self, x, h):
        rbar = torch.matmul(x, self.Vr) \
            + torch.matmul(h, self.Wr) \
            + self.br
        r = self.gate_r_act.forward(rbar)
        # update gate
        ubar = torch.matmul(x, self.Vu) \
            + torch.matmul(h, self.Wu) \
            + self.bu
        u = self.gate_u_act(ubar)
        # block itorchut
        hbar = r*h
        zbar = torch.matmul(x, self.V) \
            + torch.matmul(hbar, self.W) \
            + self.b
        z = self.state_activation(zbar)
        # recurrent update
        h = u*z + (1 - u)*h
        return h

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

