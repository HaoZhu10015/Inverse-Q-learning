from objectworld import ObjectWorld
from utils import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Q(nn.Module):
    """
    Neural Network for Q function approximation.
    """
    def __init__(self, num_states, num_actions):
        """
        :param num_states: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(Q, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 20)
        self.fc2 = nn.Linear(20, 80)
        self.fc3 = nn.Linear(80, self.num_actions)
        self.actvation = nn.ReLU()

    def forward(self, x):
        x = self.actvation(self.fc1(x))
        x = self.actvation(self.fc2(x))
        return self.fc3(x)


class ShiftQ(nn.Module):
    """
    Neural Network for Shift-Q function approximation.
    """
    def __init__(self, num_states, num_actions):
        """
        :param num_states: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(ShiftQ, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 20)
        self.fc2 = nn.Linear(20, 80)
        self.fc3 = nn.Linear(80, self.num_actions)
        self.actvation = nn.ReLU()

    def forward(self, x):
        x = self.actvation(self.fc1(x))
        x = self.actvation(self.fc2(x))
        return self.fc3(x)


class Rho(nn.Module):
    """
    Neural Network for policy function approximation.
    """

    def __init__(self, num_states, num_actions):
        """
        :param num_states: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(Rho, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 80)
        self.fc2 = nn.Linear(80, 20)
        self.fc3 = nn.Linear(20, self.num_actions)
        self.actvation = nn.ReLU()

    def forward(self, x):
        x = self.actvation(self.fc1(x))
        x = self.actvation(self.fc2(x))
        return self.fc3(x)


class R(nn.Module):
    """
    Neural Network for reward function approximation.
    """

    def __init__(self, num_states, num_actions):
        """
        :param num_states: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(R, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 20)
        self.fc2 = nn.Linear(20, 80)
        self.fc3 = nn.Linear(80, self.num_actions)
        self.actvation = nn.ReLU()

    def forward(self, x):
        x = self.actvation(self.fc1(x))
        x = self.actvation(self.fc2(x))
        return self.fc3(x)


class DeepInverseQLearning:
    def __init__(self,
                 q_function,
                 q_sh_function,
                 rho_function,
                 r_function,
                 num_states,
                 num_actions,
                 learning_rate,
                 ):
        self.num_states = num_states
        self.num_actions = num_actions

        self.rho = rho_function(self.num_states, self.num_actions)
        self.rho_optim = optim.Adam(self.rho.parameters(), lr=learning_rate)

        self.q_sh = q_sh_function(self.num_states, self.num_actions)
        self.q_sh_target = q_sh_function(self.num_states, self.num_actions)
        self.q_sh_optim = optim.Adam(self.q_sh.parameters, lr=learning_rate)

        self.r = r_function(self.num_states, self.num_actions)
        self.r_target = r_function(self.num_states, self.num_actions)
        self.r_optim = optim.Adam(self.r.parameters(), lr=learning_rate)

        self.q = q_function(self.num_states, self.num_actions)
        self.q_target = q_function(self.num_states, self.num_actions)
        self.q_optim = optim.Adam(self.q.parameters(), lr=learning_rate)

    def _update_rho(self, states, actions, next_states):
        raise NotImplementedError

    def _update_q_sh(self, states, actions, next_states):
        raise NotImplementedError

    def _update_r(self, states, actions, next_actions):
        raise NotImplementedError

    def _update_q(self, states, actions, next_actions):
        raise NotImplementedError

    def _soft_update_target_function(self, target_function, source_function):
        raise NotImplementedError

    def train(self, states, actions, next_states):
        self._update_q_sh(states, actions, next_states)
        self._update_rho(states, actions, next_states)
        self._update_r(states, actions, next_states)
        self._update_q(states, actions, next_states)

        self._soft_update_target_function(self.q_sh_target, self.q_sh)
        self._soft_update_target_function(self.r_target, self.r)
        self._soft_update_target_function(self.q_target, self.q)


def tt(nparray, device=torch.device('cuda')):
    return torch.tensor(nparray, dtype=torch.float, device=device)