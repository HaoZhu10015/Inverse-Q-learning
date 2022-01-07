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
    def __init(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


def tt(nparray, device=torch.device('cuda')):
    return torch.tensor(nparray, dtype=torch.float, device=device)