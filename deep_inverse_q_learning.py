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
    """
    Deep Inverse Q-learning algorithm.
    """
    def __init__(self,
                 q_function,
                 q_sh_function,
                 rho_function,
                 r_function,
                 num_states,
                 num_actions,
                 discount,
                 learning_rate,
                 device=torch.device('cuda')
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self._discount = discount

        self._device = device

        self._rho = rho_function(self.num_states, self.num_actions).to(self._device)
        self._rho_optim = optim.Adam(self._rho.parameters(), lr=learning_rate)

        self._q_sh = q_sh_function(self.num_states, self.num_actions).to(self._device)
        self._q_sh_target = q_sh_function(self.num_states, self.num_actions).to(self._device)
        self._q_sh_optim = optim.Adam(self._q_sh.parameters, lr=learning_rate)

        self._r = r_function(self.num_states, self.num_actions).to(self._device)
        self._r_target = r_function(self.num_states, self.num_actions).to(self._device)
        self._r_optim = optim.Adam(self._r.parameters(), lr=learning_rate)

        self._q = q_function(self.num_states, self.num_actions).to(self._device)
        self._q_target = q_function(self.num_states, self.num_actions).to(self._device)
        self._q_optim = optim.Adam(self._q.parameters(), lr=learning_rate)

        self._cs_loss = nn.CrossEntropyLoss()
        self._mse_loss = nn.MSELoss()

    def _tt(self, nparray):
        return torch.tensor(nparray, dtype=torch.float, device=self._device)

    def _update_rho(self, states, actions):
        """
        train the function approximation model for policy.

        :param states: random sampled states. nparray. (Batch_Size, num_states).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        """

        self._rho_optim.zero_grad()

        output = self._rho(self._tt(states))
        loss = self._cs_loss(output, self._tt(actions))
        loss.backward()
        self._rho_optim.step()

    def _update_q_sh(self, states, actions, next_states):
        """
        train the function approximation model for Shifted Q function Q_sh.

        :param states: random sampled states. nparray. (Batch_Size, num_states).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        :param next_states: next states of random sampled states. nparray. (Batch_Size, num_states).
        """

        self._q_sh_optim.zero_grad()

        gather_index = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self._device)
        pred_q_sh = self._q_sh(self._tt(states)).gather(1, gather_index).squeeze(1)

        target = self._discount * np.max(self._q_target(self._tt(next_states)).cpu().detach().numpy(), axis=1)

        loss = self._mse_loss(pred_q_sh, self._tt(target))
        loss.backward()
        self._q_sh_optim.step()

    def _update_r(self, states, actions, next_states):
        raise NotImplementedError

    def _update_q(self, states, actions, next_states):
        """
        train the function approximation model for state value function Q.

        :param states: random sampled states. nparray. (Batch_Size, num_states).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        :param next_states: next states of random sampled states. nparray. (Batch_Size, num_states).
        """

        self._q_optim.zero_grad()

        gather_index = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self._device)
        pred_q = self._q(self._tt(states)).gather(1, gather_index).squeeze(1)

        target_r = self._r_target(self._tt(states)).gather(1, gather_index).squeeze(1).cpu().detach().numpy()
        target = target_r + self._discount * np.max(self._q_target(self._tt(next_states)).cpu().detach().numpy(),
                                                    axis=1)

        loss = self._mse_loss(pred_q, self._tt(target))
        loss.backward()
        self._q_optim.step()

    def _soft_update_target_function(self, target_function, source_function):
        raise NotImplementedError

    def train(self, states, actions, next_states):
        self._update_q_sh(states, actions, next_states)
        self._update_rho(states, actions)
        self._update_r(states, actions, next_states)
        self._update_q(states, actions, next_states)

        self._soft_update_target_function(self._q_sh_target, self._q_sh)
        self._soft_update_target_function(self._r_target, self._r)
        self._soft_update_target_function(self._q_target, self._q)


