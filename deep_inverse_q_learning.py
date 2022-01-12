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
    def __init__(self, state_feature_length, num_actions):
        """
        :param state_feature_length: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(Q, self).__init__()
        self.state_feature_length = state_feature_length
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.state_feature_length, 20)
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
    def __init__(self, state_feature_length, num_actions):
        """
        :param state_feature_length: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(ShiftQ, self).__init__()
        self.state_feature_length = state_feature_length
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.state_feature_length, 20)
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

    def __init__(self, state_feature_length, num_actions):
        """
        :param state_feature_length: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(Rho, self).__init__()
        self.state_feature_length = state_feature_length
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.state_feature_length, 80)
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

    def __init__(self, state_feature_length, num_actions):
        """
        :param state_feature_length: number of states. int.
        :param num_actions: number of actions. int.
        """

        super(R, self).__init__()
        self.state_feature_length = state_feature_length
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.state_feature_length, 20)
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
                 state_feature_length,
                 num_actions,
                 discount,
                 learning_rate,
                 tau,
                 device=torch.device('cuda')
                 ):

        self.state_feature_length = state_feature_length
        self.num_actions = num_actions
        self._discount = discount
        self._tau = tau

        self._device = device

        self._rho = Rho(self.state_feature_length, self.num_actions).to(self._device)
        self._rho_optim = optim.Adam(self._rho.parameters(), lr=learning_rate)

        self._q_sh = ShiftQ(self.state_feature_length, self.num_actions).to(self._device)
        self._q_sh_target = ShiftQ(self.state_feature_length, self.num_actions).to(self._device)
        self._q_sh_optim = optim.Adam(self._q_sh.parameters(), lr=learning_rate)

        self._r = R(self.state_feature_length, self.num_actions).to(self._device)
        self._r_target = R(self.state_feature_length, self.num_actions).to(self._device)
        self._r_optim = optim.Adam(self._r.parameters(), lr=learning_rate)

        self._q = Q(self.state_feature_length, self.num_actions).to(self._device)
        self._q_target = Q(self.state_feature_length, self.num_actions).to(self._device)
        self._q_optim = optim.Adam(self._q.parameters(), lr=learning_rate)

        self._cs_loss = nn.CrossEntropyLoss()
        self._mse_loss = nn.MSELoss()

    def _tt(self, nparray):
        return torch.tensor(nparray, dtype=torch.float, device=self._device)

    def _update_rho(self, states, actions):
        """
        train the function approximation model for policy.

        :param states: random sampled states. nparray. (Batch_Size, state_feature_length).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        """

        self._rho_optim.zero_grad()

        output = self._rho(self._tt(states))
        loss = self._cs_loss(output, torch.tensor(actions, dtype=torch.int64).to(self._device))
        loss.backward()
        self._rho_optim.step()

    def _update_q_sh(self, states, actions, next_states):
        """
        train the function approximation model for Shifted Q function Q_sh.

        :param states: random sampled states. nparray. (Batch_Size, state_feature_length).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        :param next_states: next states of random sampled states. nparray. (Batch_Size, state_feature_length).
        """

        self._q_sh_optim.zero_grad()

        gather_index = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self._device)
        pred_q_sh = self._q_sh(self._tt(states)).gather(1, gather_index).squeeze(1)

        target = self._discount * np.max(self._q_target(self._tt(next_states)).cpu().detach().numpy(), axis=1)

        loss = self._mse_loss(pred_q_sh, self._tt(target))
        loss.backward()
        self._q_sh_optim.step()

    def _update_r(self, states, actions):
        """
        train the function approximation model for reward function r.

        :param states: random sampled states. nparray. (Batch_Size, state_feature_length).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        """

        self._r_optim.zero_grad()

        gather_a_index = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self._device)
        gather_b_index = [[i for i in range(self.num_actions)] for j in range(len(actions))]
        for i, a in enumerate(actions):
            gather_b_index[i].remove(a)
        gather_b_index = torch.tensor(gather_b_index, dtype=torch.int64).to(self._device)

        pred_r = self._r(self._tt(states)).gather(1, gather_a_index).squeeze(1)

        eta = torch.log(self._rho(self._tt(states))) - self._q_sh_target(self._tt(states))
        eta_a = eta.gather(1, gather_a_index).squeeze(1)
        eta_b = eta.gather(1, gather_b_index)
        r_b = self._r_target(self._tt(states)).gather(1, gather_b_index)
        target = eta_a + 1 / (len(actions) - 1) * torch.sum((r_b - eta_b), dim=1)

        loss = self._mse_loss(pred_r, target.detach())
        loss.backward()
        self._r_optim.step()


    def _update_q(self, states, actions, next_states):
        """
        train the function approximation model for state value function Q.

        :param states: random sampled states. nparray. (Batch_Size, state_feature_length).
        :param actions: actions for random sampled states. nparray. (Batch_Size, 1).
        :param next_states: next states of random sampled states. nparray. (Batch_Size, state_feature_length).
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
        for key in target_function.state_dict().keys():
            target_w = target_function.state_dict()[key]
            source_w = source_function.state_dict()[key]
            target_function.state_dict()[key] = (1 - self._tau) * target_w + self._tau * source_w

    def train(self, states, actions, next_states):
        self._update_q_sh(states, actions, next_states)
        self._update_rho(states, actions)
        self._update_r(states, actions)
        self._update_q(states, actions, next_states)

        self._soft_update_target_function(self._q_sh_target, self._q_sh)
        self._soft_update_target_function(self._r_target, self._r)
        self._soft_update_target_function(self._q_target, self._q)

