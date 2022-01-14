import numpy as np

class TabularInverseQLearning:
    """
    Tabular Inverse Q-learning algorithm
    """
    def __init__(self, num_states, num_actions, discount, learning_rate):
        """
        :param num_states: number of states in the enviroment. int.
        :param num_actions: number of actions available. int.
        :param discount: discount rate gamma. float.
        :param learning_rate: learning rate. float.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self._discount = discount

        self._lr = learning_rate

        self._rho = np.ones((self.num_states, self.num_actions))
        self._q_sh = np.zeros((self.num_states, self.num_actions))
        self._r = np.zeros((self.num_states, self.num_actions))
        self._q = np.zeros((self.num_states, self.num_actions))

    def _update_rho(self, state, action):
        self._rho[state, action] += 1

    def _update_q_sh(self, state, action, next_state):
        self._q_sh[state, action] = (1 - self._lr) * self._q[state, action] + \
                                    self._lr * self._discount * np.max(self._q[next_state, :])

    def _update_r(self, state, action):
        pi = np.asarray([self._rho[state, i] / np.sum(self._rho[state, :]) for i in range(self.num_actions)])
        eta = np.log(pi) - self._q_sh[state, :]
        target = eta[action] + 1 / (self.num_actions - 1) * (np.sum(self._r[state, :] - eta) - (self._r[state, action] - eta[action]))
        self._r[state, action] = (1 - self._lr) * self._r[state, action] + self._lr * target

    def _update_q(self, state, action, next_state):
        self._q[state, action] = (1 - self._lr) * self._q[state, action] + self._lr * (self._r[state, action] + self._discount * np.max(self._q[next_state, :]))

    def train(self, state, action, next_state):
        """
        :param state: current state. int.
        :param action: action at the current state. int.
        :param next_state: next state after action being processed. int.
        """
        self._update_rho(state, action)
        self._update_q_sh(state, action, next_state)
        self._update_r(state, action)
        self._update_q(state, action, next_state)

    def get_reward(self):
        return self._r

    def get_q(self):
        return self._q
