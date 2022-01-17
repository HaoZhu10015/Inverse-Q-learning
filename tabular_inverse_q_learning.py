import numpy as np


class TabularInverseQLearning:
    """
    Tabular Inverse Q-learning algorithm
    """

    def __init__(self, num_states, num_actions, discount, alpha_r, alpha_q_sh, alpha_q):
        """
        :param num_states: number of states in the enviroment. int.
        :param num_actions: number of actions available. int.
        :param discount: discount rate gamma. float.
        :param learning_rate: learning rate. float.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self._discount = discount

        self._alpha_r = alpha_r
        self._alpha_q_sh = alpha_q_sh
        self._alpha_q = alpha_q

        self._rho = np.zeros((self.num_states, self.num_actions))
        self._q_sh = np.zeros((self.num_states, self.num_actions))
        self._r = np.zeros((self.num_states, self.num_actions))
        self._q = np.zeros((self.num_states, self.num_actions))

        self._epsilon = 1e-6

    def _update_rho(self, state, action):
        self._rho[state, action] += 1

    def _update_q_sh(self, state, action, next_state):
        self._q_sh[state, action] = (1 - self._alpha_q_sh) * self._q_sh[state, action] + \
                                    self._alpha_q_sh * self._discount * np.max(self._q[next_state, :])

    def _update_r(self, state, action):
        pi = np.asarray([self._rho[state, i] / np.sum(self._rho[state, :]) for i in range(self.num_actions)])
        action_b = [b for b in range(self.num_actions) if b != action]
        eta_a = np.log(pi[action] + self._epsilon) - self._q_sh[state, action]
        eta_b = np.log(pi[action_b] + self._epsilon) - self._q_sh[state, action_b]
        target = eta_a + 1 / (self.num_actions - 1) * (np.sum(self._r[state, action_b] - eta_b))
        self._r[state, action] = (1 - self._alpha_r) * self._r[state, action] + self._alpha_r * target

    def _update_q(self, state, action, next_state):
        self._q[state, action] = (1 - self._alpha_q) * self._q[state, action] + self._alpha_q * (
                    self._r[state, action] + self._discount * np.max(self._q[next_state, :]))

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
        return self._r.copy()

    def get_q(self):
        return self._q.copy()

