import numpy as np
import sys

class InverseActionValueIteration:
    def __init__(self, num_states,
                 num_actions,
                 transition_probability_matrix,
                 expert_policy,
                 discount=0.8,
                 threshold=1e-3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_probability_matrix = transition_probability_matrix
        self.expert_policy = expert_policy
        self.discount = discount
        self.threshold = threshold
        self.epsilon = 1e-6

        self._r = np.random.random((self.num_states, self.num_actions))
        self._q = np.random.random((self.num_states, self.num_actions))

    def train(self):
        X = np.ones((self.num_actions, self.num_actions))
        X *= -1 / (self.num_actions - 1)
        for i in range(self.num_actions):
            X[i, i] = 1

        e = 0
        while True:
            e += 1
            delta = 0
            for s in range(self.num_states):
                tp = self.transition_probability_matrix[s, :, :]
                eta = np.log(self.expert_policy[s, :] + self.epsilon) - self.discount * np.matmul(
                    tp.T, np.max(self._q, axis=1).reshape(-1, 1)).reshape(-1)

                Y = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    eta_a = eta[a]
                    action_b = [b for b in range(self.num_actions) if b != a]
                    eta_b = eta[action_b]
                    Y[a] = eta_a - 1 / (self.num_actions - 1) * np.sum(eta_b)

                r = np.linalg.lstsq(X, Y, rcond=None)[0]

                delta = max(delta, np.max(np.abs(self._r[s, :] - r)))

                self._r[s, :] = r
                self._q[s, :] = r + self.discount * np.matmul(tp.T, np.max(self._q, axis=1).reshape(-1, 1)).reshape(-1)

            print("\rDelta: {:6f} || Epoch: {}".format(delta, e), end="")
            sys.stdout.flush()

            if delta < self.threshold:
                break

    def get_reward(self):
        return self._r.copy()

    def get_q(self):
        return self._q.copy()