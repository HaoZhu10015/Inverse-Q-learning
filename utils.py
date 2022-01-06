import numpy as np


class ReplayBuffer:
    """
    Implement replay buffer.
    """
    def __init__(self, max_size):
        """
        :param max_size: the maximum number of experiences stored in the replay buffer. int.
        """

        self._max_size = max_size
        self._data = {
            "states": [],
            "actions": [],
            "next_states": [],
        }
        self._size = 0

    def get_num_experiences(self):
        """
        :return: current number of experiences stored in the replay buffer. int.
        """
        return self._size

    def add_experience(self, experience):
        """
        add generated experience to the replay buffer.

        :param experience: experience to be added to the replay buffer. (state, action, next_state).
        """
        if self._size == self._max_size:
            for i, key in enumerate(self._data.keys()):
                self._data[key].pop(0)
                self._data[key].append(experience[i])
        else:
            for i, key in enumerate(self._data.keys()):
                self._data[key].append(experience[i])
            self._size += 1

    def random_sample(self, batch_size):
        """
        Generate random sample batch.

        :param batch_size: number of experiences in the generated random sample batch. int.
        :return: generated random sample batch. tuple:(states, actions, next_states).
        """
        if batch_size > self._size:
            raise RuntimeError("No enough experiences available for the given batch size.")
        samples = {
            "states": [],
            "actions": [],
            "next_states": [],
        }

        index = np.random.choice(self._size, batch_size)
        for key in self._data.keys():
            samples[key] = [self._data[key][i] for i in index]

        return tuple(np.asarray(value) for value in samples.values())


def find_optimal_action_value(reward, transition_probability_matrix, num_states, num_actions, discount=0.99, threshold=1e-2):
    """
    calculate the optimal action value function of given enviroment.

    :param reward: reward vector. nparray. (states, )
    :param transition_probability_matrix: transition probability p(st | s, a). nparray. (states, states, actions).
    :param discount: discount rate gamma. float. Default: 0.99
    :param num_states: number of states. int.
    :param num_actions: number of actions. int.
    :param threshold: stop when difference smaller than threshold. float.
    :return: optimal action value function. nparray. (states, actions)
    """

    assert num_states == transition_probability_matrix.shape[0]
    assert num_actions == transition_probability_matrix.shape[-1]

    q = np.zeros((num_states, num_actions))

    while True:
        delta = 0

        for s in range(num_states):
            for a in range(num_actions):
                tp = transition_probability_matrix[s, :, a]
                max_q = np.array([max(q[s_prime, :]) for s_prime in range(num_states)])
                q_prime = np.dot(tp, (reward + discount * max_q))

                diff = abs(q_prime - q[s, a])
                delta = max(delta, diff)

                q[s, a] = q_prime

        if delta < threshold:
            break

    return q


def find_optimal_state_value(reward, transition_probability_matrix, num_states, num_actions, discount=0.99, threshold=1e-2):
    """
    calculate the optimal state value function of given enviroment.

    :param reward: reward vector. nparray. (states, )
    :param transition_probability_matrix: transition probability p(st | s, a). nparray. (states, states, actions).
    :param discount: discount rate gamma. float. Default: 0.99
    :param num_states: number of states. int.
    :param num_actions: number of actions. int.
    :param threshold: stop when difference smaller than threshold. float.
    :return: optimal state value function. nparray. (states)
    """

    v = np.zeros(num_states)

    while True:
        delta = 0

        for s in range(num_states):
            max_v = float("-inf")
            for a in range(num_actions):
                tp = transition_probability_matrix[s, :, a]
                max_v = max(max_v, np.dot(tp, (reward + discount * v)))

            diff = abs(v[s] - max_v)
            delta = max(delta, diff)

            v[s] = max_v

        if delta < threshold:
            break

    return v



def find_policy(q, num_states, num_actions, **kwargs):
    """
    Generate policy according to given action value function Q and method assigned in **kwargs.

    Shape: (states, actions).

    Values in the generated policy matrix represents the probability of taking action A under state S.

    :param q: action value function Q. nparray. (states, actions).
    :param num_states: number of states. int.
    :param num_actions: number of actions. int.
    :param kwargs:
            1. "method": method to generating policy.
                         default: "greedy"
                         other possible values: ("epsilon_greedy", "boltzmann")
            2. "epsilon": need to be assigned when method="epsilon_greedy“。
                          default: 0.2
    :return: policy generated according to the action value function and method specified. nparray. (states, actions)
    """

    policy = np.zeros((num_states, num_actions))

    if kwargs["method"] == "epsilon_greedy":
        if "epsilon" in kwargs.keys():
            epsilon = kwargs["epsilon"]
        else:
            epsilon = 0.2

        policy += epsilon / (num_actions - 1)
        greedy_policy = np.argmax(q, axis=1)
        for i, greedy_action in enumerate(greedy_policy):
            policy[i, greedy_action] = 1 - epsilon

    elif kwargs["method"] == "boltzmann":
        for s in range(num_states):
            sigma_exp_q = np.sum(np.exp(q[s, :]))
            for a in range(num_actions):
                policy[s, a] = np.exp(q[s, a]) / sigma_exp_q

    else:  # generate greedy policy
        for i, greedy_action in enumerate(np.argmax(q, axis=1)):
            policy[i, greedy_action] = 1

    return policy




