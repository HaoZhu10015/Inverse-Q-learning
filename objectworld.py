"""
Implements the ObjectWorld MDP described in:
Nonlinear Inverse Reinforcement Learning with Gaussian Processes, Levine et al. 2011


Hao Zhu, 2021
"""

import numpy as np
import math
from itertools import product
from matplotlib import pyplot as plt
import os

class OWObject:
    """
    Objectworld object.
    """

    def __init__(self, inner_color, outer_color):
        """
        :param inner_color: Inner color of object. int.
        :param outer_color: Outer color of object. int.
        """

        self.inner_color = inner_color
        self.outer_color = outer_color

    def __str__(self):
        return "<OWObject (Inner: {}) (Outer: {})>".format(self.inner_color, self.outer_color)


class ObjectWorld:
    def __init__(self, num_objects, num_colors, grid_size, wind):
        """
        :param num_objects: total number of objects in the enviroment. int.
        :param num_colors: total number of colors. int.
        :param grid_size: objectworld grid size. int.
        :param wind: probability of taking a random action. float.
        """

        self.num_objects = num_objects
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.num_states = grid_size ** 2

        self.actions = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        self.num_actions = len(self.actions)
        self.wind = wind

        # Generate objects in the enviroment.
        self.objects = {}
        for i in range(self.num_objects):
            obj = OWObject(
                np.random.randint(self.num_colors),
                np.random.randint(self.num_colors)
            )

            while True:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        # Calculate the transition probability matrix. (states, states, actions).
        self.transition_probability_matrix = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.num_actions)]
              for j in range(self.num_states)]
             for i in range(self.num_states)])

        # Calculate the reward vector. (states, ).
        self.reward_vector = np.array([self._reward(s) for s in range(self.num_states)])

    def _is_neighbour(self, s1, s2):
        """
        Judge if state_1 and state_2 are neighbours.

        :param s1: state_1, int.
        :param s2: state_2, int.
        :return: if state_1 and state_2 are neighbours. bool.
        """
        x1, y1 = self.int_to_state(s1)
        x2, y2 = self.int_to_state(s2)

        return True if np.abs(x1 - x2) + np.abs(y1 - y2) <= 1 else False

    def _is_corner(self, s):
        """
        Judge if a state is in the corner.

        :param s: state. int.
        :return: if the state is in the corner. bool.
        """
        x, y = self.int_to_state(s)
        return True if (x, y) in ((0, 0),
                                  (0, self.grid_size-1),
                                  (self.grid_size-1, 0),
                                  (self.grid_size-1, self.grid_size-1)) else False

    def _is_edge(self, s):
        """
        Judge if a state in on the edge (INCLUDING CORNER!).

        :param s: state. int.
        :return: if the state is on the edge. bool.
        """
        x, y = self.int_to_state(s)

        return True if x in (0, self.grid_size-1) or y in (0, self.grid_size-1) else False

    def _is_in_the_world(self, x, y):
        """
        Judge if a stage is inside the enviroment.

        :param x: x coordinate of state. int.
        :param y: y coordinate of state. int.
        :return: if the state is inside the enviroment. bool.
        """

        return True if 0 <= x < self.grid_size and 0 <= y < self.grid_size else False

    def _transition_probability(self, s, st, a):
        """
        Calculate the transition probability p(st | s, a)

        :param s: initial state. int.
        :param st: target state. int.
        :param a: action to take. int.
        :return: transition probability. float
        """

        if not self._is_neighbour(s, st):
            return 0.0

        x, y = self.int_to_state(s)
        xt, yt = self.int_to_state(st)
        dx, dy = self.int_to_action(a)

        if not self._is_edge(s):
            if (x+dx, y+dy) == (xt, yt):
                return 1-self.wind + self.wind / self.num_actions
            else:
                return self.wind / self.num_actions

        else:
            if self._is_corner(s):
                if s == st:
                    if (dx, dy) == (0, 0) or not self._is_in_the_world(x+dx, y+dy):
                        return 1-self.wind + 3 * self.wind / self.num_actions
                    else:
                        return 3 * self.wind / self.num_actions
                else:
                    if (x+dx, y+dy) == (xt, yt):
                        return 1-self.wind + self.wind / self.num_actions
                    else:
                        return self.wind / self.num_actions
            else:
                if s == st:
                    if (dx, dy) == (0, 0) or not self._is_in_the_world(x+dx, y+dy):
                        return 1-self.wind + 2 * self.wind / self.num_actions
                    else:
                        return 2 * self.wind / self.num_actions
                else:
                    if (x + dx, y + dy) == (xt, yt):
                        return 1 - self.wind + self.wind / self.num_actions
                    else:
                        return self.wind / self.num_actions

    def get_state_feature_length(self, discrete):
        return 2 * self.num_colors * self.grid_size if discrete else 2 * self.num_colors

    def feature_vector(self, s, discrete=True):
        """
        calculate feature vector of states.

        :param s: state. int.
        :param discrete: using discrete or continuous feature. bool.
        :return: feature vector. nparray.
        """

        x0, y0 = self.int_to_state(s)

        nearest_inner_color = {}
        nearest_outer_color = {}

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in self.objects:
                    inner_color = self.objects[x, y].inner_color
                    outer_color = self.objects[x, y].outer_color
                    dist = math.hypot(x-x0, y-y0)

                    if inner_color not in nearest_inner_color:
                        nearest_inner_color[inner_color] = dist
                    elif dist < nearest_inner_color[inner_color]:
                        nearest_inner_color[inner_color] = dist

                    if outer_color not in nearest_outer_color:
                        nearest_outer_color[outer_color] = dist
                    elif dist < nearest_outer_color[outer_color]:
                        nearest_outer_color[outer_color] = dist

        for c in range(self.num_colors):
            if c not in nearest_inner_color:
                nearest_inner_color[c] = 0.0
            if c not in nearest_outer_color:
                nearest_outer_color[c] = 0.0

        if discrete:
            state = np.zeros(2 * self.num_colors * self.grid_size)
            i = 0
            for c in range(self.num_colors):
                for n in range(1, self.grid_size+1):
                    state[i] = int(nearest_inner_color[c] < n)
                    state[i + self.num_colors * self.grid_size] = int(nearest_outer_color[c] < n)
                    i += 1

        else:
            state = np.zeros(2 * self.num_colors)
            i = 0
            for c in range(self.num_colors):
                state[i] = nearest_inner_color[c]
                state[i + self.num_colors] = nearest_outer_color[c]

        return state

    def feature_matrix(self, discrete=True):
        """
        Generate the feature of all states.

        :param discrete: using the discrete or continuous feature. bool.
        :return: feature matrix of all states. nparray.
        """

        return np.array([self.feature_vector(s, discrete) for s in range(self.num_states)])

    def _reward(self, s):
        """
        Calculate the reward for given state.

        :param s: state. int.
        :return: one of 1, -1, and 0. int.
        """

        xs, ys = self.int_to_state(s)

        within_c0 = False
        within_c1 = False

        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            xt = xs + dx
            yt = ys + dy

            if 0 <= xt < self.grid_size and 0 <= yt < self.grid_size and (xt, yt) in self.objects:
                if abs(dx) + abs(dy) <= 3 and self.objects[xt, yt].outer_color == 0:
                    within_c0 = True
                if abs(dx) + abs(dy) <= 2 and self.objects[xt, yt].outer_color == 1:
                    within_c1 = True

        if within_c0 and within_c1:
            return 1
        elif within_c0 and not within_c1:
            return -1
        else:
            return 0

    def int_to_state(self, i):
        """
        Transform integer expression of state into state coordinate.

        :param i: integer expression of state. int. [0, grid_size^2].
        :return: coordinate expression of state. (sx, sy).
        """

        return (i % self.grid_size, i // self.grid_size)

    def int_to_action(self, i):
        """
        Transform integer expression of action into coordinate change.

        :param i: integer expression of action to take. int. [0, 4].
        :return: coordinate expression of action to take. (dx, dy).
        """

        return self.actions[i]

    def step(self, s, a):
        """
        Given state and action, calculate the nest state.

        :param s: start state. int.
        :param a: action to take. int.
        :return: next state. int.
        """

        return np.random.choice(self.num_states, p=self.transition_probability_matrix[s, :, a])

    def generate_trajectories(self, num_traj, len_traj, policy, initiate_s=None):
        """
        generate trajectories.

        :param num_traj: number of trajectories to generate. int.
        :param len_traj: length of every generated trajectory. int.
        :param policy: policy for choosing the action. nparray. (states, actions).
        :param initiate_s: start state of every trajectory. int.
        :return: generated trajectories. nparray. (num_traj, len_traj, 3: (state, action, next_state))
        """
        trajectories = []

        for _ in range(num_traj):
            trajectory = []

            if initiate_s is not None:
                s = initiate_s
            else:
                s = np.random.randint(self.num_states)

            for i in range(len_traj):
                a = np.random.choice(self.num_actions, p=policy[s])
                ns = self.step(s, a)

                trajectory.append([s, a, ns])
                s = ns

            trajectories.append(trajectory)

        return np.asarray(trajectories)

    def draw_objectworld(self):
        raise NotImplementedError

    def draw_reward_map(self, fig_size=10, save_path=None):
        fig = plt.figure(fig_size)
        plt.imshow(self.reward_vector.reshape((self.grid_size, self.grid_size)))
        plt.colorbar()
        plt.show()

        if save_path is not None:
            fig.savefig(os.path.join(save_path, 'reward_map.png'))
