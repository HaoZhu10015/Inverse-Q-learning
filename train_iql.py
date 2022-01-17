import os
import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
from objectworld import ObjectWorld
from utils import find_optimal_action_value, find_optimal_state_value, find_policy, policy_eval
from tabular_inverse_q_learning import TabularInverseQLearning


if __name__ == '__main__':

    discount = 0.8
    alpha_r = 1e-4
    alpha_q_sh = 1e-2
    alpha_q = 1e-2

    train_threshold = 1e-3
    env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objectworld_env")
    with open(os.path.join(env_dir, "objectworld_env.txt"), 'rb') as f:
        ow = pickle.load(f)

    num_actions = ow.num_actions
    grid_size = ow.grid_size
    num_states = ow.num_states

    trajectories = np.load(os.path.join(env_dir, "trajectories.npy"))

    iql_env = TabularInverseQLearning(
        num_states=num_states,
        num_actions=num_actions,
        discount=discount,
        alpha_r=alpha_r,
        alpha_q_sh=alpha_q_sh,
        alpha_q=alpha_q
    )

    e = 0
    while True:
        e += 1
        source_r = iql_env.get_reward()
        for traj in trajectories:
            for s, a, ns in traj:
                iql_env.train(s, a, ns)

        delta_r = np.max(np.abs(source_r - iql_env.get_reward()))
        print("\rDelta: {:6f} || Epoch: {}".format(delta_r, e), end="")
        sys.stdout.flush()

        if delta_r < train_threshold:
            break

    # Save trained model
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    iql_results_dir = os.path.join(result_dir, "IQL")
    if not os.path.exists(iql_results_dir):
        os.mkdir(iql_results_dir)

    np.save(os.path.join(iql_results_dir, "q.npy"), iql_env.get_q())
    np.save(os.path.join(iql_results_dir, "r.npy"), iql_env.get_reward())

    pred_policy = find_policy(iql_env.get_q(), num_states=num_states, num_actions=num_actions, method='boltzmann')
    eval_v = policy_eval(
        policy=pred_policy,
        reward=ow.reward_vector,
        transition_probability_matrix=ow.transition_probability_matrix,
        num_states=num_states,
        discount=0.8,
    )

    np.save(os.path.join(iql_results_dir, "policy.npy"), pred_policy)
    np.save(os.path.join(iql_results_dir, "v.npy"), eval_v)

    fig = plt.figure(10)
    plt.imshow(eval_v.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(iql_results_dir, "v.png"), dpi=300)


