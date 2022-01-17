import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from objectworld import ObjectWorld
from utils import find_optimal_action_value, find_optimal_state_value, find_policy, policy_eval


if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objectworld_env")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    grid_size = 32
    num_objects = 50
    num_colors = 2
    wind = 0.3

    discount = 0.8
    threshold = 1e-4

    len_traj = 8
    num_traj = 512

    info_file = os.path.join(save_dir, "README.md")
    with open(info_file, 'w') as f:
        f.write("## Parameters for the enviroment \n\n")
        f.write("*grid size*: {}\n\n".format(grid_size))
        f.write("*number of objects*: {}\n\n".format(num_objects))
        f.write(("*number of colors*: {}\n\n".format(num_colors)))
        f.write("*wind*: {}\n\n".format(wind))
        f.write("\n\n")
        f.write("Objectworld enviroment is saved in ***objectworld_env.txt*** via *pickle.dump()*\n\n")
        f.write("\n\n")
        f.write("## Parameters for ground truth calculation\n\n")
        f.write("*discount*: {}\n\n".format(discount))
        f.write("*threshold*: {:.2e}\n\n".format(threshold))
        f.write("\n\n")
        f.write("## Generate trajectories\n\n")
        f.write("*Length of every trajectory*: {}\n\n".format(len_traj))
        f.write("*Number of trajectories generated*: {}\n\n".format(num_traj))

    objw_env = ObjectWorld(num_objects=num_objects,
                           num_colors=num_colors,
                           grid_size=grid_size,
                           wind=wind)

    env_file = os.path.join(save_dir, "objectworld_env.txt")
    with open(env_file, 'wb') as f:
        pickle.dump(objw_env, f)

    # draw reward matrix
    fig = plt.figure(10)
    # plt.title("Reward matrix")
    plt.imshow(objw_env.reward_vector.reshape(grid_size, grid_size))
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(save_dir, "reward_matrix.png"), dpi=300)

    # calculate optimal state value function
    optim_state_value_vector = find_optimal_state_value(
        reward=objw_env.reward_vector,
        transition_probability_matrix=objw_env.transition_probability_matrix,
        num_states=objw_env.num_states,
        num_actions=objw_env.num_actions,
        discount=discount,
        threshold=threshold
    )
    fig = plt.figure(10)
    # plt.title("State Value")
    plt.imshow(optim_state_value_vector.reshape(grid_size, grid_size))
    plt.colorbar()
    # plt.clim(np.ceil(np.max(optim_state_value_vector)), np.floor(np.min(optim_state_value_vector)))
    plt.show()
    fig.savefig(os.path.join(save_dir, "optimal_state_value.png"), dpi=300)

    np.save(os.path.join(save_dir, "optimal_state_value.npy"), optim_state_value_vector)

    # calculate optimal action value function
    optimal_action_value_vector = find_optimal_action_value(
        reward=objw_env.reward_vector,
        transition_probability_matrix=objw_env.transition_probability_matrix,
        num_states=objw_env.num_states,
        num_actions=objw_env.num_actions,
        discount=discount,
        threshold=threshold
    )

    np.save(os.path.join(save_dir, "optimal_action_value.npy"), optimal_action_value_vector)

    # generate trajectories
    expert_policy = find_policy(optimal_action_value_vector,
                                num_states=objw_env.num_states,
                                num_actions=objw_env.num_actions,
                                method="boltzmann")
    trajectories = objw_env.generate_trajectories(num_traj=num_traj, len_traj=len_traj, policy=expert_policy)

    traj_file = os.path.join(save_dir, "trajectories.npy")
    np.save(traj_file, trajectories)

    # Calculate the ground truth policy sampled from the generated trajectories and save.
    gt_policy = np.zeros((objw_env.num_states, objw_env.num_actions))
    for traj in trajectories:
        for s, a, ns in traj:
            gt_policy[s, a] += 1
    gt_policy[gt_policy.sum(axis=1) == 0] = 1e-6
    gt_policy /= gt_policy.sum(axis=1).reshape(-1, 1)

    np.save(os.path.join(save_dir, "gt_policy.npy"), gt_policy)

    # Calculate the ground truth state value.
    gt_state_value = policy_eval(policy=gt_policy,
                                 reward=objw_env.reward_vector,
                                 transition_probability_matrix=objw_env.transition_probability_matrix,
                                 num_states=objw_env.num_states,
                                 discount=discount)
    np.save(os.path.join(save_dir, "gt_state_value.npy"), gt_state_value)

    fig = plt.figure(10)
    plt.imshow(gt_state_value.reshape(grid_size, grid_size))
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(save_dir, "ground_truth_state_value.png"), dpi=300)
