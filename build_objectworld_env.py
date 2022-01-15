import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from objectworld import ObjectWorld
from utils import find_optimal_action_value, find_optimal_state_value


if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objectworld_env")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    grid_size = 32
    num_objects = 50
    num_colors = 3
    wind = 0.3

    discount = 0.8
    threshold = 1e-4

    info_file = os.path.join(save_dir, "info.md")
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

    objw_env = ObjectWorld(num_objects=num_objects,
                           num_colors=num_colors,
                           grid_size=grid_size,
                           wind=wind)

    env_file = os.path.join(save_dir, "objectworld_env.txt")
    with open(env_file, 'wb') as f:
        pickle.dump(objw_env, f)

    # draw reward matrix
    fig = plt.figure(10)
    plt.title("Reward matrix")
    plt.imshow(objw_env.reward_vector.reshape(grid_size, grid_size))
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(save_dir, "reward_matrix.png"), dpi=300)

    # calculate ground truth of state value function
    state_value_vector = find_optimal_state_value(
        reward=objw_env.reward_vector,
        transition_probability_matrix=objw_env.transition_probability_matrix,
        num_states=objw_env.num_states,
        num_actions=objw_env.num_actions,
        discount=discount,
        threshold=threshold
    )
    fig = plt.figure(10)
    plt.title("State Value")
    plt.imshow(state_value_vector.reshape(grid_size, grid_size))
    plt.colorbar()
    plt.clim(np.ceil(np.max(state_value_vector)), np.floor(np.min(state_value_vector)))
    plt.show()
    fig.savefig(os.path.join(save_dir, "state_value.png"), dpi=300)

    state_value_vector_file = os.path.join(save_dir, "gt_state_value.npy")
    np.save(state_value_vector_file, state_value_vector)

    # calculate ground truth of action value function
    action_value_vector = find_optimal_action_value(
        reward=objw_env.reward_vector,
        transition_probability_matrix=objw_env.transition_probability_matrix,
        num_states=objw_env.num_states,
        num_actions=objw_env.num_actions,
        discount=discount,
        threshold=threshold
    )

    action_value_vector_file = os.path.join(save_dir, "gt_action_value.npy")
    np.save(action_value_vector_file, action_value_vector)

