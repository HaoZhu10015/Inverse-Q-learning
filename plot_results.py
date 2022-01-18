import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    ow_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objectworld_env")
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    optimal_v = np.load(os.path.join(ow_env_path, "optimal_state_value.npy"))
    gt_v = np.load(os.path.join(ow_env_path, "gt_state_value.npy"))
    iavi_v = np.load(os.path.join(result_path, "IAVI", "v.npy"))
    iql_v = np.load(os.path.join(result_path, "IQL", "v.npy"))

    grid_size = int(optimal_v.shape[0] ** 0.5)

    # calculate EVD for different menthods
    evd_iavi = np.square(optimal_v - iavi_v).mean()
    evd_iql = np.square(optimal_v - iql_v).mean()

    # draw state value results
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(6, 2))

    axes[0].imshow(optimal_v.reshape(grid_size, grid_size))
    axes[0].set_title("Optimal")

    axes[1].imshow(gt_v.reshape(grid_size, grid_size))
    axes[1].set_title("Ground Truth")

    axes[2].imshow(iavi_v.reshape(grid_size, grid_size))
    axes[2].set_title("IAVI")
    axes[2].text(0, 36, "EVD={:.3f}".format(evd_iavi))

    axes[3].imshow(iql_v.reshape((grid_size, grid_size)))
    axes[3].set_title("IQL")
    axes[3].text(0, 36, "EVD={:.3f}".format(evd_iql))

    for a in range(4):
        axes[a].axes.get_xaxis().set_visible(False)
        axes[a].axes.get_yaxis().set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    fig.savefig(os.path.join(result_path, "state_value.png"), dpi=300)

