import dataclasses
import pathlib
import logging
import numpy as np
import openpi.training.config as _config
import time

import scripts.serve_policy as _serve_policy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg，避免图形界面冲突
import matplotlib.pyplot as plt

import h5py
import torch
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent



# import saver as _saver
import tyro
import tqdm

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8005
    num_steps: int = 10

def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    ground_truth_actions = []
    predicted_actions = []
    ground_truth_states = []

    test_hdf5_path = "/home/extra/liuruiqiang/openpi/validate_tmp/0211/test_episode_2488.hdf5"
    # test_hdf5_path = '/home/yuran/Projects/openpi/validate_tmp/test_episode_1800.hdf5'
    # prompt_embedding = torch.load(f'/home/extra/liuruiqiang/openpi/lang_emb/pnp_0208/pick_up_the_lemon_and_place_to_the_yellow_plate.pt')['embeddings'][0].float().numpy()

    # test_hdf5_path = '/home/extra/liuruiqiang/openpi/test_episode_1.hdf5'
    # prompt_embedding = torch.load(f'/home/extra/liuruiqiang/openpi/lang_emb/pen_holder/put_the_pen_into_the_pen_holder..pt')['embeddings'][0].float().numpy()

    with h5py.File(test_hdf5_path, 'r') as f:
        qpos = f['joints_dict']['joints_position_state'][:]
        actions = f['joints_dict']['joints_position_command'][:]
        num_steps = qpos.shape[0]
        step = {}
        cache = []
        for step_index in tqdm.trange(1, num_steps):
            if len(cache) == 0:
                state = qpos[step_index]
                # state = np.concatenate([state[2:6], state[-10:]], axis=0)
                
                step['state'] = state
                # step['actions'] = state

                # step['images_dict.head.rgb'] = np.array(f['images_dict']['head']['rgb'][step_index]).astype(np.uint8)
                # step['images_dict.left.rgb'] = np.array(f['images_dict']['left']['rgb'][step_index]).astype(np.uint8)
                # step['images_dict.right.rgb'] = np.array(f['images_dict']['right']['rgb'][step_index]).astype(np.uint8)
                step['prompt'] = 'pick up the lemon and place to the yellow plate'
                # step['image_mask'] = {
                #     'cam_high': np.array(True),
                #     'cam_left_wrist': np.array(True),
                #     'cam_right_wrist': np.array(True),
                # }
                step['images'] = {
                    'cam_high': np.array(f['images_dict']['head']['rgb'][step_index]).astype(np.uint8),
                    'cam_left_wrist': np.array(f['images_dict']['left']['rgb'][step_index]).astype(np.uint8),
                    'cam_right_wrist': np.array(f['images_dict']['right']['rgb'][step_index]).astype(np.uint8),
                }
                # step['prompt_embedding'] = prompt_embedding
                # step['prompt_mask'] = np.ones(prompt_embedding.shape[0], dtype=bool)

                cache = policy.infer(step)['actions'].tolist()[:16]
            predicted_actions.append(cache.pop(0))
            # predicted_actions.append(a)
            action = actions[step_index][14:22]
            ground_truth_states.append(qpos[step_index][14:22])
            # action = np.concatenate([action[2:6], action[-10:]], axis=0)
            ground_truth_actions.append(action)
    predicted_actions = np.array(predicted_actions)
    ground_truth_actions = np.array(ground_truth_actions)
    ground_truth_states = np.array(ground_truth_states)
    n_timesteps, n_dims = ground_truth_actions.shape

    # Create a figure with subplots for each action dimension
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims), sharex=True)
    fig.suptitle('Ground Truth vs Predicted Actions')

    # Plot each dimension
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes

        ax.plot(ground_truth_actions[:, i], label='Ground Truth', color='blue')
        ax.plot(predicted_actions[:, i], label='Predicted', color='red', linestyle='--')
        ax.plot(ground_truth_states[:, i], label='State', color='green', linestyle='--')
        ax.set_ylabel(f'Dim {i+1}')
        ax.legend()

    # Set common x-label
    axes[-1].set_xlabel('Timestep')

    plt.tight_layout()
    plt.savefig('./test.png')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
