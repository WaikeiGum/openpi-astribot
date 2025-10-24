import os
import cv2
import h5py
import shutil
import random
import argparse
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Any
from multiprocessing import Process

# from datatools.utils import merge_txt

# 需根据节点CPU算力情况进行调整，H100内部节点建议总进程数16个左右，本机建议4-8个节点
NUM_CPU = 2
PROC_PER_DEVICE = 2
target_size = (384, 384)

cols = [
    'cartesian_so3_command_poses_dict.astribot_arm_left',
    'cartesian_so3_command_poses_dict.astribot_arm_right',
    'cartesian_so3_command_poses_dict.astribot_head',
    'cartesian_so3_command_poses_dict.astribot_torso',
    'cartesian_so3_command_poses_dict.merge_pose'
]
                
def add_last_command(data_list: List[Path]) -> None:
    for data in tqdm(data_list):
        df = pd.read_parquet(data)
        for col in cols:
            new_col = col.replace('command_poses_dict', 'command_state_dict')
            df[new_col] = df[col].shift(1)
            df.at[0, new_col] = df.at[0, col]
        df.to_parquet(data)

    
def run(data_dir: str) -> None:

    
    root_dir = Path(data_dir)
    data_list = list(root_dir.rglob('*.parquet'))
    random.shuffle(data_list)
    print(f"find parquet files: {len(data_list)}")
    
    num_proc = NUM_CPU * PROC_PER_DEVICE
    num_per_proc = int(len(data_list) / num_proc) + 1
    
    process_list = []
    for i in range(NUM_CPU):
        for j in range(PROC_PER_DEVICE):
            process_id = i * PROC_PER_DEVICE + j
            start_idx = process_id * num_per_proc
            end_idx = min(start_idx + num_per_proc, len(data_list))
            slice_data = data_list[start_idx: end_idx]
            proc = Process(target=add_last_command, args=(slice_data,))
            proc.start()
            process_list.append(proc)
    
    for proc in process_list:
        proc.join()
            

def arg_parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', dest='data_dir', type=str, required=True, help='hdf5 file path to be loaded')
    
    return argparser.parse_args()
                
if __name__ == '__main__':
    args = arg_parse()
    run(**vars(args))
