"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
import numpy as np
import tqdm
import tyro
import json

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
from pathlib import Path
import multiprocessing as mp
import dataclasses

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if len(data_config.repo_id_list) > 0:
        # 创建新配置时设置 train_ratios
        import dataclasses
        new_data_config = dataclasses.replace(
            data_config, 
            train_ratios = [1.0] * len(data_config.repo_id_list)
        )
        data_config = new_data_config
    dataset = _data_loader.create_dataset(data_config, config.model, config.seed)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def cal_norm(output_path: Path, config: _config.TrainConfig, max_frames: int | None = None):
    data_config, dataset = create_dataset(config)
    
    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=min(mp.cpu_count()-1, 20),
        # num_workers=0,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    i = 0
    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.new_save(output_path, norm_stats)


def create_dataset_new(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    if len(data_config.repo_id) > 0:
        # 创建新配置时设置 train_ratios
        import dataclasses
        new_data_config = dataclasses.replace(
            data_config, 
            train_ratios=[1.0]*len(data_config.repo_id)
        )
        data_config = new_data_config

    data_len_dict = dict()
    root_dir = Path(data_config.dataset_root)
    for id in data_config.repo_id:
        info_file = root_dir / id / "meta" / "info.json"
        with open(info_file, "r") as f:
            frames = json.load(f)["total_frames"]
        data_len_dict[id] = frames
    return data_config, data_len_dict


def read_data(data_config: _config.DataConfig, data_len_dict: dict, keys: list, max_frames: int | None = None) -> dict:
    def dict_to_numpy(data):
        if isinstance(data, dict):
            return {key: dict_to_numpy(value) for key, value in data.items()}
        elif isinstance(data, list):
            return np.array(data)
        else:
            return data
    
    def get_range(DICT: dict, range: tuple) -> np.array:
        for key in ["std", "mean"]:
            arr = DICT[key]
            selected = arr[range[0] : range[1]]  # 包含索引20
            result = np.zeros_like(arr)
            result[:len(selected)] = selected
            DICT[key] = result
        return DICT
    
    print(data_config.dataset_root)
    # exit(0)

    
    dataset_root = Path(data_config.dataset_root)
    state_dict = {i:[] for i in keys}

    for repo_id in tqdm.tqdm(data_config.repo_id, desc="[compute norm]"):
 
        json_file = dataset_root / repo_id / "meta" / "norm_stats_so3_self_delta_all.json"

        print(json_file)
        
        # 如果数据集的结果不存在则进行计算
        if not json_file.exists():
            print(f"Start to calculate metrics from dataset: {repo_id}")
   
            config = _config.get_config("so3_self_delta_temple")
            new_data_config = dataclasses.replace(config.data, dataset_root=dataset_root)  

            # print(new_data_config)
            # exit(0)
            new_config = dataclasses.replace(config, data=new_data_config)
            config = new_config
            # print(f"use_delta_joint: {config.data.use_delta_joint}")
            config.data.repo_id.clear()
            config.data.repo_id.append(repo_id)
            cal_norm(json_file, config, max_frames)

        with open(json_file, "r") as f:
            temp = json.load(f)["norm_stats"]
            for key in keys:
                state = dict_to_numpy(temp[key])
                state = get_range(state, data_config.data_transforms.inputs[0].joint_range)
                state["count"] = data_len_dict[repo_id]
                state_dict[key].append(state)

    return state_dict
    


def combine_stats(stats_list):
    if not stats_list:
        raise ValueError("stats_list 不能为空")

    # 取第一个数据集的统计信息作为初始值
    combined_mean = np.array(stats_list[0]['mean'])
    combined_var = np.array(stats_list[0]['std'])**2  # 标准差平方得到方差
    total_count = stats_list[0]['count']

    # 逐个合并数据集
    for stats in stats_list[1:]:
        count = stats['count']
        new_mean = np.array(stats['mean'])
        new_var = np.array(stats['std'])**2  # 标准差平方得到方差

        # 计算新的总样本数
        new_total_count = total_count + count

        # 更新均值
        updated_mean = (combined_mean * total_count + new_mean * count) / new_total_count

        # 更新方差（使用 pooled variance 公式），不清楚理论依据，暂时不用
        # updated_var = (
        #     ((total_count - 1) * combined_var + (count - 1) * new_var) / (new_total_count - 1) +
        #     (total_count * count * (combined_mean - new_mean) ** 2) / (new_total_count * (new_total_count - 1))
        # )
        # https://zhuanlan.zhihu.com/p/655234722，D(X) = E(X^2) - E(X)^2
        updated_var = (total_count * (combined_mean ** 2 + combined_var) + count * (new_mean ** 2 + new_var)) / new_total_count - updated_mean ** 2

        # 更新变量
        combined_mean = updated_mean
        combined_var = updated_var
        total_count = new_total_count

    # 计算最终的标准差
    combined_std = np.sqrt(combined_var)

    return {
        'std': combined_std,
        'mean': combined_mean
    }



def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, data_len_dict = create_dataset_new(config)

    print('data config:')
    print(data_config)

    keys = {"state" : "state", "actions" : "actions"}
    value_dict = read_data(data_config, data_len_dict, list(keys.keys()), max_frames)
    
    norm_stats = {}
    for key, stats_list in value_dict.items():
        combined_stats = combine_stats(stats_list)
        norm_stats[keys[key]] = normalize.NormStats(mean=combined_stats['mean'], std=combined_stats['std'])

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)



if __name__ == "__main__":
    # 自动解析命令行参数，传递给main函数
    tyro.cli(main)
