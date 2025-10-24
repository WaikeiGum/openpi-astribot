"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from pathlib import Path
import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from typing import Protocol, TypeVar
import openpi.policies.s1_policy as s1_policy

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access.用于定义 ​数据集 的接口，必须实现的两个功能"""

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")

class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset, transforms):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index):
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class LeRobotDatasetWrapper(_data_loader.Dataset):
    def __init__(self, dataset = None):
        self.dataset = None
        self.multi_dataset = None
        if isinstance(dataset, lerobot_dataset.MultiLeRobotDataset):
            self.multi_dataset = dataset
        elif isinstance(dataset, lerobot_dataset.LeRobotDataset):
            self.dataset = dataset
        else:
            assert('ERROR')

    def __len__(self):
        if self.multi_dataset is not None:
            return sum([len(dataset) for dataset in self.multi_dataset._datasets])
        elif self.dataset is not None:
            return len(self.dataset)

    def _get_item_multi_dataset(self, idx):
        start_idx = 0
        dataset_idx = 0
        for dataset in self.multi_dataset._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        dataset = self.multi_dataset._datasets[dataset_idx]
        return self._get_item(idx - start_idx, dataset)        


    def _get_item(self, idx, dataset=None):
        if dataset is None:
            dataset = self.dataset
        item = dataset.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if dataset.delta_indices is not None:
            current_ep_idx = dataset.episodes.index(ep_idx) if dataset.episodes is not None else ep_idx
            query_indices, padding = dataset._get_query_indices(idx, current_ep_idx)
            query_result = dataset._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        return item
    
    def __getitem__(self, idx):
        if self.multi_dataset is not None:
            return self._get_item_multi_dataset(idx)
        elif self.dataset is not None:
            return self._get_item(idx)
        else:
            raise ValueError("No dataset found")
        
        
    def save_new_key(self, key_name: str, key_value: list, data_file_dir = None):
        if data_file_dir is None:
            key_list = list(self.dataset.hf_dataset.download_checksums.keys())
            assert len(key_list) == 1
            data_file_dir = key_list[0]
        table = pq.read_table(data_file_dir)
        key_value_array = pa.array(key_value)
        table = table.append_column(key_name, key_value_array)
        pq.write_table(table, data_file_dir)

def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, Dataset]:
    # 1. 完全复用训练时的配置创建逻辑，获取包含所有转换的 data_config
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 2. 使用正确的 data_config 创建原始数据集
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset_wrapper = LeRobotDatasetWrapper(dataset)

    # 3. 应用与训练时相同的 repack 和 data 转换
    dataset = TransformedDataset(
        dataset_wrapper,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None, save_dir: str | None = None):
    output_path = Path('./assets/') / config_name
    print(f"Will save stats to: {output_path}")

    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = True

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True
    
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=1,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    del config.data.repack_transforms.inputs[0].structure['images']

    # i = 0
    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats", ncols=100):       
        # i += 1
        # if i >= 1000:
        #     break
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {}
    for key, s in stats.items():
        stat_obj = s.get_statistics()  # NormStats 对象

        # 如果想保证 length >= 32，可以单独处理每个字段
        def pad_array(arr, target_len=32):
            if arr is None:
                return None
            if arr.shape[0] < target_len:
                pad_len = target_len - arr.shape[0]
                return np.pad(arr, (0, pad_len), mode='constant', constant_values=0.0)
            return arr
        norm_stats[key] = normalize.NormStats(
            mean=pad_array(stat_obj.mean),
            std=pad_array(stat_obj.std),
            q01=pad_array(stat_obj.q01),
            q99=pad_array(stat_obj.q99)
        )

    normalize.save(output_path, norm_stats)
    print('done')

if __name__ == "__main__":
    import sys
    sys.argv = ["openpi/scripts/compute_stats.py", "--config-name", "pi_0_heat_cupcakes_w_fixed_19"]
    tyro.cli(main)