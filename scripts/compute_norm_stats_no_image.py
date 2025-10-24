"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}



class LeRobotDatasetWrapper(_data_loader.Dataset):
    def __init__(self, dataset = None):
        if dataset is lerobot_dataset.MultiLeRobotDataset:
            self.multi_dataset = dataset
        elif dataset is lerobot_dataset.LeRobotDataset:
            self.dataset = dataset
    
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

def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset_wrapper = LeRobotDatasetWrapper(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset_wrapper,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None, save_dir: str | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    # wrapper here
    # dataset_wrapper = LeRobotDatasetWrapper(dataset)
    
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if save_dir is None:
        output_path = config.assets_dirs / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
    else:
        normalize.save(output_path, save_dir)


if __name__ == "__main__":
    tyro.cli(main)
