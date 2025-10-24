from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from torch.utils.data import Sampler


import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)



import torch
from torch.utils.data import Sampler
import numpy as np


class MultiDatasetWeightedSampler(Sampler):
    """
    支持 MultiLeRobotDataset 的自定义采样器，可动态平滑过渡采样权重。
    """
    def __init__(
        self, dataset, ratios,
        num_samples=None, replacement=True,
        final_ratios=None, transition_steps=1

    ):
        """
        ratios: 初始采样权重，比如 [8, 2]
        final_ratios: 最终采样权重（默认全1均匀），如 [1,1]
        transition_steps: 经过多少轮（epoch/iter）完成平滑
        """
        self.dataset = dataset._dataset
        self.init_ratios = np.array(ratios, dtype=np.float32)
        self.final_ratios = np.array(final_ratios, dtype=np.float32) if final_ratios is not None else np.ones_like(self.init_ratios)
        self.transition_steps = transition_steps
        self.iter_count = 0  # 第几轮（每个epoch或每次被__iter__调用）

        self.lengths = [len(ds) for ds in dataset._dataset._datasets]
        self.total_samples = sum(self.lengths)
        self.cum_lengths = np.cumsum([0] + self.lengths)
        self.num_datasets = len(self.lengths)
        self.replacement = replacement
        self.num_samples = num_samples if num_samples is not None else self.total_samples
        self.idx_pool = [np.arange(self.cum_lengths[i], self.cum_lengths[i+1]) for i in range(self.num_datasets)]

    def current_ratios(self):
        # 线性插值，滑动过渡
        alpha = min(self.iter_count / self.transition_steps, 1.0)
        return (1 - alpha) * self.init_ratios + alpha * self.final_ratios

    def set_ratios(self, ratios):
        self.init_ratios = np.array(ratios, dtype=np.float32)

    def __iter__(self):
        self.iter_count += 1
        ratios = self.current_ratios()
        probs = ratios / ratios.sum()

        chosen_datasets = np.random.choice(self.num_datasets, size=self.num_samples, p=probs)

        # repo_ids 假设就是 self.dataset.repo_ids 或 self.dataset._datasets[i].repo_id

        chosen_datasets = np.random.choice(self.num_datasets, size=self.num_samples, p=probs)
        # 统计每个子集被采多少次

        from collections import Counter
        counts = Counter(chosen_datasets)
        total = sum(counts.values())


        # 写log到文件
        log_lines = [f"[Sampler] 第{self.iter_count}次采样"]

        # 输出 repo_id 采样比例
        print(f"[Sampler] 第{self.iter_count}次采样")
        for i in range(self.num_datasets):
            # 获取 repo_id

            if hasattr(self.dataset, "repo_ids"):
                repo_id = self.dataset.repo_ids[i]
            elif hasattr(self.dataset._datasets[i], "repo_id"):
                repo_id = self.dataset._datasets[i].repo_id
            else:
                repo_id = str(i)
            ratio = counts[i] / total if total else 0

            log_lines.append(f"    repo_id={repo_id}, count={counts[i]}, ratio={ratio:.2%}")

        # 追加写入日志文件
        with open("/home/bai/project/openpi/train_log/sampler.log", "a") as f:
            for line in log_lines:
                print(line, file=f)
            print(f"    repo_id={repo_id}, count={counts[i]}, ratio={ratio:.2%}")

        indices = []
        for i in range(self.num_datasets):
            n = counts[i]
            if n == 0: continue
            pool = self.idx_pool[i]
            if self.replacement:
                selected = np.random.choice(pool, size=n, replace=True)
            else:
                selected = np.random.choice(pool, size=min(n, len(pool)), replace=False)
            indices.extend(selected.tolist())
        np.random.shuffle(indices)
        return iter(indices)


    def __len__(self):
        return self.num_samples


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access.用于定义 ​数据集 的接口，必须实现的两个功能"""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader.数据加载器必须实现的两个核心方法"""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()
        self.action_horizon = model_config.action_horizon  # 加上 horizon

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = (self.action_horizon, *spec.shape[1:])  # 注意这里加 horizon
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }


# class FakeDataset(Dataset):
#     def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
#         self._num_samples = num_samples
#         self._observation_spec, self._action_spec = model_config.inputs_spec()

#     def __getitem__(self, index: SupportsIndex) -> dict:
#         rng = jax.random.key(index.__index__())

#         def make_from_spec(spec: jax.ShapeDtypeStruct):
#             nonlocal rng
#             rng, data_rng = jax.random.split(rng)
#             # Remove the batch dimension.
#             shape = spec.shape[1:]
#             if spec.dtype == jnp.float32:
#                 return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
#             if spec.dtype == jnp.int32:
#                 return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
#             return jnp.zeros(shape=shape, dtype=spec.dtype)

#         observation = jax.tree.map(make_from_spec, self._observation_spec) # 在 JAX 中，树形结构（tree）是指可以递归展开的数据结构
#         action = jax.tree.map(make_from_spec, self._action_spec)

#         return {
#             **observation.to_dict(),
#             "actions": action,
#         }

#     def __len__(self) -> int:
#         return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if model_config is not None:
        action_horizon = model_config.action_horizon
    else:
        action_horizon = 32

    # import pdb
    # pdb.set_trace()

    if data_config.multi_rerobot:
        return lerobot_dataset.MultiLeRobotDataset( # 多数据集类，用于加载和管理多个 LeRobotDataset 数据集
            repo_ids=data_config.repo_id,
            root=data_config.dataset_root,
            # delta_timestamps={'cartesian_so3_dict.cartesian_pose_state': [t / 30 for t in range(model_config.action_horizon)]},
            delta_timestamps={'cartesian_so3_dict.cartesian_pose_command': [t / 30 for t in range(action_horizon)]},  # 记得给action
            local_files_only=True,
        )
    else:
        # single lerobot
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
            local_files_only=data_config.local_files_only,
        )

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 重新打包数据
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    sampler=None,  # 新增
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    
    
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)


    # 使用权重配比
    if getattr(config, "new_repo_id", None):
        all_repo_ids = data_config.repo_id 
        high_weight_repo_ids = config.new_repo_id  
        
        high_weight_count = len(high_weight_repo_ids)
        low_weight_count = len(all_repo_ids) - high_weight_count
        
        ratios = []
        for repo in all_repo_ids:
            if repo in high_weight_repo_ids:
                ratios.append(8 / high_weight_count if high_weight_count > 0 else 0)
            else:
                ratios.append(2 / low_weight_count if low_weight_count > 0 else 0)
                
        sampler = MultiDatasetWeightedSampler(dataset, ratios=ratios, num_samples=50000)
        shuffle = False
    else:
        sampler = None
        

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
        sampler=sampler,  # 新增
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler=None,  # 新增
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        # import pdb
        # pdb.set_trace()
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            # mp_context = multiprocessing.get_context("spawn")
            mp_context = multiprocessing.get_context("fork")

            # mp_context = multiprocessing.get_context("forkserver")
            

        generator = torch.Generator()
        generator.manual_seed(seed)
         

        
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            prefetch_factor = (8 if num_workers > 0 else None),
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            pin_memory=True,
            sampler=sampler,  # 新增
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
