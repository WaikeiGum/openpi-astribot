import torch
from torch.utils.data import Sampler
import numpy as np

class MultiDatasetWeightedSampler(Sampler):
    """
    支持 MultiLeRobotDataset 的自定义采样器，可动态设定各子集的采样权重。
    """
    def __init__(self, dataset, ratios, num_samples=None, replacement=True):
        """
        dataset: MultiLeRobotDataset 实例
        ratios: list, 每个子集的采样权重，如 [6, 3, 1]
        num_samples: 总采样数（如不设定，默认一轮=总帧数）
        replacement: 是否放回采样，建议为 True
        """
        self.dataset = dataset
        self.ratios = np.array(ratios, dtype=np.float32)
        self.lengths = [len(ds) for ds in dataset._datasets]
        self.total_samples = sum(self.lengths)
        self.cum_lengths = np.cumsum([0] + self.lengths)
        self.num_datasets = len(self.lengths)
        self.replacement = replacement
        self.num_samples = num_samples if num_samples is not None else self.total_samples

        # 预生成全局 idx 列表（每个子集对应的区间），加速采样
        self.idx_pool = [np.arange(self.cum_lengths[i], self.cum_lengths[i+1]) for i in range(self.num_datasets)]

    def set_ratios(self, ratios):
        self.ratios = np.array(ratios, dtype=np.float32)

    def __iter__(self):
        # 采样分布归一化
        probs = self.ratios / self.ratios.sum()
        # 按概率，抽取 num_samples 个子集编号
        chosen_datasets = np.random.choice(self.num_datasets, size=self.num_samples, p=probs)
        # 对每个子集，从自己的索引池采样（放回 or 不放回）
        indices = []
        for i in range(self.num_datasets):
            n = (chosen_datasets == i).sum()
            if n == 0: continue
            pool = self.idx_pool[i]
            if self.replacement:
                selected = np.random.choice(pool, size=n, replace=True)
            else:
                selected = np.random.choice(pool, size=min(n, len(pool)), replace=False)
            indices.extend(selected.tolist())
        np.random.shuffle(indices)  # 打乱顺序
        return iter(indices)

    def __len__(self):
        return self.num_samples
