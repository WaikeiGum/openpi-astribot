import numpy as np
import dataclasses
from tqdm import tqdm

from openpi.models import tokenizer as _tokenizer
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.training.config as _train_config
import openpi.transforms as _transforms
from openpi.training.data_loader import create_dataset, transform_dataset

@dataclasses.dataclass
class Args:
    train_config: str = "id11"
    save_dir: str = None


def get_data_without_image(idx, multi_dataset: lerobot_dataset):
    start_idx = 0
    dataset_idx = 0
    for dataset in multi_dataset._datasets:
        if idx >= start_idx + dataset.num_frames:
            start_idx += dataset.num_frames
            dataset_idx += 1
            continue
        break
    else:
        raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
    dataset = multi_dataset._datasets[dataset_idx]
    item = dataset.hf_dataset[idx - start_idx]
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

def main(args: Args) -> None:
    train_config = _train_config.get_config(args.train_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = create_dataset(data_config, train_config.model)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    # actions = np.random.rand(4000, 50, 14)
    actions = []
    for frame_id in tqdm(range(dataset.num_frames)):
        # frame = dataset[frame_id]
        frame = get_data_without_image(frame_id, dataset)
        for transform in data_config.repack_transforms.inputs:
            frame = transform(frame)
        for transform in data_config.data_transforms.inputs:
            frame = transform(frame)
        normalier = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        frame = normalier(frame)
        actions.append(np.asarray(frame["actions"]))
    # save actions
    np.save("actions.npy", np.asarray(actions))
    tokenizer._fast_tokenizer.fit(actions, vocab_size=1024)
    tokenizer._fast_tokenizer.save_pretrained("tokenizers_ckpt/double_grasp_hor_32_action_dim_8/")


if __name__ == "__main__":
    main(Args())