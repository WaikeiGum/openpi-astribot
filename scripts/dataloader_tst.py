import sys

sys.path.append("/home/bai/project")
sys.path.append("/home/bai/project/openpi/.venv/lib/python3.11/site-packages")


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch



data_episode_id = 4
dataset_root_path = "/cognition/lerobot_Oatmeal/"
dataset_repo_id = "250314"


# 加载数据集
dataset = LeRobotDataset(
    root=dataset_root_path + dataset_repo_id,
    repo_id=dataset_repo_id,
    episodes=[data_episode_id],
    local_files_only=True,
)

# 打印数据集信息
print(dataset)

print(dir(dataset))

# 获取第一个样本的 prompt
prompt = dataset[20]["timestamp"]
print(prompt)
print('###########')
print(dataset[0]["joints_dict.joints_position_command"])
print(dataset[0]["joints_dict.joints_position_command"].shape)

print(dataset[0]["images_dict.head.rgb"].shape)


# # 检查数据类型
# print(f"Prompt type: {type(prompt)}")

# # 检查数据结构
# if isinstance(prompt, np.ndarray):
#     print("Prompt is a NumPy array:")
#     print(f"Shape: {prompt.shape}")
#     print(f"Values: {prompt}")
# elif isinstance(prompt, torch.Tensor):
#     print("Prompt is a PyTorch tensor:")
#     print(f"Shape: {prompt.shape}")
#     print(f"Values: {prompt}")
# else:
#     print(f"Prompt is of type: {type(prompt)}")
#     print(f"Value: {prompt}")

# # 解码 prompt（如果是 ASCII 码数组）
# if isinstance(prompt, (torch.Tensor)):
#     prompt_str = ''.join(chr(int(x)) for x in prompt)
#     print(f"Decoded prompt: {prompt_str}")
# else:
#     print("Prompt is not an integer array, cannot decode.")