from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import torch
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol): # 数据转换函数的接口
    def __call__(self, data: DataDict) -> DataDict:  # 定义数据转换函数的行为
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group: 
    """A group of transforms.​管理数据转换函数"""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order.
        链式处理DataTransformFn
    """

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            # print(data)
            # import time
            # time.sleep(30)
            
            # if type(transform) == DeltaActionsSO3:
            #     import pdb;pdb.set_trace()
            #     import copy

            #     raw = copy.deepcopy(data)
            #     data_delta = transform(data)  # Delta
            #     delta_action_mask = make_bool_mask(9, 9, -1, 9, -1, -3) 
            #     ABS = AbsoluteActionsSO3(mask=delta_action_mask, structure=[9, 9, -1, 9, -1, -3])
            #     recon = ABS(data_delta)       # should be equal to raw
            #     import pdb;pdb.set_trace()
                
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform.
    合并为一个 ​组合式数据转换函数"""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.
        将输入的字典数据 ​重新打包 为一个新的字典结构
    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.
    使用 / 作为分隔符，支持嵌套字典的扁平化路径
    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]  # PyTree 是 JAX 中的一个概念，表示 ​树形结构，可以是嵌套的字典、列表、元组等

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure) # 遍历 self.structure 的每个叶子节点（即路径），从 flat_item 中提取对应的值，并构建新的字典


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = True
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        # print('len(x) is ', len(x))
        assert stats.q01 is not None
        assert stats.q99 is not None

        norm = (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
        # if (norm > 2).any():
        #     print(norm)
        #     import pdb;pdb.set_trace()

        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data

@dataclasses.dataclass(frozen=True)
class GetDimRange(DataTransformFn):
    key: list[str]
    dim: list[list[int]]

    def __call__(self, data: DataDict) -> DataDict:
        # print('self.dim ', self.dim)
        # import pdb
        # pdb.set_trace()

        for k, d in zip(self.key, self.dim):
            data[k] = data[k][...,d]
            # print(len(data[k]))                          # 29

        return data

@dataclasses.dataclass(frozen=True)
class PadState(DataTransformFn):
    action_dim: int
    def __call__(self, data: DataDict) -> DataDict:
        
        data['state'] = pad_to_dim(data['state'], self.action_dim)
        return data

@dataclasses.dataclass(frozen=True)
class PadAction(DataTransformFn):
    action_dim: int
    def __call__(self, data: DataDict) -> DataDict:
        data['actions'] = pad_to_dim(data['actions'], self.action_dim)
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


import dataclasses
from typing import Sequence
import numpy as np

def so3_6d_to_matrix(so3_6d):
    x_raw = so3_6d[..., 0:3]
    y_raw = so3_6d[..., 3:6]

    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=-2)
    return mat

def matrix_to_so3_6d(matrix):
    r1 = matrix[..., 0, :]
    r2 = matrix[..., 1, :]
    return np.concatenate([r1, r2], axis=-1)



@dataclasses.dataclass(frozen=True)
class DeltaActions_so3(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[int] | None
    axis: str = "self"
    state_trans_head: bool = False
    def compute_state_xyz_so3(self,P1,P3 = None):

        R1 = so3_to_matrix(P1[3:]).reshape(3, 3)
        T_1 = np.array(P1[:3])
        if P3 is not None:
            P3 = np.squeeze(P3)
            if self.axis == "head_so3":
                R3 = so3_to_matrix(P3[3:]).reshape(3, 3)
                T_3 = np.array(P3[:3])
                R1 = R3.T @ R1
                T_1 = R3.T @ (T_1 - T_3)

        return np.concatenate([T_1 , matrix_to_so3(R1)])

    def compute_change_xyz_so3(self,P1, P2,P3 =None):

        # print(P1, P1.sum())
        # print(P2, P2.sum())
        if P1[3:].sum() == 0:
            return P1

        R1 = so3_to_matrix(P1[3:]).reshape(3, 3)
        R2 = so3_to_matrix(P2[3:]).reshape(3, 3)

        T_1 = np.array(P1[:3])
        T_2 = np.array(P2[:3])

        if self.axis == "self":
            # 计算旋转矩
            R_d = R1.T @ R2
            T_d = R1.T @ (T_2 - T_1)


        else:
            R_d = R2 @ R1.T
            T_d = (T_2 - T_1)
            if P3 is not None:
                P3 = np.squeeze(P3)
                if self.axis == "head_so3":
                    R3 = so3_to_matrix(P3[3:]).reshape(3, 3)
                    R_d = R3.T @ R_d @ R3
                    T_d = R3.T @ T_d


                elif self.axis == "head_quat":
                    R3 = quat_to_matrix(P3[3:]).reshape(3, 3)
                    R_d = R3.T @ R_d @ R3
                    T_d = R3.T @ T_d

        return np.concatenate([T_d , matrix_to_so3(R_d)])

    def compute_change_xyz_so3_list(self, P1, P2, P3=None):
        so3_delta_list = []
        for P2_now in P2:
            so3_delta_list += [self.compute_change_xyz_so3(P1, P2_now,P3)]
        so3_delta = np.array(so3_delta_list)
        return so3_delta


    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            """
            if self.state_trans_head:
                state = data["state"]
                last_idx = 0
                head_poses = data["head_so3_poses"]
                for idx,num in enumerate(self.mask):
                    cur_idx = last_idx + abs(num)
                    if num == 9:
                        state[...,last_idx:cur_idx] = self.compute_state_xyz_so3(state[...,last_idx:cur_idx],head_poses)
                    last_idx = cur_idx
                data["state"] = state"""
            return data

        state, actions = data["state"], data["actions"]
        last_idx = 0
        head_poses = None
        """
        if self.axis == "head_so3" :
            head_poses = data["head_so3_poses"]
        elif self.axis == "head_quat" :
            head_poses = data["head_quat_poses"]"""

        for idx,num in enumerate(self.mask):
            cur_idx = last_idx + abs(num)
            if num == 9:
                actions[...,last_idx:cur_idx] = self.compute_change_xyz_so3_list(state[...,last_idx:cur_idx], actions[...,last_idx:cur_idx], head_poses)
                # if self.state_trans_head:
                #     state[...,last_idx:cur_idx] = self.compute_state_xyz_so3(state[...,last_idx:cur_idx],head_poses)


            last_idx = cur_idx
        data["actions"] = actions
        data["state"] = state

        return data

def so3_to_matrix( so3):
    R1_partial = np.array(so3).reshape(2, 3)
    # 计算第三行
    R1_partial[0] /= np.linalg.norm(R1_partial[0])  # 归一化
    R1_partial[1] /= np.linalg.norm(R1_partial[1])  # 归一化
    R1_third_row = np.cross(R1_partial[0], R1_partial[1])
    R1_third_row /= np.linalg.norm(R1_third_row)  # 归一化

    # 构造完整的旋转矩阵
    R1 = np.vstack((R1_partial, R1_third_row))
    return R1

def quat_to_matrix( quat):
    # 计算旋转矩阵
    quat = np.array(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)  # 归一化，确保是单位四元数
    x, y, z, w = quat
    R = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                  [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
                  [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]])
    return R

def matrix_to_so3( R1):
    R1 = np.array(R1).reshape(3, 3)
    R1_partial = R1[:2]
    return R1_partial.flatten()


@dataclasses.dataclass(frozen=True)
class AbsoluteActions_so3(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[int] | None
    axis: str = "self"

    def compute_add_xyz_so3(self, P1, Pd,P3=None):

        R1 = so3_to_matrix(P1[3:]).reshape(3, 3)
        R_d = so3_to_matrix(Pd[3:]).reshape(3, 3)

        T_1 = np.array(P1[:3])
        T_d = np.array(Pd[:3])

        if self.axis == "self":
            # 计算旋转矩
            R2 = R1 @ R_d
            T_2 = R1 @ T_d + T_1
        else:
            if P3 is not None:
                if self.axis == "head_so3":
                    R3 = so3_to_matrix(P3[3:]).reshape(3, 3)
                    # print(R_d)
                    R_d = R3 @ R_d @ R3.T
                    T_d = R3 @ T_d
                elif self.axis == "head_quat":
                    R3 = quat_to_matrix(P3[3:]).reshape(3, 3)
                    R_d = R3 @ R_d @ R3.T
                    T_d = R3 @ T_d
            R2 = R_d @ R1
            T_2 = T_d + T_1


        return np.concatenate([T_2, matrix_to_so3(R2)])

    def compute_add_xyz_so3_list(self, P1, Pd,P3=None):
        P2_list = []
        for Pd_now in Pd:
            P2_list += [self.compute_add_xyz_so3(P1, Pd_now,P3)]
        P2 = np.array(P2_list)
        return P2

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        last_idx = 0

        head_poses = None
        if self.axis == "head_so3" :
            head_poses = data["head_so3_poses"]
        elif self.axis == "head_quat" :
            head_poses = data["head_quat_poses"]

        for idx,num in enumerate(self.mask):
            cur_idx = last_idx + abs(num)
            if num == 9:
                actions[...,last_idx:cur_idx] = self.compute_add_xyz_so3_list(state[...,last_idx:cur_idx], actions[...,last_idx:cur_idx],head_poses)

            last_idx = cur_idx
        data["actions"] = actions

        return data



@dataclasses.dataclass(frozen=True)
class DeltaActionsSO3(DataTransformFn):
    """
    针对 block 结构如 [9, 9, -1, 9, -1, -3] 的 delta 动作转换
    """
    mask: Sequence[bool] | None
    structure: Sequence[int] = dataclasses.field(default_factory=lambda: [9, 9, -1, 9, -1, -3])  # 默认你的结构

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
    
        mask = np.asarray(self.mask)
        assert actions.shape[0] == mask.shape[0], "mask长度应等于动作维数"

        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()
        if isinstance(state, torch.Tensor):
            state = state.numpy()

        new_actions = actions.copy()

        idx = 0
        for block in self.structure:

            if block > 0:
                # 处理 9 维 block（3 xyz + 6 so3）
                # xyz: 3维
                new_actions[..., idx:idx+3] -= state[..., idx:idx+3]
                # so3: 6维 delta
                so3_act = actions[..., idx+3:idx+9]
                so3_state = state[..., idx+3:idx+9]
                mat_act = so3_6d_to_matrix(so3_act)
                mat_state = so3_6d_to_matrix(so3_state)
                delta_mat = np.matmul(mat_act, np.swapaxes(mat_state, -2, -1))
                delta_6d = matrix_to_so3_6d(delta_mat)
                new_actions[..., idx+3:idx+9] = delta_6d
                idx += block
            elif block < 0:
                # 负值：跳过这些维度或者直接mask相减
                for j in range(idx, idx + abs(block)):
                    if mask[j]:
                        new_actions[..., j] -= state[..., j]
                idx += abs(block)

        data["actions"] = new_actions
        return data

@dataclasses.dataclass(frozen=True)
class AbsoluteActionsSO3(DataTransformFn):
    """
    delta action 还原为 absolute action，xyz直接加，so3(6d)用李群乘法还原，支持block结构。
    """
    mask: Sequence[bool] | None
    structure: Sequence[int] = dataclasses.field(default_factory=lambda: [9, 9, -1, 9, -1, -3])

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        assert actions.shape[1] == mask.shape[0], "mask长度应等于动作维数"

        new_actions = actions.copy()
        idx = 0
        for block in self.structure:
            if block > 0:
                # xyz: 3维，绝对动作 = delta + state
                new_actions[..., idx:idx+3] += state[..., idx:idx+3]
                # so3: 6维，delta(6d)+state(6d) -> abs(6d)
                delta_6d = actions[..., idx+3:idx+9]
                state_6d = state[..., idx+3:idx+9]
                delta_mat = so3_6d_to_matrix(delta_6d)
                state_mat = so3_6d_to_matrix(state_6d)
                abs_mat = np.matmul(delta_mat, state_mat)  # 注意顺序
                abs_6d = matrix_to_so3_6d(abs_mat)
                new_actions[..., idx+3:idx+9] = abs_6d
                idx += block
            elif block < 0:
                # 负数block，直接mask为True部分相加
                for j in range(idx, idx + abs(block)):
                    if mask[j]:
                        new_actions[..., j] += state[..., j]
                idx += abs(block)

        data["actions"] = new_actions
        return data


# @dataclasses.dataclass(frozen=True)
# class DeltaActions(DataTransformFn):
#     """Repacks absolute actions into delta action space."""

#     # Boolean mask for the action dimensions to be repacked into delta action space. Length
#     # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
#     # See `make_bool_mask` for more details.
#     mask: Sequence[bool] | None

#     def __call__(self, data: DataDict) -> DataDict:
#         if "actions" not in data or self.mask is None:
#             return data

#         state, actions = data["state"], data["actions"]
#         if actions.ndim == 1:
#             assert len(state) == actions.shape[0], "State维度必须与action的第二维一致"
#         else:
#             assert len(state) == actions.shape[1], "State维度必须与action的第二维一致"
#         # print("state.shape:", np.shape(state), "actions.shape:", np.shape(actions))


#         mask = np.asarray(self.mask)
#         dims = mask.shape[-1]      # D
#         state = np.asarray(state)  # (T, D) or (..., T, D)；此处按 (T, D) 用
#         actions = np.asarray(actions)

#         T = state.shape[0]
#         D = dims



#         rhs = np.expand_dims(np.where(mask, state[..., :D], 0), axis=-2)  # (T,1,D)
#         actions[..., :D] = actions[..., :D] - rhs
        
#         data["actions"] = actions
        
#         return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        assert len(state) == actions.shape[1], "State维度必须与action的第二维一致"

        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        
        return data



@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            prompt = self.default_prompt
        if prompt is None:
            raise ValueError("Prompt is required")

        if isinstance(prompt, np.ndarray):
            string = ''.join(chr(num) for num in prompt)
            prompt = string.split('\0')[0]  # 截取结束符号前的部分
            
        if isinstance(prompt, torch.Tensor):  # 检查是否为 PyTorch 张量
            string = ''.join(chr(num) for num in prompt.cpu().numpy())  # 先转换为 numpy 数组
            prompt = string.split('\0')[0]  # 截取结束符号前的部分

        assert isinstance(prompt, str)

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        
        if isinstance(prompt, torch.Tensor):
            string = ''.join(chr(num) for num in prompt.numpy())
            prompt = string.split('\0')[0]  # 截取结束符号前的部分
            
        assert isinstance(prompt, str)

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )



# class DebugPrintTransform(DataTransformFn):
#     """打印数据形状、数据类型和部分内容的调试插件"""

#     def __call__(self, data: DataDict) -> DataDict:
#         print("\n[DEBUG] 数据内容:")
#         for key, value in data.items():
#             # 打印基础信息
#             if isinstance(value, (np.ndarray, torch.Tensor)):
#                 print(f"{key}: shape={value.shape}, dtype={value.dtype}, sample={value[..., :2]}")
#             elif isinstance(value, (list, tuple)):
#                 print(f"{key}: len={len(value)}, sample={value[:2]}")
#             else:
#                 print(f"{key}: type={type(value)}, value={str(value)[:50]}...")
#         return data  # 原样返回数据，不修改
    
    
    
# class DebugPrintTransform(DataTransformFn):
#     """仅打印 action 数据的调试插件"""

#     def __call__(self, data: DataDict) -> DataDict:
#         if "actions" not in data:
#             return data  # 如果没有 action 字段，直接返回

#         action = data["actions"]
#         print("\n[DEBUG] Action 数据内容:")

#         # 检查是否为 numpy 数组或 PyTorch 张量
#         if isinstance(action, (np.ndarray, torch.Tensor)):
#             print(f"Shape: {action.shape}, Dtype: {action.dtype}")

#             # 如果维度是 (32, 32)，提取前 3 行和前 8 列
#             if action.ndim == 2 and action.shape[0] == 32 and action.shape[1] == 32:
#                 sliced = action[:3, :8]  # 切片操作
#                 print("前 3 行、前 8 列的数据示例:")
#                 print(sliced if isinstance(sliced, np.ndarray) else sliced.cpu().numpy())
#             else:
#                 # 其他维度打印部分数据
#                 print("数据示例（前 2 个元素）:")
#                 print(action[..., :2] if isinstance(action, np.ndarray) else action[..., :2].cpu().numpy())
#         else:
#             print(f"Type: {type(action)}, Value: {str(action)[:50]}...")

#         return data  # 原样返回数据
    
class DebugPrintTransform(DataTransformFn):
    """打印 action 和 prompt 数据的调试插件"""

    def __call__(self, data: DataDict) -> DataDict:
        print("\n[DEBUG] 数据内容:")

        # 1. 打印 action
        if "actions" in data:
            action = data["actions"]
            print("[DEBUG] Action 数据:")
            if isinstance(action, (np.ndarray, torch.Tensor)):
                print(f"  Shape: {action.shape}, Dtype: {action.dtype}")
                if action.ndim == 2 and action.shape == (32, 32):
                    sliced = action[:3, :8].cpu().numpy() if isinstance(action, torch.Tensor) else action[:3, :8]
                    print("  前 3 行、前 8 列:\n", sliced)
                else:
                    sample = action[..., :2].cpu().numpy() if isinstance(action, torch.Tensor) else action[..., :2]
                    print("  部分数据:\n", sample)
            else:
                print(f"  Type: {type(action)}, Value: {str(action)[:50]}...")

        # 2. 打印 prompt
        if "prompt" in data:
            prompt = data["prompt"]
            print("[DEBUG] Prompt 数据:")
            if isinstance(prompt, (str, list, np.ndarray, torch.Tensor)):
                if isinstance(prompt, str):
                    print(f"  Text: {prompt[:50]}...")  # 截断长文本
                elif isinstance(prompt, (list, tuple)):
                    print(f"  Length: {len(prompt)}, Sample: {prompt[:2]}")
                elif isinstance(prompt, (np.ndarray, torch.Tensor)):
                    prompt_np = prompt.cpu().numpy() if isinstance(prompt, torch.Tensor) else prompt
                    print(f"  Shape: {prompt_np.shape}, Dtype: {prompt_np.dtype}, Sample: {prompt_np[:2]}")
            else:
                print(f"  Type: {type(prompt)}, Value: {str(prompt)[:50]}...")

        return data  # 原样返回数据