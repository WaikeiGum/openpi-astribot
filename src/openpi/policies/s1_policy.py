import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_s1_example() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.rand(3, 480, 640).astype(np.float32),
            "cam_low": np.random.rand(3, 480, 640).astype(np.float32),
            "cam_left_wrist": np.random.rand(3, 480, 640).astype(np.float32),
            "cam_right_wrist": np.random.rand(3, 480, 640).astype(np.float32),
        },
    }


@dataclasses.dataclass(frozen=True)
class S1Inputs(transforms.DataTransformFn):
    """Inputs for the S1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    action_horizon: int = 32

    # If true, will adapt the joint and gripper values to match the pi runtime.
    adapt_to_pi: bool = False
    use_so3: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist", "cam_stereo")

    def __call__(self, data: dict) -> dict:
        data = _decode_s1(data, adapt_to_pi=self.adapt_to_pi,use_so3=self.use_so3)
        inputs = {}

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        inputs["state"] = state
        
        # print(data.keys())
        
        if "images" in data.keys():
            in_images = data["images"]
            if set(in_images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

            # Assume that base image always exists.
            base_image = in_images["cam_high"]

            images = {
                "base_0_rgb": base_image,
            }
            image_masks = {
                "base_0_rgb": np.True_,
            }

            # Add the extra images.
            extra_image_names = {
                "left_wrist_0_rgb": "cam_left_wrist",
                "right_wrist_0_rgb": "cam_right_wrist",
            }
            for dest, source in extra_image_names.items():
                if source in in_images:
                    images[dest] = in_images[source]
                    image_masks[dest] = np.True_
                else:
                    images[dest] = np.zeros_like(base_image)
                    image_masks[dest] = np.False_

            inputs.update({
                "image": images,
                "image_mask": image_masks,
            })


        # Actions are only available during training.       
        if "actions" in data:
            actions = np.asarray(data["actions"])

            # import pdb;pdb.set_trace()

            # subtask 结束的时候，要提前截止以防止取到后面的 action   data['sub_task_index']
            sub_task_index_start,sub_task_index_end = data['sub_task_index']
            if sub_task_index_end > actions.shape[0]:
                end_idx = actions.shape[0]
            else:
                end_idx = sub_task_index_end
            last_action =  actions[-1]
            actions = actions[0:end_idx, :]
            to_pad_len = self.action_horizon - actions.shape[0]
            if to_pad_len > 0:
                actions = np.concatenate([actions, np.tile(last_action, (to_pad_len, 1))], axis=0)

            # import pdb;pdb.set_trace()
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)


        inputs['prompt'] = data.get('prompt', None) # update_0305
        return inputs


@dataclasses.dataclass(frozen=True)
class S1Outputs(transforms.DataTransformFn):
    """Outputs for the S1 policy."""

    # If true, will adapt the joint and gripper values to match the pi runtime.
    adapt_to_pi: bool = False
    use_so3: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        # if self.use_so3:
        #     actions = np.asarray(data["actions"][:, 0:20])
        # else:
        #     # actions = np.asarray(data["actions"][:, 0:22])
        #     actions = np.asarray(data["actions"][:, 14:22])
        return {"actions": np.asarray(data["actions"])}
        # return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}

def _get_s1_torso_right_head_joint(ori_joint) -> np.ndarray:
    """
    Get the torso-left-head joint from the original joint.
    torso joint: 2,3,4,5
    right arm joint: 14,15,16,17,18,19,20
    right gripper joint: 21
    head joint: 22,23
    """

    return_dis = [2,3,4,5,  14,15,16,17,18,19,20,  21, 22,23]
    return ori_joint[return_dis]

def _get_s1_torso_left_right_head_joint(ori_joint) -> np.ndarray:
    """
    Get the torso-left-head joint from the original joint.
    torso joint: 2,3,4,5
    left arm joint: 6,7,8,9,10,11,12
    left gripper joint: 13
    right arm joint: 14,15,16,17,18,19,20
    right gripper joint: 21
    head joint: 22,23
    """
    return ori_joint[2:]

def _get_s1_right_joint(ori_joint) -> np.ndarray:
    """
    Get the torso-left-head joint from the original joint.
    torso joint: 2,3,4,5
    left arm joint: 6,7,8,9,10,11,12
    left gripper joint: 13
    right arm joint: 14,15,16,17,18,19,20
    right gripper joint: 21
    head joint: 22,23
    """
    return ori_joint[14:22]

def _joint_flip_mask() -> np.ndarray:
    """Used to convert between s1 and pi joint angles."""
    return np.array([1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1,  1,
                     1, 1, 1, 1, 1, 1, 1,  1,
                     1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # S1 transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the S1 code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by S1.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the S1 code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_s1(data: dict, *, adapt_to_pi: bool = False,use_so3:bool=False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    # if use_so3:
    #     state = state
    # else:
    #     state = _get_s1_right_joint(state)
    # state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        shape = img.shape
        if len(shape) == 3 and shape[0] == 3:
            return einops.rearrange(img, "c h w -> h w c")
        else:
            return img

    if "images" in data.keys():
        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}
        data["images"] = images_dict
    
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the S1 runtime.
        # state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        # actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
