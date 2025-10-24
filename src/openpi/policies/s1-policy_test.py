import numpy as np
import pytest
import sys

sys.path.append("/home/bai/project/openpi/src/openpi/policies")


from s1_policy import S1Inputs  # 替换为实际模块路径



@pytest.fixture
def sample_data():
    """生成测试数据"""
    return {
        "state": np.random.rand(14),
        "actions": np.random.rand(3, 14),  # action_horizon=3
        "prompt": np.array([72, 101, 108, 108, 111, 0]),  # "Hello\0"
        "images": {
            "cam_high": np.random.rand(3, 256, 256),
            "cam_left_wrist": np.random.rand(3, 256, 256),
            "cam_right_wrist": np.random.rand(3, 256, 256),
        }
    }

def test_actions_prompt_replacement(sample_data, mocker):
    """测试 30% 概率替换 actions 和 prompt"""
    # Mock 随机数生成器固定返回 0.2（触发替换）
    mocker.patch("numpy.random.rand", return_value=0.2)
    
    transform = S1Inputs(action_dim=16)
    inputs = transform(sample_data.copy())
    
    # 验证 actions 被替换为扩展后的 state
    assert inputs["actions"].shape == (3, 16)
    assert np.allclose(inputs["actions"], np.tile(transform.pad_to_dim(sample_data["state"], 16), (3, 1)))
    
    # 验证 prompt 被编码为 "stop\0"
    expected_prompt = np.array([115, 116, 111, 112, 0], dtype=np.int64)
    assert np.array_equal(inputs["prompt"], expected_prompt)

def test_no_replacement(sample_data, mocker):
    """测试不触发替换时的逻辑"""
    # Mock 随机数生成器返回 0.4（不触发替换）
    mocker.patch("numpy.random.rand", return_value=0.4)
    
    transform = S1Inputs(action_dim=16)
    inputs = transform(sample_data.copy())
    
    # 验证 actions 正常填充到 16 维
    assert inputs["actions"].shape == (3, 16)
    
    # 验证 prompt 保留原始值
    assert np.array_equal(inputs["prompt"], sample_data["prompt"])

def test_string_prompt_conversion():
    """测试字符串 prompt 编码逻辑"""
    transform = S1Inputs(action_dim=16)
    data = {
        "state": np.zeros(14),
        "actions": np.zeros((3, 14)),
        "prompt": "open",  # 字符串输入
        "images": {"cam_high": np.zeros((3, 256, 256))}
    }
    
    inputs = transform(data)
    expected_prompt = np.array([111, 112, 101, 110, 0], dtype=np.int64)
    assert np.array_equal(inputs["prompt"], expected_prompt)

def test_image_validation():
    """测试图像名称校验"""
    transform = S1Inputs(action_dim=16)
    invalid_data = {
        "state": np.zeros(14),
        "images": {"invalid_camera": np.zeros((3, 256, 256))}
    }
    
    with pytest.raises(ValueError, match="Expected images to contain"):
        transform(invalid_data)