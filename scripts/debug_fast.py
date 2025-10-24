import numpy as np
from transformers import AutoProcessor

# Load the tokenizer from the Hugging Face hub
fast_tokenizer_path = "tokenizers_ckpt/double_grasp_hor_32"
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
# action_data = np.random.rand(1, 32, 8)    # one batch of action chunks
# tokens = tokenizer(action_data)              # 10 , 128
# tokens = np.random.rand(1, 75)
# print(tokens)
# decoded_actions = tokenizer.decode(tokens, time_horizon=32, action_dim=8)

print(tokenizer.action_dim, tokenizer.time_horizon)
# print("tokens:", len(tokens))
# print("tokens:", len(tokens[0]))
# print("decoded actions:", decoded_actions.shape)