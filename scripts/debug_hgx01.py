import os
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
# os.environ['NCCL_NVLS_ENABLE'] = '0'

import jax
from jax.lib import xla_bridge
# jax.distributed.initialize()

print(jax.device_count())
print(jax.local_device_count())
# print(xla_bridge.get_backend().platform)  # 应该输出 'gpu'

xs = jax.numpy.ones(jax.local_device_count())
print(jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs))
