import os
# 导入所有关键组件
import nvidia.cudnn, nvidia.nccl, nvidia.cublas, nvidia.cuda_runtime
import nvidia.cufft, nvidia.curand, nvidia.cusolver, nvidia.cusparse
import nvidia.cuda_nvrtc

libs = [
    nvidia.cudnn, nvidia.nccl, nvidia.cublas, nvidia.cuda_runtime,
    nvidia.cufft, nvidia.curand, nvidia.cusolver, nvidia.cusparse,
    nvidia.cuda_nvrtc
]

paths = []
for lib in libs:
    try:
        p = os.path.dirname(lib.__file__) + '/lib'
        if os.path.exists(p):
            paths.append(p)
    except:
        pass

path_str = ':'.join(paths)
print('\n========== 请复制下面这一整行 ==========')
print('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:' + path_str)
print('=======================================')
