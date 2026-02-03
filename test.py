# import os
# # 导入所有关键组件
# import nvidia.cudnn, nvidia.nccl, nvidia.cublas, nvidia.cuda_runtime
# import nvidia.cufft, nvidia.curand, nvidia.cusolver, nvidia.cusparse
# import nvidia.cuda_nvrtc

# libs = [
#     nvidia.cudnn, nvidia.nccl, nvidia.cublas, nvidia.cuda_runtime,
#     nvidia.cufft, nvidia.curand, nvidia.cusolver, nvidia.cusparse,
#     nvidia.cuda_nvrtc
# ]

# paths = []
# for lib in libs:
#     try:
#         p = os.path.dirname(lib.__file__) + '/lib'
#         if os.path.exists(p):
#             paths.append(p)
#     except:
#         pass

# path_str = ':'.join(paths)
# print('\n========== 请复制下面这一整行 ==========')
# print('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:' + path_str)
# print('=======================================')


import torch
import torchvision.models as models

print("正在尝试连接 PyTorch 服务器下载 MobileNetV3-Small...")
# 这一步会自动下载权重到你的 ~/.cache/torch/hub/checkpoints/ 目录
net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

print("✅ 下载并加载成功！")
print(f"总参数量: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M") 
# 你应该会看到大约 2.54 M，这就是我们要的轻量级模型