import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
import cv2

from ..config import config
from . import _gradient

def mround(x):
    x_ = x.copy()
    idx = (x - np.floor(x)) >= 0.5
    x_[idx] = np.floor(x[idx]) + 1
    idx = ~idx
    x_[idx] = np.floor(x[idx])
    return x_

class Feature:
    def init_size(self, img_sample_sz, cell_size=None):
        if cell_size is not None:
            max_cell_size = max(cell_size)
            new_img_sample_sz = (1 + 2 * mround(img_sample_sz / ( 2 * max_cell_size))) * max_cell_size
            feature_sz_choices = np.array([(new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1)) // x for x in cell_size])
            num_odd_dimensions = np.sum((feature_sz_choices % 2) == 1, axis=(0,1))
            best_choice = np.argmax(num_odd_dimensions.flatten())
            img_sample_sz = mround(new_img_sample_sz + best_choice)

        self.sample_sz = img_sample_sz
        self.data_sz = [img_sample_sz // self._cell_size]
        return img_sample_sz

    def _sample_patch(self, im, pos, sample_sz, output_sz):
        pos = np.floor(pos)
        sample_sz = np.maximum(mround(sample_sz), 1)
        xs = np.floor(pos[1]) + np.arange(0, sample_sz[1]+1) - np.floor((sample_sz[1]+1)/2)
        ys = np.floor(pos[0]) + np.arange(0, sample_sz[0]+1) - np.floor((sample_sz[0]+1)/2)
        xmin = max(0, int(xs.min()))
        xmax = min(im.shape[1], int(xs.max()))
        ymin = max(0, int(ys.min()))
        ymax = min(im.shape[0], int(ys.max()))
        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]
        left = right = top = down = 0
        if xs.min() < 0:
            left = int(abs(xs.min()))
        if xs.max() > im.shape[1]:
            right = int(xs.max() - im.shape[1])
        if ys.min() < 0:
            top = int(abs(ys.min()))
        if ys.max() > im.shape[0]:
            down = int(ys.max() - im.shape[0])
        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
        im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])), cv2.INTER_CUBIC)
        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch

    def _feature_normalization(self, x):
        if hasattr(config, 'normalize_power') and config.normalize_power > 0:
            if config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** config.normalize_size * (x.shape[2]**config.normalize_dim) / (x**2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** config.normalize_size) * (x.shape[2]**config.normalize_dim) / ((np.abs(x) ** (1. / config.normalize_power)).sum(axis=(0, 1, 2)))

        if config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)

class CNNFeature(Feature):
    def _forward(self, x):
        pass

    def get_features(self, img, pos, sample_sz, scales):
        # 确保输入图像是 RGB (OpenCV读取默认是BGR)
        # 这里假设输入 img 已经是 RGB (由 tracker.py 控制)，如果不是可以在这里转换
        if img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
            
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
            
        patches = []
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            
            # --- PyTorch 预处理 ---
            # 1. 归一化到 [0, 1] 并转为 Float32
            patch = patch.astype(np.float32) / 255.0
            
            # 2. Numpy (H, W, C) -> Tensor (C, H, W)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)
            
            # 3. ImageNet 标准化 (Mean, Std)
            # PyTorch 的 Normalize 期望输入是 Tensor
            transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            normalized = transform(patch_tensor)
            
            # 4. 增加 Batch 维度: (C, H, W) -> (1, C, H, W)
            patches.append(normalized.unsqueeze(0))

        # 拼接 Batch: (N, C, H, W)
        batch_tensor = torch.cat(patches, dim=0)
        
        # 移至 GPU
        if hasattr(self, 'device'):
            batch_tensor = batch_tensor.to(self.device)

        # 前向传播 (返回的是 List[numpy array])
        # _forward 内部会负责 Tensor -> Numpy 以及维度的重排
        f1, f2 = self._forward(batch_tensor)
        
        f1 = self._feature_normalization(f1)
        f2 = self._feature_normalization(f2)
        return f1, f2

class MobileNetV3Feature(CNNFeature):
    def __init__(self, fname=None, compressed_dim=None):
        # 配置 Device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        # 加载 MobileNetV3-Small
        # print("Loading MobileNetV3-Small (PyTorch)...")
        self.net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # 定义特征提取层
        # MobileNetV3 Small 结构:
        # features[0]: Conv (stride 2) -> /2
        # features[1]: InvertedResidual (stride 2) -> /4  <-- Layer 1 (对应 ResNet pool1)
        # ...
        # features[9]: InvertedResidual (stride 2) -> /16 <-- Layer 2 (对应 ResNet stage4)
        
        self.features = self.net.features
        
        # 冻结参数 & 设为评估模式
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()
        self.features.to(self.device)

        # 配置参数 (用于 init_size 计算)
        self._compressed_dim = compressed_dim
        self._cell_size = [4, 8] # 对应两个特征层的下采样倍率
        self.penalty = [0., 0.]
        self.min_cell_size = np.min(self._cell_size)

    def init_size(self, img_sample_sz, cell_size=None):
        img_sample_sz = img_sample_sz.astype(np.int32)
        feat1_shape = np.ceil(img_sample_sz / 4)
        feat2_shape = np.ceil(img_sample_sz / 16)
        desired_sz = feat2_shape + 1 + feat2_shape % 2
        img_sample_sz = desired_sz * 16
        
        # MobileNetV3 Small 通道数:
        # Layer 1 (idx 1): 16 channels
        # Layer 2 (idx 8): 48 channels (MobileNetV3 Small 的 layer 8 输出通常是 48)
        # 注意：这里需要根据实际打印出的网络结构确认通道数，MobileNetV3-Small 默认此处是 48
        self.num_dim = [16, 24] 
        
        self.sample_sz = img_sample_sz
        self.data_sz = [np.ceil(img_sample_sz / 4),
                        np.ceil(img_sample_sz / 8)]
        return img_sample_sz

    def _forward(self, x):
        # x shape: (N, 3, H, W)
        with torch.no_grad():
            # Layer 1: Stride 4 (features 0-1)
            x = self.features[0](x)
            x = self.features[1](x)
            f1 = x # Save stride 4 feature

            # Layer 2: Stride 16 (features 2-9)
            for i in range(2, 4):
                x = self.features[i](x)
            f2 = x # Save stride 16 feature

        # ECO 算法要求输出格式为 (Height, Width, Channel, Batch)
        # PyTorch 输出格式为 (Batch, Channel, Height, Width)
        # 转换操作: permute(2, 3, 1, 0) -> (H, W, C, N)
        
        f1_np = f1.permute(2, 3, 1, 0).cpu().numpy()
        f2_np = f2.permute(2, 3, 1, 0).cpu().numpy()
        
        return [f1_np, f2_np]

class ResNet18HybridFeature(CNNFeature):
    def __init__(self, fname=None, compressed_dim=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        full_net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # --- 模块拆分 ---
        # 1. 浅层部分 (到 layer1 结束)
        # Input -> /4, 64ch
        self.stage1 = torch.nn.Sequential(
            full_net.conv1, full_net.bn1, full_net.relu, full_net.maxpool,
            full_net.layer1 
        )
        
        # 2. 深层部分 (layer2 + layer3)
        # layer2: /8, 128ch
        # layer3: 原本 /16, 我们魔改成 /8 (Dilated), 256ch
        self.stage2_3 = torch.nn.Sequential(
            full_net.layer2,
            full_net.layer3
        )
        
        # 将 layer3 改为空洞卷积 (Stride=1, Dilation=2)
        self._make_dilated(self.stage2_3[1])

        # 冻结参数
        for m in [self.stage1, self.stage2_3]:
            for param in m.parameters():
                param.requires_grad = False
            m.eval()
            m.to(self.device)

        # --- 关键配置 ---
        # 这里的 compressed_dim 对应 config.py 里的设置
        # 建议设为 [16, 64]
        self._compressed_dim = compressed_dim 
        
        # 对应两层的下采样倍率: [Stride 4, Stride 8]
        self._cell_size = [4, 8] 
        self.min_cell_size = np.min(self._cell_size)
        self.penalty = [0., 0.]

    def _make_dilated(self, layer_block):
        for module in layer_block.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.stride == (2, 2):
                    # 情况 A: 3x3 卷积 (主分支)
                    # 需要 Dilation=2 和 Padding=2 来保持分辨率并扩大感受野
                    if module.kernel_size == (3, 3):
                        module.stride = (1, 1)
                        module.dilation = (2, 2)
                        module.padding = (2, 2)
                    
                    # 情况 B: 1x1 卷积 (Shortcut 下采样层)
                    # 只需要改 Stride 为 1，千万不能加 Padding！
                    elif module.kernel_size == (1, 1):
                        module.stride = (1, 1)
                        # module.dilation 保持默认 (1,1) 即可
                        # module.padding 保持默认 (0,0) 即可

    def init_size(self, img_sample_sz, cell_size=None):
        img_sample_sz = img_sample_sz.astype(np.int32)
        
        # Layer 1: /4
        feat1_shape = np.ceil(img_sample_sz / 4)
        # Layer 3: /8
        feat2_shape = np.ceil(img_sample_sz / 8)
        
        desired_sz = feat2_shape + 1 + feat2_shape % 2
        img_sample_sz = desired_sz * 8
        
        # 原始通道数 (用于 PCA 初始化)
        # Layer 1: 64, Layer 3: 256
        self.num_dim = [64, 256] 
        
        self.sample_sz = img_sample_sz
        self.data_sz = [np.ceil(img_sample_sz / 4),
                        np.ceil(img_sample_sz / 8)]
        return img_sample_sz

    def _forward(self, x):
        with torch.no_grad():
            # 跑第一段
            x1 = self.stage1(x) # 得到 f1 (64ch, Stride 4)
            
            # 跑第二段 (接着 x1 跑)
            x2 = self.stage2_3(x1) # 得到 f2 (256ch, Stride 8)
            
        # 转换维度 (N, C, H, W) -> (H, W, C, N)
        f1_np = self.quantize_simulate(x1.permute(2, 3, 1, 0).cpu().numpy())
        f2_np = self.quantize_simulate(x2.permute(2, 3, 1, 0).cpu().numpy())
        
        return [f1_np, f2_np]

    def quantize_simulate(self,tensor_data, bits=8):
        # 简单的线性量化模拟
        # 1. 找最大最小值
        min_val = tensor_data.min()
        max_val = tensor_data.max()
        
        # 2. 计算 Scale
        # 2^8 - 1 = 255
        scale = (max_val - min_val) / 255.0
        
        # 3. 量化 (Float -> Int)
        # round() 是模拟量化噪声的关键
        q_data = np.round((tensor_data - min_val) / scale)
        
        # 4. 截断 (模拟溢出)
        q_data = np.clip(q_data, 0, 255)
        
        # 5. 反量化 (Int -> Float)
        # 让后续的 ECO 算法以为它是 float，但其实它只剩下了 8-bit 的精度
        dq_data = q_data * scale + min_val   
        
        return dq_data

# FHogFeature 和 TableFeature 是纯 Numpy/C 实现，保持不变即可，
# 但为了完整性，这里一并列出，确保 import 不出错
def fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = _gradient.gradMag(I.astype(np.float32), 0, True)
    H = _gradient.fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H

class FHogFeature(Feature):
    def __init__(self, fname, cell_size=6, compressed_dim=10, num_orients=9, clip=.2):
        self.fname = fname
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._soft_bin = -1
        self._bin_size = cell_size
        self._num_orients = num_orients
        self._clip = clip
        self.min_cell_size = self._cell_size
        self.num_dim = [3 * num_orients + 5 - 1]
        self.penalty = [0.]

    def get_features(self, img, pos, sample_sz, scales):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            M, O = _gradient.gradMag(patch.astype(np.float32), 0, True)
            H = _gradient.fhog(M, O, self._bin_size, self._num_orients, self._soft_bin, self._clip)
            H = H[:, :, :-1]
            feat.append(H)
        # 1. 归一化 (通常是硬件的高精度部分完成的)
        feat = self._feature_normalization(np.stack(feat, axis=3))
        # ==========================================================
        # 2. 模拟 INT8 量化 (Simulation of INT8 Storage)
        # ==========================================================
        # 注意：这里模拟的是存入样本库之前的数据截断
        
        # 方法 A: 动态量化 (利用当前帧最大值) - 精度较高，适合软件模拟
        # 方法 B: 静态量化 (利用理论最大值, FHOG通常clip在0.2) - 更像硬件定点数
        
        # 这里我们采用更稳健的动态量化方案：
        max_val = np.max(feat)
        
        if max_val > 1e-6: # 防止全0除零错误
            # 计算缩放因子：映射到 0-255
            scale_factor = 255.0 / max_val
            
            # 量化 (Float -> INT8)
            feat_int8 = np.round(feat * scale_factor)
            
            # 截断 (确保不溢出)
            feat_int8 = np.clip(feat_int8, 0, 255)
            
            # 反量化 (INT8 -> Float) 
            # 这一步是欺骗后续的 ECO 算法，让它以为还是浮点数，但实际上精度已经丢了
            feat = feat_int8 / scale_factor
        
        # ==========================================================

        return [feat]

class TableFeature(Feature):
    def __init__(self, fname, compressed_dim, table_name, use_for_color, cell_size=1):
        self.fname = fname
        self._table_name = table_name
        self._color = use_for_color
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._factor = 32
        self._den = 8
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._table = pickle.load(open(os.path.join(dir_path, "lookup_tables", self._table_name+".pkl"), "rb"))
        self.num_dim = [self._table.shape[1]]
        self.min_cell_size = self._cell_size
        self.penalty = [0.]
        self.sample_sz = None
        self.data_sz = None

    def integralVecImage(self, img):
        w, h, c = img.shape
        intImage = np.zeros((w+1, h+1, c), dtype=img.dtype)
        intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
        return intImage

    def average_feature_region(self, features, region_size):
        region_area = region_size ** 2
        if features.dtype == np.float32:
            maxval = 1.
        else:
            maxval = 255
        intImage = self.integralVecImage(features)
        i1 = np.arange(region_size, features.shape[0]+1, region_size).reshape(-1, 1)
        i2 = np.arange(region_size, features.shape[1]+1, region_size).reshape(1, -1)
        region_image = (intImage[i1, i2, :] - intImage[i1, i2-region_size,:] - intImage[i1-region_size, i2, :] + intImage[i1-region_size, i2-region_size, :])  / (region_area * maxval)
        return region_image

    def get_features(self, img, pos, sample_sz, scales):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            h, w, c = patch.shape
            if c == 3:
                RR = patch[:, :, 0].astype(np.int32)
                GG = patch[:, :, 1].astype(np.int32)
                BB = patch[:, :, 2].astype(np.int32)
                index = RR // self._den + (GG // self._den) * self._factor + (BB // self._den) * self._factor * self._factor
                features = self._table[index.flatten()].reshape((h, w, self._table.shape[1]))
            else:
                features = self._table[patch.flatten()].reshape((h, w, self._table.shape[1]))
            if self._cell_size > 1:
                features = self.average_feature_region(features, self._cell_size)
            feat.append(features)
        feat = self._feature_normalization(np.stack(feat, axis=3))
        return [feat]