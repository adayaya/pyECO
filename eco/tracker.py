import numpy as np
import cv2
import scipy
import time


from scipy import signal
# from numpy.fft import fftshift

from .config import config
from .features import FHogFeature, TableFeature, mround, ResNet50Feature, VGG16Feature
from .fourier_tools import cfft2, interpolate_dft, shift_sample, full_fourier_coeff,\
        cubic_spline_fourier, compact_fourier_coeff, ifft2, fft2, sample_fs
from .optimize_score import optimize_score
from .sample_space_model import GMM
from .train import train_joint, train_filter
from .scale_filter import ScaleFilter

if config.use_gpu:
    import cupy as cp


class ECOTracker:
    def __init__(self, is_color):
        self._is_color = is_color
        self._frame_num = 0
        self._frames_since_last_train = 0
        if config.use_gpu:
            cp.cuda.Device(config.gpu_id).use()
    """
    1. 为什么要用这个？(核心原理)
    在 ECO 这类算法中,我们需要把图像转换到频域(Fourier Domain)进行计算。 傅里叶变换(FFT)有一个前提假设:信号是无限周期循环的。
    但是,真实的图像块并不是周期的(图片左边缘和右边缘通常长得不一样)。如果我们直接对图片做 FFT,边缘的不连续会产生大量的高频噪声(称为“频谱泄露”或边界效应),严重干扰跟踪结果。
    余弦窗的作用就像一个“聚光灯”:
    它中间的值是 1(保留目标中心的信息)。
    它边缘的值平滑地降为 0。
    结果:当我们用这个窗乘以图像块时,图像边缘会被自然地“抹黑”变成 0。这样,图片的左边和右边都是 0,就完美衔接了,FFT 就不会报错乱了。
    """
    def _cosine_window(self, size): # 生成二维汉宁窗
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]+2))[:, np.newaxis].dot(np.hanning(int(size[1]+2))[np.newaxis, :]) # 用1维汉宁窗生成2维汉宁窗
        cos_window = cos_window[1:-1, 1:-1][:, :, np.newaxis, np.newaxis].astype(np.float32) # 去除边缘
        if config.use_gpu:
            cos_window = cp.asarray(cos_window) # 搬到显存
        return cos_window

    """
    在 ECO 这类相关滤波跟踪算法中，为了实现亚像素级别的定位精度（不仅仅精确到整数坐标，而是精确到小数坐标，如 (10.5, 20.3)），算法需要在频域对特征图进行插值。
    这段代码并没有直接做插值，而是准备好了用来做插值的"工具"（滤波器）。
    """
    def _get_interp_fourier(self, sz): # 在频域内计算插值函数的核函数
        """
            compute the fourier series of the interpolation function. 计算插值核的傅里叶系数
        """
        f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis] / sz[0] # 生成频率变量 （高度）方向
        interp1_fs = np.real(cubic_spline_fourier(f1, config.interp_bicubic_a) / sz[0]) # 计算了三次样条函数在频域的解析式，在时域做双三次插值，等于在频域乘以这个函数的系数
        f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :] / sz[1] # 生成频率变量 （宽度）方向
        interp2_fs = np.real(cubic_spline_fourier(f2, config.interp_bicubic_a) / sz[1])
        if config.interp_centering: # 中心对齐，修正特征网格与图像像素网格之间的半个像素偏差
            f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis]
            interp1_fs = interp1_fs * np.exp(-1j*np.pi / sz[0] * f1) # 一个复指数相位移动=时域的平移，修正半个像素的偏差
            f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :]
            interp2_fs = interp2_fs * np.exp(-1j*np.pi / sz[1] * f2)

        if config.interp_windowing: # 频域加窗，减少插值伪影
            win1 = np.hanning(sz[0]+2)[:, np.newaxis]
            win2 = np.hanning(sz[1]+2)[np.newaxis, :]
            interp1_fs = interp1_fs * win1[1:-1]
            interp2_fs = interp2_fs * win2[1:-1]
        if not config.use_gpu:
            return (interp1_fs[:, :, np.newaxis, np.newaxis],
                    interp2_fs[:, :, np.newaxis, np.newaxis])
        else:
            return (cp.asarray(interp1_fs[:, :, np.newaxis, np.newaxis]),
                    cp.asarray(interp2_fs[:, :, np.newaxis, np.newaxis]))

    """
    空间正则化滤波器的作用是对滤波器的权重进行空间上的约束，防止滤波器在某些区域过度响应，从而提高跟踪的鲁棒性和准确性。
    具体来说，空间正则化滤波器通过在滤波器的权重上施加一个空间变化的惩罚项，使得滤波器在目标区域内的响应较强，而在背景区域的响应较弱。
    这样可以有效地抑制背景干扰，提高跟踪器对目标的区分能力。
    这使得 ECO 即使在搜索范围很大的情况下，也不会轻易被背景里的树木、行人等干扰物带偏。
    这个正则化矩阵只有在以下两种情况同时发生时，才需要重新调用该函数：滤波器更新时，且目标的尺寸变化时。
    大部分帧（检测帧）完全不需要计算它；即使在训练帧，如果目标大小没变，也可以直接复用上一帧的矩阵。
    用于加速滤波器更新的速度。
    """
    def _get_reg_filter(self, sz, target_sz, reg_window_edge): #对正则化矩阵稀疏化，人为设定的惩罚规则，告诉滤波器，哪些区域可以重点关注，哪些区域需要忽略
        """
            compute the spatial regularization function and drive the
            corresponding filter operation used for optimization
        """
        if config.use_reg_window:
            # normalization factor
            reg_scale = 0.5 * target_sz

            # construct grid
            wrg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wcg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wrs, wcs = np.meshgrid(wrg, wcg)

            # construct the regularization window 构造了数学上的抛物面，形状像一个碗底：中心最低，边缘最高
            # 根据距离中心的远近，计算每个位置的正则化权重
            reg_window = (reg_window_edge - config.reg_window_min) * (np.abs(wrs/reg_scale[0])**config.reg_window_power + \
                            np.abs(wcs/reg_scale[1])**config.reg_window_power) + config.reg_window_min
            # 在早期的算法（如 SRDCF）中，为了应用这个“碗”状的惩罚，需要进行复杂的数学运算（时域乘法 = 频域卷积）。频域卷积的计算量是巨大的。
            # ECO 的作者想出了一个绝招：既然这个“碗”在频域里很多系数都很小，不如直接把它们扔掉（置为0）？
            # compute the DFT and enforce sparsity
            reg_window_dft = fft2(reg_window) / np.prod(sz) # 计算正则化窗口的傅里叶变换，转到频域
            # 强行稀疏化，把原本很小的只直接变成0，这一步极大地减少了非零元素的数量。原本需要做全尺寸卷积，现在只需要对剩下那一点点非零值做计算，速度提升了几个数量级。
            reg_window_dft[np.abs(reg_window_dft) < config.reg_sparsity_threshold* np.max(np.abs(reg_window_dft.flatten()))] = 0

            # do the inverse transform, correct window minimum 
            # 修正直流分量，确保时域的正则化窗口的最小值是 config.reg_window_min
            reg_window_sparse = np.real(ifft2(reg_window_dft))
            reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(sz) * np.min(reg_window_sparse.flatten()) + config.reg_window_min
            reg_window_dft = np.fft.fftshift(reg_window_dft).astype(np.complex64)

            # find the regularization filter by removing the zeros
            # 压缩矩阵，只保留那些非零行和列，后续计算时，矩阵乘法的规模就变得非常小了
            row_idx = np.logical_not(np.all(reg_window_dft==0, axis=1))
            col_idx = np.logical_not(np.all(reg_window_dft==0, axis=0))
            mask = np.outer(row_idx, col_idx)
            reg_filter = np.real(reg_window_dft[mask]).reshape(np.sum(row_idx), -1)
        else:
            # else use a scaled identity matrix
            reg_filter = config.reg_window_min
        if not config.use_gpu:
            return reg_filter.T
        else:
            return cp.asarray(reg_filter.T)

    # 初始化投影矩阵，用于将高维特征降维到较低维度，以减少计算复杂度和内存占用。 分解卷积的实现起点。
    """
    输入: 你提取的特征(特别是深度学习特征 ResNet/VGG)维度通常极高(几百甚至上千个通道)。
    问题: 如果直接在这么高维的特征上做相关滤波(卷积)，计算量会大到爆炸，根本达不到实时。
    解决: ECO 引入了一个投影矩阵 P。
    """
    def _init_proj_matrix(self, init_sample, compressed_dim, proj_method):
        """
            init the projection matrix
        """
        if config.use_gpu:
            xp = cp.get_array_module(init_sample[0])
        else:
            xp = np
        # 变形，把（H，W，C）变成（Pixels，C）
        x = [xp.reshape(x, (-1, x.shape[2])) for x in init_sample]
        # 去均值， 做PCA标准步骤
        x = [z - z.mean(0) for z in x]
        proj_matrix_ = []
        # PCA主成分分析
        if config.proj_init_method == 'pca':
            for x_, compressed_dim_  in zip(x, compressed_dim):
                # 计算协方差矩阵的近似（X^T * X）,得到（C,C）的矩阵，代表不同通道间的相关性，利用SVD分解找到特征变化最剧烈的那些方向。
                proj_matrix, _, _ = xp.linalg.svd(x_.T.dot(x_))
                # 提取前K个主成分
                proj_matrix = proj_matrix[:, :compressed_dim_]
                proj_matrix_.append(proj_matrix)
        elif config.proj_init_method == 'rand_uni':
            for x_, compressed_dim_ in zip(x, compressed_dim):
                proj_matrix = xp.random.uniform(size=(x_.shape[1], compressed_dim_))
                proj_matrix /= xp.sqrt(xp.sum(proj_matrix**2, axis=0, keepdims=True))
                proj_matrix_.append(proj_matrix)
        return proj_matrix_

    def _proj_sample(self, x, P): # 投影样本，降维
        if config.use_gpu:
            xp = cp.get_array_module(x[0])
        else:
            xp = np
        return [xp.matmul(P_.T, x_) for x_, P_ in zip(x, P)]

    # 初始化跟踪器，在视频第一帧被调用
    """
    核心任务：根据第一帧的目标位置和大小，初始化跟踪器的各种参数和模型。
    主要步骤包括：
    1. 计算搜索区域和尺度标准化。（确定看哪里，以及看多大）
    2. 提取特征并初始化特征提取器。
    3. 构建高斯标签函数和余弦窗。
    4. 构建空间正则化滤波器。
    5. 初始化样本空间模型（GMM）。
    6. 提取初始样本并训练初始滤波器。
    这些步骤为后续的目标跟踪奠定了基础，使得跟踪器能够在后续帧中准确地定位和跟踪目标。
    """
    def init(self, frame, bbox, total_frame=np.inf):
        """
            frame -- image
            bbox -- need xmin, ymin, width, height
        """
        self._pos = np.array([bbox[1]+(bbox[3]-1)/2., bbox[0]+(bbox[2]-1)/2.], dtype=np.float32) # 目标的中心坐标
        self._target_sz = np.array([bbox[3], bbox[2]]) # 目标的宽和高
        self._num_samples = min(config.num_samples, total_frame)
        xp = cp if config.use_gpu else np

        # calculate search area and initial scale factor
        search_area = np.prod(self._target_sz * config.search_area_scale) # 不仅会看目标本身，还会看周围的背景，通常是目标大小的倍数
        if search_area > config.max_image_sample_size: # 限制搜索区域的最大尺寸，防止计算量过大
            self._current_scale_factor = np.sqrt(search_area / config.max_image_sample_size)
        elif search_area < config.min_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.min_image_sample_size)
        else:
            self._current_scale_factor = 1.

        # target size at the initial scale，成比例缩放目标的大小
        self._base_target_sz = self._target_sz / self._current_scale_factor

        # target size, taking padding into account
        if config.search_area_shape == 'proportional':
            self._img_sample_sz = np.floor(self._base_target_sz * config.search_area_scale)
        elif config.search_area_shape == 'square': # 让目标形状变为正方形，方便后续计算
            self._img_sample_sz = np.ones((2), dtype=np.float32) * np.sqrt(np.prod(self._base_target_sz * config.search_area_scale))
        else:
            raise("unimplemented")
        # 特征提取器初始化
        features = [feature for feature in config.features
                if ("use_for_color" in feature and feature["use_for_color"] == self._is_color) or
                    "use_for_color" not in feature]

        self._features = []
        cnn_feature_idx = -1
        for idx, feature in enumerate(features):
            if feature['fname'] == 'cn' or feature['fname'] == 'ic':
                self._features.append(TableFeature(**feature)) # LUT， 提取颜色特征
            elif feature['fname'] == 'fhog': # 提取形状边缘信息
                self._features.append(FHogFeature(**feature))
            elif feature['fname'].startswith('cnn'): # 提取深层语义信息
                cnn_feature_idx = idx
                netname = feature['fname'].split('-')[1]
                if netname == 'resnet50':
                    self._features.append(ResNet50Feature(**feature))
                elif netname == 'vgg16':
                    self._features.append(VGG16Feature(**feature))
            else:
                raise("unimplemented features")
        self._features = sorted(self._features, key=lambda x:x.min_cell_size) # Resnet：cell sz[4,16] FHOG: cell sz[6]

        # calculate image sample size
        # 确定最终送入网络的图像尺寸，并计算对应的输出特征形状
        if cnn_feature_idx >= 0:# 获取深度特征对应的图像尺寸
            self._img_sample_sz = self._features[cnn_feature_idx].init_size(self._img_sample_sz)
        else:
            cell_size = [x.min_cell_size for x in self._features]
            self._img_sample_sz = self._features[0].init_size(self._img_sample_sz, cell_size)

        for idx, feature in enumerate(self._features):
            if idx != cnn_feature_idx:
                feature.init_size(self._img_sample_sz) # 其他特征按CNN图像尺寸准备

        if config.use_projection_matrix: # 样本维度：给滤波器用的，是滤波器实际处理的通道数，决定了滤波器的厚度。
            sample_dim = [ x for feature in self._features for x in feature._compressed_dim ]
        else:
            sample_dim = [ x for feature in self._features for x in feature.num_dim ]

        # 原始特征维度：给投影矩阵用的，是投影矩阵输入的通道数，刚从图片提取出来的原始通道数。
        feature_dim = [ x for feature in self._features for x in feature.num_dim ] 
        # 特征图尺寸：每个特征层输出的空间尺寸，给频域计算用。 输入一张图片，经过HOG/CNN特征提取器后，得到的特征图的高和宽。
        feature_sz = np.array([x for feature in self._features for x in feature.data_sz ], dtype=np.int32)

        # number of fourier coefficients to save for each filter layer, this will be an odd number
        filter_sz = feature_sz + (feature_sz + 1) % 2 # 确保滤波器尺寸是奇数，确保滤波器有一个精确的中心像素，防止在进行傅里叶变换和高斯标签生成时出现坐标偏移或对齐错误。

        # the size of the label function DFT. equal to the maximum filter size
        self._k1 = np.argmax(filter_sz, axis=0)[0] # 获取分辨率最高的特征大小。低于这个分辨率的就插值
        self._output_sz = filter_sz[self._k1]
        
        self._num_feature_blocks = len(feature_dim)

        # get the remaining block indices
        self._block_inds = list(range(self._num_feature_blocks))
        self._block_inds.remove(self._k1)

        # how much each feature block has to be padded to the obtain output_sz
        # 需要padding的尺寸，达到统一的输出分辨率
        self._pad_sz = [((self._output_sz - filter_sz_) / 2).astype(np.int32) for filter_sz_ in filter_sz]

        # compute the fourier series indices and their transposes
        # 生成频域的kx和ky坐标，在频域中，不同的位置代表不同的频率
        self._ky = [np.arange(-np.ceil(sz[0]-1)/2, np.floor((sz[0]-1)/2)+1, dtype=np.float32)
                        for sz in filter_sz]
        self._kx = [np.arange(-np.ceil(sz[1]-1)/2, 1, dtype=np.float32)
                        for sz in filter_sz]

        # construct the gaussian label function using poisson formula 生成高斯标签（标准答案），利用泊松求和公式，频域定义的、中心值为1的高斯波峰。
        sig_y = np.sqrt(np.prod(np.floor(self._base_target_sz))) * config.output_sigma_factor * (self._output_sz / self._img_sample_sz)
        yf_y = [np.sqrt(2 * np.pi) * sig_y[0] / self._output_sz[0] * np.exp(-2 * (np.pi * sig_y[0] * ky_ / self._output_sz[0])**2)
                    for ky_ in self._ky]
        yf_x = [np.sqrt(2 * np.pi) * sig_y[1] / self._output_sz[1] * np.exp(-2 * (np.pi * sig_y[1] * kx_ / self._output_sz[1])**2)
                    for kx_ in self._kx]

        self._yf = [yf_y_.reshape(-1, 1) * yf_x_ for yf_y_, yf_x_ in zip(yf_y, yf_x)]
        if config.use_gpu:
            self._yf = [cp.asarray(yf) for yf in self._yf]
            self._ky = [cp.asarray(ky) for ky in self._ky]
            self._kx = [cp.asarray(kx) for kx in self._kx]

        # construct cosine window 给每个分辨率特征生成二维汉宁窗，加窗，消除边界效应
        self._cos_window = [self._cosine_window(feature_sz_) for feature_sz_ in feature_sz]

        # compute fourier series of interpolation function
        self._interp1_fs = []
        self._interp2_fs = []
        for sz in filter_sz: #对于每个分辨率，生成高、宽两个方向的插值滤波器
            interp1_fs, interp2_fs = self._get_interp_fourier(sz)
            self._interp1_fs.append(interp1_fs)
            self._interp2_fs.append(interp2_fs)

        # get the reg_window_edge parameter 
        reg_window_edge = []
        for feature in self._features:# 正则化窗口
            if hasattr(feature, 'reg_window_edge'):
                reg_window_edge.append(feature.reg_window_edge)
            else:
                reg_window_edge += [config.reg_window_edge for _ in range(len(feature.num_dim))]

        # construct spatial regularization filter
        # 输入图像的转成正方形的size， 缩放后的目标size， 正则化窗口生成需要的参数
        self._reg_filter = [self._get_reg_filter(self._img_sample_sz, self._base_target_sz, reg_window_edge_)
                                for reg_window_edge_ in reg_window_edge]

        # compute the energy of the filter (used for preconditioner)
        if not config.use_gpu: # 计算正则化滤波器的能量，用于预处理，在共轭梯度法中加速收敛，少迭代几次就能算出结果
            self._reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self._reg_filter]
        else:
            self._reg_energy = [cp.real(cp.vdot(reg_filter.flatten(), reg_filter.flatten()))
                            for reg_filter in self._reg_filter]

        if config.use_scale_filter: # 单独算尺度,先用平移滤波器算x，y，再用专门的尺度滤波器算w，h
            self._scale_filter = ScaleFilter(self._target_sz)
            self._num_scales = self._scale_filter.num_scales
            self._scale_step = self._scale_filter.scale_step
            self._scale_factor = self._scale_filter.scale_factors
        else: # 一起算，生成五个不同大小的候选图（批处理，时间上应该小于5倍），看谁得分高，基于上一帧的目标size计算得出，用于多尺度计算目标，动态变化框的大小。每一帧都能变化，因此时间上累积可以实现更广的放缩。
            # use the translation filter to estimate the scale 
            self._num_scales = config.number_of_scales
            self._scale_step = config.scale_step
            scale_exp = np.arange(-np.floor((self._num_scales-1)/2), np.ceil((self._num_scales-1)/2)+1)
            self._scale_factor = self._scale_step**scale_exp
            print(self._scale_factor)

        if self._num_scales > 0:
            # force reasonable scale changes
            self._min_scale_factor = self._scale_step ** np.ceil(np.log(np.max(5 / self._img_sample_sz)) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(frame.shape[:2] / self._base_target_sz)) / np.log(self._scale_step))

        # set conjugate gradient options
        # 配置共轭梯度法（CG）来训练滤波器
        init_CG_opts = {'CG_use_FR': True, # 初始配置，用于第一帧，要求更高，迭代次数更多，为了让模型在起跑线上足够精准。
                        'tol': 1e-6,
                        'CG_standard_alpha': True
                       }
        self._CG_opts = {'CG_use_FR': config.CG_use_FR, # 日常配置，用于后续帧更新
                         'tol': 1e-6,
                         'CG_standard_alpha': config.CG_standard_alpha
                        }
        # 控制CG算法在重启时保留多少旧的搜索方向，如果学习率高就忘掉旧的，如果低，就用旧的加速搜索。
        if config.CG_forgetting_rate == np.inf or config.learning_rate >= 1:
            self._CG_opts['init_forget_factor'] = 0.
        else:
            self._CG_opts['init_forget_factor'] = (1 - config.learning_rate) ** config.CG_forgetting_rate

        # init ana allocate
        self._gmm = GMM(self._num_samples) # (滤波器高、滤波器宽、通道数、样本数)，存储过去几十帧里提取出的、经过压缩的频域特征。
        self._samplesf = [[]] * self._num_feature_blocks # 3个特征块 CNN*2 + HOG*1 [[61,31,10,50]，[61,31,16,50],[15,8,64,50]]
        print(self._num_feature_blocks)
        for i in range(self._num_feature_blocks):
            if not config.use_gpu:
                self._samplesf[i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                    sample_dim[i], config.num_samples), dtype=np.complex64)
            else:
                self._samplesf[i] = cp.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                    sample_dim[i], config.num_samples), dtype=cp.complex64)
        # allocate
        self._num_training_samples = 0

        # extract sample and init projection matrix
        sample_pos = mround(self._pos) # 目标中心取整，因为提取特征只能在整数像素上切图。
        sample_scale = self._current_scale_factor # 目标缩放倍数
        xl = [x for feature in self._features # 根据搜索区域获取特征
                for x in feature.get_features(frame, sample_pos, self._img_sample_sz, self._current_scale_factor) ]  # get features

        if config.use_gpu:
            xl = [cp.asarray(x) for x in xl]

        xlw = [x * y for x, y in zip(xl, self._cos_window)]                                                          # do windowing 提取到的特征加窗
        xlf = [cfft2(x) for x in xlw]                                                                                # fourier series 转为频域
        xlf = interpolate_dft(xlf, self._interp1_fs, self._interp2_fs)                                               # interpolate features, 频域插值
        xlf = compact_fourier_coeff(xlf)                                                                             # new sample to be added， 压缩频谱，频域结果对称，可以只保留一半系数
        # 亚像素校正： 由于中心位置强制取整，因此为了放置偏差，需要乘以相位因子，修正偏差，保证特征准确对着目标真实中心。
        shift_sample_ = 2 * np.pi * (self._pos - sample_pos) / (sample_scale * self._img_sample_sz)
        xlf = shift_sample(xlf, shift_sample_, self._kx, self._ky)
        # 初始化投影矩阵
        self._proj_matrix = self._init_proj_matrix(xl, sample_dim, config.proj_init_method)
        xlf_proj = self._proj_sample(xlf, self._proj_matrix)
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
            self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples) # 更新GMM
        self._num_training_samples += 1 # 样本数+1，GMM容量-1

        if config.update_projection_matrix:#降维后的频域特征存入样本库
            for i in range(self._num_feature_blocks):# 第一帧，这里降维矩阵也训练，使得提取最能区分目标和背景的特征，因为背景在变，其次PCA只是提取主要特征，保留最大信息，但训练能丢弃那些虽然明显但无用的背景特征，放大能锁定目标的特征。
                self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i] 

        # train_tracker 训练
        self._sample_energy = [xp.real(x * xp.conj(x)) for x in xlf_proj]

        # init conjugate gradient param
        self._CG_state = None
        if config.update_projection_matrix:
            init_CG_opts['maxit'] = np.ceil(config.init_CG_iter / config.init_GN_iter) #外层循环：高斯牛顿法迭代次数；内层循环：共轭梯度法迭代次数
            self._hf = [[[]] * self._num_feature_blocks for _ in range(2)] # 初始化滤波器，全为0
            feature_dim_sum = float(np.sum(feature_dim))
            proj_energy = [2 * xp.sum(xp.abs(yf_.flatten())**2) / feature_dim_sum * xp.ones_like(P) # 计算投影矩阵能量，归一化梯度
                    for P, yf_ in zip(self._proj_matrix, self._yf)]
        else:
            self._CG_opts['maxit'] = config.init_CG_iter
            self._hf = [[[]] * self._num_feature_blocks]

        # init the filter with zeros
        for i in range(self._num_feature_blocks):
            self._hf[0][i] = xp.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                int(sample_dim[i]), 1), dtype=xp.complex64)

        if config.update_projection_matrix:
            # init Gauss-Newton optimization of the filter and projection matrix
            self._hf, self._proj_matrix = train_joint( # 联合训练，训练降维矩阵和滤波器
                                                  self._hf,
                                                  self._proj_matrix,
                                                  xlf,
                                                  self._yf,
                                                  self._reg_filter,
                                                  self._sample_energy,
                                                  self._reg_energy,
                                                  proj_energy,
                                                  init_CG_opts)
            # re-project and insert training sample
            xlf_proj = self._proj_sample(xlf, self._proj_matrix) #精调后的投影矩阵，非PCA
            # self._sample_energy = [np.real(x * np.conj(x)) for x in xlf_proj]
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, 0:1] = xlf_proj[i] # 用新投影矩阵更新样本库的第一个样本

            # udpate the gram matrix since the sample has changed
            if config.distance_matrix_update_type == 'exact': #因为重投影了，所以需要修改能量值。 精确模式。
                # find the norm of the reprojected sample
                new_train_sample_norm = 0.
                for i in range(self._num_feature_blocks):
                    new_train_sample_norm += 2 * xp.real(xp.vdot(xlf_proj[i].flatten(), xlf_proj[i].flatten()))
                self._gmm._gram_matrix[0, 0] = new_train_sample_norm
        self._hf_full = full_fourier_coeff(self._hf) # 保存滤波器全谱，这里用于可视化？或空间域转换。

        if config.use_scale_filter and self._num_scales > 0:# ECO-HC的尺度识别
            self._scale_filter.update(frame, self._pos, self._base_target_sz, self._current_scale_factor)
        self._frame_num += 1

    """
    跟踪器的核心函数，在每一帧图像上被调用，用于定位目标并更新模型。
    负责在每一帧新图像中找到目标，并根据目标的新样子学习更新跟踪模型。
    主要步骤包括：
    1. 目标定位：通过多次迭代细化目标位置，提取特征，计算得分，并使用牛顿法优化位置。
    2. 可视化得分（可选）：如果启用可视化选项，显示当前得分图。
    3. 模型更新: 使用检测到的新位置提取样本, 并通过GMM更新样本空间模型。
    4. 滤波器训练：使用共轭梯度法训练空间正则化滤波器，更新滤波器权重。
    该函数结合了目标检测和模型更新两个关键步骤，确保跟踪器能够在动态场景中持续准确地跟踪目标。
    """
    def update(self, frame, train=True, vis=False):
        # target localization step
        xp = cp if config.use_gpu else np
        pos = self._pos
        old_pos = np.zeros((2))
        for _ in range(config.refinement_iterations):
            # if np.any(old_pos != pos):
            if not np.allclose(old_pos, pos):
                old_pos = pos.copy() # 记录上一次的位置
                # extract fatures at multiple resolutions
                sample_pos = mround(pos) # 采样中心，取整
                sample_scale = self._current_scale_factor * self._scale_factor # 多尺度采样: 目标放大倍数+搜索框放大倍数
                # 提取多尺度特征
                xt = [x for feature in self._features
                        for x in feature.get_features(frame, sample_pos, self._img_sample_sz, sample_scale) ]  # get features
                if config.use_gpu:
                    xt = [cp.asarray(x) for x in xt]
                # 先投影
                xt_proj = self._proj_sample(xt, self._proj_matrix)                                             # project sample
                # 加窗
                xt_proj = [feat_map_ * cos_window_
                        for feat_map_, cos_window_ in zip(xt_proj, self._cos_window)]                          # do windowing
                # FFT变换，计算傅里叶系数
                xtf_proj = [cfft2(x) for x in xt_proj]                                                         # compute the fourier series
                # 频域插值，并非放大频域图size，而是将离散的特征点映射到连续域系数上，仍然保留了低分辨率属性
                xtf_proj = interpolate_dft(xtf_proj, self._interp1_fs, self._interp2_fs)                       # interpolate features to continuous domain

                # compute convolution for each feature block in the fourier domain, then sum over blocks
                # 计算响应图，用训练好的滤波器与当前帧做卷积（频域点乘），再将各个特征块的响应图相加，得到最终的响应图。
                scores_fs_feat = [[]] * self._num_feature_blocks
                scores_fs_feat[self._k1] = xp.sum(self._hf_full[self._k1] * xtf_proj[self._k1], 2)
                scores_fs = scores_fs_feat[self._k1]

                # scores_fs_sum shape: height x width x num_scale
                for i in self._block_inds:
                    scores_fs_feat[i] = xp.sum(self._hf_full[i] * xtf_proj[i], 2)
                    # 填充，将低分辨率的CNN低频信息加到高分辨率的响应图上的中间部分。
                    scores_fs[self._pad_sz[i][0]:self._output_sz[0]-self._pad_sz[i][0],
                              self._pad_sz[i][1]:self._output_sz[0]-self._pad_sz[i][1]] += scores_fs_feat[i]

                # optimize the continuous score function with newton's method.
                # 牛顿法找到最优位置，使用牛顿迭代法在响应图峰值附近算斜率，找到亚像素级的精确峰值位置，同时确定哪个尺度得分最高。
                trans_row, trans_col, scale_idx = optimize_score(scores_fs, config.newton_iterations)

                # show score
                if vis:
                    if config.use_gpu:
                       xp = cp
                    self.score = xp.fft.fftshift(sample_fs(scores_fs[:,:,scale_idx],
                            tuple((10*self._output_sz).astype(np.uint32))))
                    if config.use_gpu:
                       self.score = cp.asnumpy(self.score)
                    self.crop_size = self._img_sample_sz * self._current_scale_factor

                # compute the translation vector in pixel-coordinates and round to the cloest integer pixel
                #计算位移和更新位置：将频域的位移量换算回图像的像素坐标
                translation_vec = np.array([trans_row, trans_col]) * (self._img_sample_sz / self._output_sz) * \
                                    self._current_scale_factor * self._scale_factor[scale_idx]
                # 新尺度缩放因子
                scale_change_factor = self._scale_factor[scale_idx]

                # udpate position 更新目标位置
                pos = sample_pos + translation_vec

                if config.clamp_position:
                    pos = np.maximum(np.array(0, 0), np.minimum(np.array(frame.shape[:2]), pos))

                # do scale tracking with scale filter
                # 尺度跟踪
                if self._num_scales > 0 and config.use_scale_filter:
                    scale_change_factor = self._scale_filter.track(frame, pos, self._base_target_sz,
                           self._current_scale_factor)

                # udpate the scale
                self._current_scale_factor *= scale_change_factor

                # adjust to make sure we are not to large or to small
                # 截断，防止跟踪框变得太小（消失）或者变得太大（撑爆屏幕）。
                if self._current_scale_factor < self._min_scale_factor:
                    self._current_scale_factor = self._min_scale_factor
                elif self._current_scale_factor > self._max_scale_factor:
                    self._current_scale_factor = self._max_scale_factor

        # model udpate step
        # 模型更新：把目标新样子存入GMM，并训练大脑（滤波器）
        if config.learning_rate > 0:
            # use the sample that was used for detection
            # 只取出最佳尺度
            sample_scale = sample_scale[scale_idx]
            xlf_proj = [xf[:, :(xf.shape[1]+1)//2, :, scale_idx:scale_idx+1] for xf in xtf_proj]

            # shift the sample so that the target is centered
            # 因为检测到的位置可能偏离中心，需要移回中心，让其对齐高斯标签的峰值。
            shift_sample_ = 2 * np.pi * (pos - sample_pos) / (sample_scale * self._img_sample_sz)
            xlf_proj = shift_sample(xlf_proj, shift_sample_, self._kx, self._ky)

        # update the samplesf to include the new sample. The distance matrix, kernel matrix and prior weight are also updated
        """
        GMM样本库更新，判断新样本与库中样本的相似度，
        如果相似度高于某个阈值，则将新样本与最相似样本合并（加权平均），
        否则将新样本作为一个新的样本插入库中。
        这样可以保持样本库的多样性，同时控制样本数量，防止过拟合。
        """
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
                self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)

        if self._num_training_samples < self._num_samples:
            self._num_training_samples += 1
        
        # 根据GMM的决策，将新样本合并或插入样本库
        if config.learning_rate > 0:
            for i in range(self._num_feature_blocks):
                if merged_sample_id >= 0:
                    self._samplesf[i][:, :, :, merged_sample_id:merged_sample_id+1] = merged_sample[i]
                if new_sample_id >= 0:
                    self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # training filter deep模式初始时只训练一帧，后续每隔几帧训练一次；HC模式初始10帧都参与训练。
        if self._frame_num < config.skip_after_frame or \
                self._frames_since_last_train >= config.train_gap:
            # print("Train filter: ", self._frame_num)
            # 根据最新的样本库重新训练滤波器
            new_sample_energy = [xp.real(xlf * xp.conj(xlf)) for xlf in xlf_proj] # 计算新样本能量：幅值的平方，指数移动平均。
            self._CG_opts['maxit'] = config.CG_iter # 更新迭代次数
            self._sample_energy = [(1 - config.learning_rate)*se + config.learning_rate*nse
                                for se, nse in zip(self._sample_energy, new_sample_energy)] # 能量分布，告诉优化器哪里陡峭哪里平缓

            # do conjugate gradient optimization of the filter
            self._hf, self._CG_state = train_filter( # 只训练滤波器，不训练投影矩阵
                                                 self._hf,
                                                 self._samplesf,
                                                 self._yf,
                                                 self._reg_filter,
                                                 self._gmm.prior_weights,
                                                 self._sample_energy,
                                                 self._reg_energy,
                                                 self._CG_opts,
                                                 self._CG_state)
            # reconstruct the ful fourier series
            self._hf_full = full_fourier_coeff(self._hf) # 新起点
            self._frames_since_last_train = 0
        else:
            self._frames_since_last_train += 1
        if config.use_scale_filter:
            self._scale_filter.update(frame, pos, self._base_target_sz, self._current_scale_factor)

        # udpate the target size 
        # 准备下一帧图像，更新目标大小
        self._target_sz = self._base_target_sz * self._current_scale_factor

        # save position and calculate fps
        # 更新目标大小和位置
        bbox = (pos[1] - self._target_sz[1]/2, # xmin
                pos[0] - self._target_sz[0]/2, # ymin
                pos[1] + self._target_sz[1]/2, # xmax
                pos[0] + self._target_sz[0]/2) # ymax
        self._pos = pos
        self._frame_num += 1
        return bbox
