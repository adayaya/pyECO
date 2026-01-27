class OTBDeepConfig:
    fhog_params = {'fname': 'fhog', #补充边缘和纹理细节
                   'num_orients': 9,
                   'cell_size': 4,
                   'compressed_dim': 10,
                   # 'nDim': 9 * 3 + 5 -1
                   }

    # cn_params = {"fname": 'cn', # CNN已经有极强的颜色和光照不变性信息，不需要额外的这些简单信息
    #              "table_name": "CNnorm",
    #              "use_for_color": True,
    #              "cell_size": 4,
    #              "compressed_dim": 3,
    #              # "nDim": 10
    #              }

    # ic_params = {'fname': 'ic',
    #              "table_name": "intensityChannelNorm6",
    #              "use_for_color": False,
    #              "cell_size": 4,
    #              "compressed_dim": 3,
    #              # "nDim": 10
    #              }

    cnn_params = {'fname': "cnn-resnet50",
                  'compressed_dim': [16, 64]
                  }
    # cnn_params = {'fname': "cnn-vgg16",
    #               'compressed_dim': [16, 64]
    #               }
    features = [fhog_params, cnn_params]

    # feature parameters 特征承诺书
    normalize_power = 2
    normalize_size = True
    normalize_dim = True
    square_root_normalization = False

    # image sample parameters 图像采样参数
    search_area_shape = 'square' # 形状，正方形
    search_area_scale = 4.5 # 放大倍数
    min_image_sample_size = 200 ** 2
    max_image_sample_size = 250 ** 2 # 限制最大采样区域

    # detection parameters 检测参数,更细粒度的坐标
    refinement_iterations = 1           # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5
    clamp_position = False              # clamp the target position to be inside the image

    # learning parameters 学习参数，滤波器更新使用的样本数，滤波器更新间隔
    output_sigma_factor = 1 / 8.     # label function sigma
    learning_rate = 0.010
    num_samples = 50 # 保存50个样本
    sample_replace_startegy = 'lowest_prior'
    lt_size = 0
    train_gap = 5
    skip_after_frame = 1
    use_detection_sample = True

    # factorized convolution parameters 分解卷积参数，PCA降维
    use_projection_matrix = True
    update_projection_matrix = True
    proj_init_method = 'pca'
    projection_reg = 5e-8

    # generative sample space model parameters
    use_sample_merge = True
    sample_merge_type = 'merge'
    distance_matrix_update_type = 'exact'

    # CG paramters 滤波器更新迭代次数
    CG_iter = 5 # 每次滤波器更新时的迭代次数
    init_CG_iter = 15 * 15 # 初始时次数，因为样本数少，所以速度不会慢。
    init_GN_iter = 15
    CG_use_FR = False
    CG_standard_alpha = True
    CG_forgetting_rate = 75
    precond_data_param = 0.3
    precond_reg_param= 0.015
    precond_proj_param = 35

    # regularization window paramters 正则化窗口参数
    use_reg_window = True
    reg_window_min = 1e-4
    reg_window_edge = 10e-3
    reg_window_power = 2
    reg_sparsity_threshold = 0.05

    # interpolation parameters 插值参数，更精细的坐标
    interp_method = 'bicubic'  # 双三次插值
    interp_bicubic_a = -0.75
    interp_centering = True
    interp_windowing = False

    # scale parameters
    number_of_scales = 5
    scale_step = 1.02# 1.015
    use_scale_filter = False

    # gpu
    use_gpu = True
    gpu_id = 0
