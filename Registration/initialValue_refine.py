import numpy as np
from scipy.spatial.transform import Rotation as R_

# Jack 线度判断
def is_points_near_line(points, threshold=0.999):
    """
    判断三维点集合是否接近一条直线（PCA方法）

    Args:
        points: np.ndarray, shape (N,3), 点集
        threshold: float, 最大允许的非主方向特征值比值，越小表示越直

    Returns:
        bool: True 如果点集接近直线
        linearity: float, 线性度指标 (越接近1越像直线)
        main_direction: np.ndarray, shape (3,), 主方向向量
    """
    if points.shape[0] < 2:
        return False, 0.0, None  # 少于2个点无法判断

    # 1. 中心化
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # 2. 协方差矩阵
    C = np.cov(centered, rowvar=False)

    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # 按从大到小排序
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 4. 线性度指标
    linearity = 1.0 - (eigenvalues[1] + eigenvalues[2]) / (eigenvalues[0] + 1e-12)

    # 5. 主方向向量
    main_direction = eigenvectors[:, 0]

    is_line = linearity >= threshold

    return is_line, linearity, main_direction

# Jack 使用平均旋转变换修正没有约束的Sim3计算的R
def correct_unconstrain_R(chunk_quats, matched_segment_quaternions, R, K=5):
    """
    计算前K帧的平均姿态差 (ΔR)

    Args:
        chunk_quats (ndarray): 源四元数数组，形状 (N,4)
        matched_segment_quaternions (ndarray): 目标四元数数组，形状 (N,4)
        R (ndarray): 源到世界的旋转矩阵 (3,3)
        K (int): 使用的帧数，默认 5

    Returns:
        delta_R (ndarray): 平均旋转差矩阵 (3,3)
    """
    # 取前K帧
    src_quat = chunk_quats[:K]
    tgt_quat = matched_segment_quaternions[:K]

    # 转成旋转矩阵
    src_R = R_.from_quat(src_quat).as_matrix()     # (K,3,3)
    transformed_R = R @ src_R                      # (K,3,3)
    tgt_R = R_.from_quat(tgt_quat).as_matrix()     # (K,3,3)

    # 累积 ΔR
    delta_R_sum = np.zeros((3, 3))
    for Rt, Rs in zip(tgt_R, transformed_R):
        delta_R_sum += Rt @ Rs.T

    # 平均并投影到 SO(3)
    delta_R_mean = delta_R_sum / K
    U, _, Vt = np.linalg.svd(delta_R_mean)
    delta_R = U @ Vt
    if np.linalg.det(delta_R) < 0:  # 防止反射
        U[:, -1] *= -1
        delta_R = U @ Vt

    return delta_R @ R

# Jack 解析dict，获取list
def parse_data_dict(data_dict):
    """
    解析数据字典，提取各个列表

    Args:
        data_dict (dict): 包含多个键值对的字典

    Returns:
        tuple: 包含各个列表的元组
    """
    chunk_indices = []
    scales = []
    rotations = []
    translations = []
    errors = []
    is_lines = []
    linearities = []

    for key, value in data_dict.items():
        chunk_idx, s, R, t, error, is_line, linearity = value
        chunk_indices.append(chunk_idx)
        scales.append(s)
        rotations.append(R)
        translations.append(t)
        errors.append(error)
        is_lines.append(is_line)
        linearities.append(linearity)

    return chunk_indices, scales, rotations, translations, errors, is_lines, linearities

# Jack 尺度的RANSAC判断，优先使用线度高的尺度作为内点先验 ===== 输出内点集合
def scale_RANSAC(data_dict, max_iterations=100, k_sigma=2.0, sample_num=1, use_iqr=False, use_softmax=True):
    print("===== 执行 scale_RANSAC 函数， 迭代次数: {} =====\n".format(max_iterations))
    print("  k_sigma: {}, sample_num: {}, use_iqr: {}, use_softmax: {}\n".format(k_sigma, sample_num, use_iqr, use_softmax))
    # 解析数据字典
    _, scale_list, _, _, _, line_list, linearity_list = parse_data_dict(data_dict)
    Scale_array = np.array(scale_list, dtype=float)
    Linearity_array = np.array(linearity_list, dtype=float)
    # 相关阈值参数
    mean_val = np.mean(Scale_array)
    std_val = np.std(Scale_array)
    threshold = k_sigma * std_val
    # 使用四分位距法确定阈值
    if use_iqr:
        print("  使用四分位距法确定阈值 \n")
        q1, q3 = np.percentile(Scale_array, [25, 75])
        iqr = q3 - q1
        threshold =  k_sigma * iqr
    cv = std_val / mean_val if mean_val != 0 else 0   # 变异系数 Coefficient of Variation
    print(f"Scale 平均值: {mean_val}, Scale 标准差: {std_val}, Scale 阈值: {threshold}, Scale 变异系数: {cv}\n")
    # 输出
    scale_inliers = []     #内点集合
    inlier_ratio = 0.0
    average_scale = 0.0
    best_inlier_count = 0
    best_inliers = None
    # ===== 初始化：线性度最大的点作为第一个 inlier =====
    max_idx = np.argmax(Linearity_array)
    init_scale = Scale_array[max_idx]
    # init_scale = Scale_array[15]
    diffs = np.abs(Scale_array - init_scale)
    init_inliers = diffs < threshold
    best_inlier_count = np.sum(init_inliers)
    best_inliers = init_inliers
    # 迭代RANSAC
    for _ in range(max_iterations):
        # ===== 优先按线性度采样 =====
        # 通过softmax概率映射来增大线度的区分，利用变异系数确定alpha
        if use_softmax:
            alpha = 1 / cv
            weights = np.exp(alpha * Linearity_array)
            probs = weights / np.sum(weights)
        else:
            probs = Linearity_array / (np.sum(Linearity_array))  # 概率分布
        candidate_scales = np.random.choice(Scale_array, size=sample_num, p=probs, replace=False)     # 高线性度优先

        # 遍历每个采样到的尺度，分别计算 inliers
        for cs in candidate_scales:
            diffs = np.abs(Scale_array - cs)
            inliers = diffs < threshold
            inlier_count = np.sum(inliers)

            # 更新最佳模型
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers


    # 结果统计
    if best_inliers is not None and best_inlier_count > 0:
        scale_inliers = Scale_array[best_inliers].tolist()
        average_scale = float(np.mean(scale_inliers))
        inlier_ratio = best_inlier_count / len(Scale_array)

    print("RANSAC finished.\n")
    print(f"Average scale: {average_scale}")
    print(f"Inlier ratio: {inlier_ratio}")
    print(f"Scale inliers: {scale_inliers}")

    return average_scale, inlier_ratio, scale_inliers

# Jack，通过RMSE和线度判断dict里的第i个是否需要尺度的修正
def RMSE_analyze_scale(data_dict, i):
    print("===== RMSE_analyze_scale =====\n")
    # 解析数据字典
    _, _, _, _, rmse, _, _ = parse_data_dict(data_dict)

    rmse = np.array(rmse)

    # 1. 计算 RMSE 的均值和标准差
    mean_error = np.mean(rmse)
    std_error = np.std(rmse)
    error_threshold = mean_error + 0.5 * std_error  # 设定阈值为均值加两倍标准差
    print(f"RMSE均值: {mean_error:.4f}, 标准差: {std_error:.4f}, 阈值: {error_threshold:.4f}")
    error_i = rmse[i]
    print(f"chunk {i} 的RMSE: {error_i:.4f}")

    if error_i > error_threshold:
        need_refine = True
        print(f"  ❗️ 第{i}块误差较大 ({error_i:.4f}m)，需要修正尺度")
    else:
        need_refine = False

    return need_refine

# Jack，通过线度判断第dict里的第i个是否需要尺度的修正
def Line_analyze_scale(data_dict, i):
    print("===== Line_analyze_scale =====\n")
    # 解析数据字典
    _, _, _, _, _, is_lines, linearities = parse_data_dict(data_dict)

    is_lines = np.array(is_lines)
    linearities = np.array(linearities)

    is_line_i = is_lines[i]
    linearity_i = linearities[i]

    if is_line_i:
        need_refine = False
    else:
        need_refine = True
        print(f"  ❗️ 第{i}块线性度较低 ({linearity_i:.4f})，需要修正尺度")

    return need_refine

# Jack，通过尺度的RANSAC判断dict里的第i个是否需要尺度的修正
def RANSAC_analyze_scale(data_dict, i):
    print("===== RANSAC_analyze_scale =====\n")
    # 解析数据字典
    _, scales, _, _, _, _, _ = parse_data_dict(data_dict)
    scale_i = scales[i]

    # 进行尺度的RANSAC
    _, _, scale_inliers = scale_RANSAC(data_dict, max_iterations=100, k_sigma=2.0, sample_num=1, use_iqr=False)

    # 判断 scale_i 是否在内点集合中
    if scale_i in scale_inliers:
        need_refine = False
    else:
        need_refine = True
        print(f"  ❗️ 第{i}块尺度 {scale_i:.4f} 不在RANSAC内点集合中，需要修正尺度")

    return need_refine

# Jack，通过RMSE和线度判断第dict里的第i个是否需要尺度的修正
# 误差过大 || scale ransac 失效 && 且不是线性位姿态配准，就需要修正
def judge_scale_refine(data_dict, i):
    print("===== 对 chunk {} 执行 judge_scale_refine 函数 =====\n".format(i))
    rmse_need_refine = RMSE_analyze_scale(data_dict, i)
    line_need_refine = Line_analyze_scale(data_dict, i)
    ransac_need_refine = RANSAC_analyze_scale(data_dict, i)
    print("rmse_need_refine:", rmse_need_refine)
    print("line_need_refine:", line_need_refine)
    print("ransac_need_refine:", ransac_need_refine)
    print("chunk {} need refine:".format(i), (rmse_need_refine or ransac_need_refine) and line_need_refine)
    return ransac_need_refine and line_need_refine
    # return ransac_need_refine

# Jack 对于i，找到一个最近的j，用来修正尺度
def find_nearest_valid_scale(data_dict, i):
    """
    找到距离 i 最近的、不需要尺度修正的索引及其数据

    Args:
        data_dict (dict): 数据字典
        i (int): 需要尺度修正的索引

    Returns:
        tuple: (nearest_idx, nearest_data)
               如果没有找到，则返回 (None, None)
    """
    print("===== 执行 find_nearest_valid_scale 函数 =====\n")
    n = len(data_dict)
    if n == 0:
        return None, None

    # 遍历距离
    for d in range(1, n):
        # 向左找
        left_idx = i - d
        if left_idx >= 0:
            if not judge_scale_refine(data_dict, left_idx):
                return left_idx, data_dict[left_idx]

        # 向右找
        right_idx = i + d
        if right_idx < n:
            if not judge_scale_refine(data_dict, right_idx):
                return right_idx, data_dict[right_idx]

    # 如果完全没找到
    print(f"⚠️ 没有找到距离 {i} 最近的有效尺度")
    return None, None

# Jack 修正误差大的Sim3变换
def refine_sim3_by_lidar_pose_transfer(s0, R0, t0,
                                       ref_pose, tgt_pose,
                                       chunk_positions,
                                       matched_segment_positions):
    """
    使用 chunk0 的 Sim3 + LiDAR 第一帧位姿传递修正 Sim3 参数

    Args:
        s0 (float): 参考 Sim3 缩放因子
        R0 (np.ndarray): 参考 Sim3 旋转 (3x3)
        t0 (np.ndarray): 参考 Sim3 平移 (3,)
        ref_pose (list/np.ndarray): 参考 LiDAR pose (tx,ty,tz,qx,qy,qz,qw)
        tgt_pose (list/np.ndarray): 目标 LiDAR pose (tx,ty,tz,qx,qy,qz,qw)
        chunk_positions (np.ndarray): 当前 chunk 的相机位置 (N,3)
        matched_segment_positions (np.ndarray): LiDAR 对应的匹配点 (N,3)

    Returns:
        s (float): 修正后的缩放因子
        R (np.ndarray): 修正后的旋转 (3x3)
        t (np.ndarray): 修正后的平移 (3,)
        error (float): 传递修正后的对齐误差
    """
    print("===== 执行 refine_sim3_by_lidar_pose_transfer 函数 =====\n")
    # 提取位移与四元数
    P_w_ref = np.array(ref_pose[0:3])
    q_ref = np.array(ref_pose[3:7])   # [qx,qy,qz,qw]
    R_w_ref = R_.from_quat(q_ref).as_matrix()

    P_w_tgt = np.array(tgt_pose[0:3])
    q_tgt = np.array(tgt_pose[3:7])
    R_w_tgt = R_.from_quat(q_tgt).as_matrix()

    # LiDAR 帧间刚体变换
    R_L = R_w_tgt @ R_w_ref.T
    P_L = P_w_tgt - R_L @ P_w_ref

    # 传递 Sim3
    s = s0
    R = R_L @ R0
    t = R_L @ t0 + P_L

    # 重新计算误差
    transformed_A_new = s * (R @ chunk_positions.T).T + t
    # 长度对齐，取短的
    min_len = min(len(matched_segment_positions), len(transformed_A_new))
    error = np.mean(np.linalg.norm(matched_segment_positions[:min_len] - transformed_A_new[:min_len], axis=1))

    return s, R, t, error
