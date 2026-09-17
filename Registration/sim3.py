"""Pose matching, Sim(3) solvers, and legacy trajectory visualization helpers."""
import numpy as np
import os
import bisect
from typing import Tuple
from scipy.spatial.transform import Rotation as R_
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import logging
import torch
import roma
logger = logging.getLogger(__name__)


def extended_umeyama(
    A: np.ndarray,
    B: np.ndarray,
    weights: np.ndarray = None,
    allow_scale: bool = True,
    eps: float = 1e-10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Closed-form SIM(3) estimation (Umeyama-style) that solves for s, R, t minimizing
        sum_i w_i * || B_i - s * R @ A_i - t ||^2

    Args:
        A: (N,3) source points (x_i)
        B: (N,3) target points (y_i)
        weights: (N,) optional non-negative weights. If None, equal weights are used.
        allow_scale: if False, force s = 1 (SE(3) case).
        eps: numerical eps for degeneracy checks.

    Returns:
        s: scalar scale
        R: (3,3) rotation matrix (orthonormal, det = +1 unless degenerate fallback)
        t: (3,) translation vector
        error: (N,) error
    """
    assert A.ndim == 2 and A.shape[1] == 3
    assert B.ndim == 2 and B.shape[1] == 3
    assert A.shape[0] == B.shape[0], "A and B must have same number of points (correspondences)."
    N = A.shape[0]

    if weights is None:
        w = np.ones(N, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        assert w.shape == (N,)
        # ensure non-negative
        w = np.maximum(w, 0.0)

    Wsum = np.sum(w)
    if Wsum <= eps:
        raise ValueError("Sum of weights is zero or too small.")

    # Weighted centroids
    mu_A = (w[:, None] * A).sum(axis=0) / Wsum
    mu_B = (w[:, None] * B).sum(axis=0) / Wsum

    # Centered coordinates
    X = B - mu_B  # shape (N,3) : y_i - mu_B
    Y = A - mu_A  # shape (N,3) : x_i - mu_A

    # Weighted cross-covariance H = sum_i w_i * Y_i * X_i^T  (3x3)
    H = (Y.T * w) @ X  # (3, N) * (N, 3) -> (3,3)

    # SVD of H
    U, S_vals, Vt = np.linalg.svd(H, full_matrices=True)
    V = Vt.T
    # Compute determinant correction to ensure rotation det = +1
    det_term = np.linalg.det(V @ U.T)
    D = np.eye(3)
    if det_term < 0:
        D[2, 2] = -1.0

    # rotation
    R = V @ D @ U.T

    # compute variance of Y: var_y = sum_i w_i * ||Y_i||^2
    var_y = (w * np.sum(Y * Y, axis=1)).sum()

    if not allow_scale:
        s = 1.0
    else:
        if var_y <= eps:
            # degenerate: all B co-located (no scale info). fallback s = 1
            s = 1.0
        else:
            # scale: s = tr(Sigma * D) / var_y
            # S_vals are singular values (vector)
            # tr(Sigma * D) = sum_i D_ii * S_vals_i (since Sigma diag = S_vals)
            tr_SD = S_vals[0] * D[0, 0] + S_vals[1] * D[1, 1] + S_vals[2] * D[2, 2]
            s = tr_SD / var_y

    # translation
    t = mu_B - s * R @ mu_A

    # compute error
    transformed_A = s * (R @ A.T).T + t
    error = np.mean(np.linalg.norm(B - transformed_A, axis=1))

    return float(s), R, t, error


def refine_rotation_with_attitude(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
    initial_R: np.ndarray,
    num_frames: int = 5
) -> np.ndarray:
    """
    Refine rotation matrix using attitude alignment from first few frames.

    This function computes a rotation correction by analyzing the attitude
    differences between source and target poses after initial transformation.

    Args:
        source_poses: (N, 7) source pose array [x, y, z, qx, qy, qz, qw]
        target_poses: (N, 7) target pose array [x, y, z, qx, qy, qz, qw]
        initial_R: (3, 3) initial rotation matrix from SIM3 transform
        num_frames: Number of frames to use for attitude analysis (default: 5)

    Returns:
        refined_R: (3, 3) refined rotation matrix

    Raises:
        ValueError: If insufficient data for refinement
    """
    if len(source_poses) < num_frames or len(target_poses) < num_frames:
        logger.warning(f"Insufficient data for refinement. Using initial rotation.")
        return initial_R

    # Extract quaternions from first num_frames
    src_quats = source_poses[:num_frames, 3:7]  # (K, 4)
    tgt_quats = target_poses[:num_frames, 3:7]  # (K, 4)

    # Convert quaternions to rotation matrices
    src_R_matrices = R_.from_quat(src_quats).as_matrix()  # (K, 3, 3)
    tgt_R_matrices = R_.from_quat(tgt_quats).as_matrix()  # (K, 3, 3)

    # Transform source rotations using initial transformation
    transformed_src_R = initial_R @ src_R_matrices  # (K, 3, 3)

    # Compute rotation differences for each frame
    # ΔR_i = R_target_i · (R_transformed_source_i)^T
    delta_R_sum = np.zeros((3, 3))
    for tgt_R, trans_src_R in zip(tgt_R_matrices, transformed_src_R):
        delta_R_sum += tgt_R @ trans_src_R.T

    # Compute average rotation difference
    delta_R_mean = delta_R_sum / num_frames

    # Project to SO(3) using SVD
    U, _, Vt = np.linalg.svd(delta_R_mean)
    delta_R = U @ Vt

    # Ensure proper rotation (det = +1, not -1)
    if np.linalg.det(delta_R) < 0:
        U[:, -1] *= -1
        delta_R = U @ Vt

    # Apply rotation correction
    refined_R = delta_R @ initial_R

    logger.info(f"Rotation refinement applied using {num_frames} frames")
    logger.debug(f"Determinant of delta_R: {np.linalg.det(delta_R):.6f}")

    return refined_R


def sim3(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
    allow_scale: bool = True,
    eps: float = 1e-10,
    refine_rotation: bool = False,
    num_refinement_frames: int = 5,
    match: bool = False,
    estimate_scale: bool = False,
    init_R: np.ndarray = None,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Compute SIM3 transformation with optional rotation refinement.

    Args:
        source_poses: (N, 7) source pose array [x, y, z, qx, qy, qz, qw]
        target_poses: (N, 7) target pose array [x, y, z, qx, qy, qz, qw]
        allow_scale: Whether to allow scaling in transformation
        eps: Numerical tolerance
        refine_rotation: Whether to apply rotation refinement
        num_refinement_frames: Number of frames for rotation refinement
        match: Whether to match the point clouds
        trim_ratio: Percentage of correspondences to keep (0 < trim_ratio <= 1)
        estimate_scale: Whether to estimate scale
        init_R: Initial rotation matrix


    Returns:
        Tuple of (scale, rotation_matrix, translation_vector, error)
    """
    # estimate scale
    if estimate_scale:
        centered_source = source_poses[:, :3] - source_poses[:, :3].mean(axis=0)
        centered_target = target_poses[:, :3] - target_poses[:, :3].mean(axis=0)

        rotated_source = (init_R @ centered_source.T).T
        trace_C_target = np.sum(np.linalg.norm(centered_target, axis=1)**2) / centered_source.shape[0]
        trace_C_rotated_source = np.sum(np.linalg.norm(rotated_source, axis=1)**2) / centered_source.shape[0]
        s = np.sqrt(trace_C_target / trace_C_rotated_source)
        init_t_corrected = np.mean(target_poses, axis=0) - s * (init_R @ np.mean(source_poses, axis=0))
        target_poses = s * (init_R @ source_poses.T).T + init_t_corrected

    # match
    if match:
        if source_poses.shape[0] < target_poses.shape[0]:
            tree = cKDTree(target_poses[:, :3])
            dists, idxs = tree.query(source_poses[:, :3])
            target_poses = target_poses[idxs]
        else:
            tree = cKDTree(source_poses[:, :3])
            dists, idxs = tree.query(target_poses[:, :3])
            source_poses = source_poses[idxs]

    # Compute initial SIM3 transformation
    s, R, t, error = extended_umeyama(
        A=source_poses[:, :3],  # positions only
        B=target_poses[:, :3],  # positions only
        allow_scale=allow_scale,
        eps=eps
    )

    # Apply rotation refinement if requested
    if refine_rotation:
        logger.info(f"Initial SIM3: scale={s:.6f}, error={error:.6f}")
        R = refine_rotation_with_attitude(
            source_poses=source_poses,
            target_poses=target_poses,
            initial_R=R,
            num_frames=num_refinement_frames
        )

    return s, R, t, error

#Jack RMSE计算
def compute_rmse(src, tgt):
    return np.sqrt(np.mean(np.sum((src - tgt) ** 2, axis=1)))

#Jack 应用 Sim3 变换
def apply_sim3_transform(points, R, t, s):
    return s * (R @ points.T).T + t

#Jack sim3变换 Umeyama 方法闭式解法， 点云
def estimate_sim3_umeyama_pcd(src, dst, with_scale=True):
    print("using Umeyama to solve Sim3")
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3

    n = src.shape[0]
    if n == 0:
        raise ValueError("No points to estimate transform.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / n
    U, S, Vt = np.linalg.svd(cov)

    det_sign = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, 1.0, det_sign])
    R = U @ D @ Vt

    if with_scale:
        var_src = np.sum(src_centered**2) / n
        s = np.sum(S * np.diag(D)) / var_src if var_src > 0 else 1.0
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src
    return R, t, s

# Jack 正则化的Sim3求解
def estimate_sim3_regularized_pcd(src, dst, with_scale=True, s0=None, lam=1e4, max_iters=50, tol=1e-6):
    """
    正则化的 Sim3 迭代求解 (交替优化)
    """

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    if n == 0:
        raise ValueError("No points to estimate transform.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    # ---------- Step 0: 用 Umeyama 初始化 ----------
    R_init, t_init, s_init = estimate_sim3_umeyama_pcd(src, dst, with_scale=True)
    if s0 is None:
        s0 = s_init  # 如果用户没传 s0，就用 Umeyama 的结果

    R = R_init
    t = t_init
    s = s_init

    prev_s = None
    for it in range(max_iters):
        # Step 1: 优化旋转
        cov = (dst_centered.T @ (src_centered * s)) / n
        U, _, Vt = np.linalg.svd(cov)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
        R = U @ D @ Vt

        # Step 2: 优化平移
        t = mu_dst - s * R @ mu_src

        # Step 3: 优化尺度 (带正则项)
        if with_scale:
            num = np.sum(dst_centered * (R @ src_centered.T).T) + lam * s0
            den = np.sum(np.sum((R @ src_centered.T).T**2, axis=1)) + lam
            s = num / den if den > 1e-12 else s0
        else:
            s = 1.0

        # ---------- 打印 ----------
        residuals = s * (R @ src.T).T + t - dst
        data_term = np.sum(np.sum(residuals**2, axis=1))
        reg_term = lam * (s - s0) ** 2
        print(f"[Iter {it}] s={s:.6f}, data={data_term:.6f}, reg={reg_term:.6f}")

        if prev_s is not None and abs(prev_s - s) < tol:
            break
        prev_s = s

    return R, t, s


# Jack 保存 Sim3 的 dict 变换为文本文件
def save_sim3_dict_to_txt(sim3_dict, save_path="sim3_pose.txt"):
    """
    保存所有chunk的Sim3变换为 s, R, t 格式
    格式：
    chunk_idx
    s
    R矩阵 3x3
    t向量 3x1
    (空一行)
    """
    with open(save_path, "w") as f:
        for idx, (s, R, t) in sim3_dict.items():
            f.write(f"{idx}\n")
            f.write(f"{s:.9f}\n")
            for row in R:
                f.write(" ".join([f"{v:.9f}" for v in row]) + "\n")
            f.write(" ".join([f"{v:.9f}" for v in t]) + "\n")
            f.write("\n")
    print(f"[INFO] 已保存Sim3变换 (s,R,t) 到 {save_path}")

# Jack for pose sim3, 读取匹配的文件，位姿
def match_poses(base_dir, images_dir, body_pcd_dir, img_time_file, lio_pose_file, time_tolerance=0.02):
    """
    Load and match image and point cloud data based on timestamps.

    Args:
        base_dir (str): Base directory containing images, pcds and timestamp files
        time_tolerance (float): Maximum time difference (in seconds) for matching

    Returns:
        tuple: (img_files, pcd_files, pcd_poses) where:
            img_files: List of matched image file paths
            pcd_files: List of matched point cloud file paths
            pcd_poses: List of poses (tx, ty, tz, qx, qy, qz, qw) for each point cloud

    Raises:
        FileNotFoundError: If required directories or files are missing
        ValueError: If no valid data or no synchronized frames found
    """
    try:
        images_dir = os.path.join(base_dir, images_dir)
        pcds_dir = os.path.join(base_dir, body_pcd_dir)
        images_time_file = os.path.join(base_dir, img_time_file)
        pcds_time_pose_file = os.path.join(base_dir, lio_pose_file)

        # Validate directories and files
        if not os.path.exists(images_dir) or not os.path.exists(pcds_dir):
            raise FileNotFoundError("Images or PCDs directory not found")
        if not os.path.exists(images_time_file) or not os.path.exists(pcds_time_pose_file):
            raise FileNotFoundError("Timestamp files not found")

        # Read image timestamps (new format: image_name timestamp)
        img_times = []
        img_files = []
        with open(images_time_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_name, ts = parts[0], parts[1]
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        img_times.append(float(ts))
                        img_files.append(img_path)
                    else:
                        print(f"Warning: Image file not found: {img_path}")

        # Read point cloud timestamps and poses (original format)
        pcd_times = []
        pcd_files = []
        pcd_poses = []
        with open(pcds_time_pose_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    idx = parts[0]
                    ts = float(parts[1])
                    tx, ty, tz = map(float, parts[2:5])
                    qw, qx, qy, qz = map(float, parts[5:9])
                    pcd_path = os.path.join(pcds_dir, f"{idx}.pcd")
                    if os.path.exists(pcd_path):
                        pcd_times.append(ts)
                        pcd_files.append(pcd_path)
                        pcd_poses.append((tx, ty, tz, qx, qy, qz, qw))
                    else:
                        print(f"Warning: Point cloud file not found: {pcd_path}")

        if not img_times or not pcd_times:
            raise ValueError("No valid image or point cloud data found")

        # Convert to numpy arrays for efficient searching
        img_times_np = np.array(img_times)
        pcd_times_np = np.array(pcd_times)

        # Initialize matched and unmatched lists
        matched_img_files = []
        matched_pcd_files = []
        matched_poses = []
        matched_img_indices = set()  # To track which images were matched
        unmatched_img_files = []    # To store unmatched images

        # Choose the closest LiDAR sample once per image. Iterating over LiDAR
        # scans duplicated images when two scans were inside the tolerance.
        if np.any(np.diff(pcd_times_np) < 0):
            raise ValueError("LiDAR timestamps must be sorted")
        for image_idx, ts in enumerate(img_times_np):
            idx = bisect.bisect_left(pcd_times_np, ts)
            candidates = []
            if idx < len(pcd_times_np):
                candidates.append((abs(pcd_times_np[idx] - ts), idx))
            if idx > 0:
                candidates.append((abs(pcd_times_np[idx - 1] - ts), idx - 1))
            dt, best_idx = min(candidates)
            if dt <= time_tolerance:
                matched_img_files.append(img_files[image_idx])
                matched_pcd_files.append(pcd_files[best_idx])
                matched_poses.append(pcd_poses[best_idx])
                matched_img_indices.add(image_idx)

        if not matched_img_files:
            raise ValueError("No synchronized frames found")

        # Find unmatched images
        for i, img_file in enumerate(img_files):
            if i not in matched_img_indices:
                unmatched_img_files.append(img_file)

        # Print matching statistics
        print(f"Successfully matched {len(matched_img_files)} image-pcd pairs")
        print(f"Unmatched images count: {len(unmatched_img_files)}")
        if unmatched_img_files:
            print("Unmatched image files:")
            for img_file in unmatched_img_files:
                print(f"  {img_file}")

        return matched_img_files, matched_pcd_files, matched_poses

    except Exception as e:
        raise ValueError(f"Failed to load file data: {str(e)}")

# Jack 可视化点云及其变换结果
def visualize_pose_sim3(source, target, transformed=None, title="Pose Sim3 Alignment"):
    """
    可视化点云及其变换结果

    Args:
        source: 源点云 (N,3)
        target: 目标点云 (N,3)
        transformed: 变换后的源点云 (N,3)，可选
        title: 图表标题
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制目标点云（蓝色）
    ax.scatter(target[:,0], target[:,1], target[:,2],
               c='b', label='Target (B)', alpha=0.5, s=10)

    # 绘制原始源点云（红色）
    ax.scatter(source[:,0], source[:,1], source[:,2],
               c='r', label='Source (A)', alpha=0.3, s=5)

    # 绘制变换后的源点云（绿色）
    if transformed is not None:
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2],
                   c='g', label='Transformed A', alpha=0.7, s=15)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    # 添加坐标系箭头
    axis_length = 0.5 * np.max(np.abs(target))
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1)

    plt.tight_layout()
    plt.show()


def visualize_pose_sim3_with_trajectory(source, target, transformed=None, title="Pose Sim3 Alignment"):
    """
    可视化轨迹投影 (XY, XZ, YZ 三个平面)
    保证：
    1. 三个平面刻度间距相等 (默认 step=10)
    2. 坐标比例一致
    3. 轨迹显示在图像中央
    """

    def plot_projection(ax, x, y, color, label, marker, alpha):
        ax.plot(x, y, color=color, linewidth=2, alpha=alpha*1.2)
        ax.scatter(x, y, color=color, label=label, marker=marker, alpha=alpha, s=25)

    def set_ticks_and_limits_equal(ax, x, y, step=10.0, margin=1.1):
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        half_range = max(max_x - min_x, max_y - min_y) / 2 * margin

        x_min, x_max = center_x - half_range, center_x + half_range
        y_min, y_max = center_y - half_range, center_y + half_range

        x_start = np.floor(x_min / step) * step
        x_end = np.ceil(x_max / step) * step
        y_start = np.floor(y_min / step) * step
        y_end = np.ceil(y_max / step) * step

        ax.set_xlim(x_start, x_end)
        ax.set_ylim(y_start, y_end)
        ax.set_xticks(np.arange(x_start, x_end + step, step))
        ax.set_yticks(np.arange(y_start, y_end + step, step))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

    # ================== XY 平面 ==================
    fig_xy, ax_xy = plt.subplots(figsize=(7, 6))
    plot_projection(ax_xy, target[:,0], target[:,1], 'b', 'LIO Poses', 'o', 0.35)
    if transformed is not None:
        plot_projection(ax_xy, transformed[:,0], transformed[:,1], 'g', 'Transformed VGGT Poses', '^', 0.6)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("XY Plane Trajectory")
    combined_x = np.concatenate([target[:,0], transformed[:,0] if transformed is not None else target[:,0]])
    combined_y = np.concatenate([target[:,1], transformed[:,1] if transformed is not None else target[:,1]])
    set_ticks_and_limits_equal(ax_xy, combined_x, combined_y, step=1.0, margin=1.1)
    ax_xy.legend()
    plt.tight_layout()
    plt.show()

    # ================== XZ 平面 ==================
    fig_xz, ax_xz = plt.subplots(figsize=(7, 6))
    plot_projection(ax_xz, target[:,0], target[:,2], 'b', 'LIO Poses', 'o', 0.35)
    if transformed is not None:
        plot_projection(ax_xz, transformed[:,0], transformed[:,2], 'g', 'Transformed VGGT Poses', '^', 0.6)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title("XZ Plane Trajectory")
    combined_x = np.concatenate([target[:,0], transformed[:,0] if transformed is not None else target[:,0]])
    combined_z = np.concatenate([target[:,2], transformed[:,2] if transformed is not None else target[:,2]])
    set_ticks_and_limits_equal(ax_xz, combined_x, combined_z, step=1.0, margin=1.1)
    ax_xz.legend()
    plt.tight_layout()
    plt.show()

    # ================== YZ 平面 ==================
    fig_yz, ax_yz = plt.subplots(figsize=(7, 6))
    plot_projection(ax_yz, target[:,1], target[:,2], 'b', 'LIO Poses', 'o', 0.35)
    if transformed is not None:
        plot_projection(ax_yz, transformed[:,1], transformed[:,2], 'g', 'Transformed VGGT Poses', '^', 0.6)
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_title("YZ Plane Trajectory")
    combined_y = np.concatenate([target[:,1], transformed[:,1] if transformed is not None else target[:,1]])
    combined_z = np.concatenate([target[:,2], transformed[:,2] if transformed is not None else target[:,2]])
    set_ticks_and_limits_equal(ax_yz, combined_y, combined_z, step=1.0, margin=1.1)
    ax_yz.legend()
    plt.tight_layout()
    plt.show()


# 从npy文件中加载相机位姿
def load_cameraposes_from_npy(base_dir, npy_dir, chunk_size):
    """
    从 preprocessed_chunk 文件夹中读取所有 chunk 文件，并按 chunk 提取
    相机在世界坐标系中的位置与姿态（四元数）。

    Args:
        base_dir (str): 包含 preprocessed_chunk 文件夹的基目录

    Returns:
        dict: key   -> chunk 文件名（不含扩展名）
              value -> dict 包含两个键
                 'positions' : (40, 3) 相机位置
                 'quaternions': (40, 4) 四元数 [x, y, z, w]
    """
    chunk_dir = os.path.join(base_dir, npy_dir)
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"preprocessed_chunk directory not found in {base_dir}")

    # 按数字顺序排序
    chunk_files = sorted(
        [f for f in os.listdir(chunk_dir)
         if f.startswith('chunk_') and f.endswith('.npy')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_dir}")

    camera_poses_by_chunk = {}

    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        data = np.load(chunk_path, allow_pickle=True).item()

        extrinsics = data['extrinsic']          # shape: (40, 3, 4)
        chunk_name = os.path.splitext(chunk_file)[0]

        positions = np.empty((len(extrinsics), 3))
        quaternions = np.empty((len(extrinsics), 4))

        for i, E in enumerate(extrinsics):
            R_cw = E[:, :3]            # 3×3 旋转
            t_cw = E[:, 3]             # 3   平移
            # 相机中心在世界坐标系中的位置
            pos_wc = -R_cw.T @ t_cw
            # 旋转矩阵 -> 四元数 [x, y, z, w]
            quat_wc = R_.from_matrix(R_cw.T).as_quat()

            positions[i] = pos_wc
            quaternions[i] = quat_wc

        camera_poses_by_chunk[chunk_name] = {
            'positions': positions,
            'quaternions': quaternions
        }

    return camera_poses_by_chunk

# Jack 将雷达位姿转换为相机位姿
def transform_lidar_to_camera(poses_world_lidar, Rcl, Pcl):
    """
    将雷达在世界坐标系下的位姿转换为相机在世界坐标系下的位置

    Args:
        poses_world_lidar: List of tuples (tx, ty, tz, qx, qy, qz, qw)
        Rcl: 雷达到相机的旋转矩阵 (3x3)
        Pcl: 雷达到相机的平移向量 (3,)

    Returns:
        np.ndarray: 相机在世界坐标系下的位置 (N, 3)
        np.ndarray: 相机在世界坐标系下的姿态 (N, 4)
    """
    # 初始化结果数组
    positions_world_cam = np.zeros((len(poses_world_lidar), 3))
    quaternions_world_cam = np.zeros((len(poses_world_lidar), 4))

    for i, pose in enumerate(poses_world_lidar):
        tx, ty, tz, qx, qy, qz, qw = pose

        Rwl = R_.from_quat([qx, qy, qz, qw]).as_matrix()
        Pwl = [tx, ty, tz]

        Plc = -Rcl.T @ Pcl
        Rwc = Rwl @ (Rcl.T)
        Pwc = Rwl @ Plc + Pwl

        positions_world_cam[i] = Pwc
        quaternions_world_cam[i] = R_.from_matrix(Rwc).as_quat()

    return positions_world_cam, quaternions_world_cam

# Jack Umeyama 方法闭式解法， Pose
def compute_pose_sim3(A, B, weights=None, allow_scale=True, eps=1e-10):
    """
    闭式求解SIM3变换 (Umeyama算法)

    Args:
        A: 源点云 (N, 3)
        B: 目标点云 (N, 3)
        weights: 权重 (N,)，可选
        allow_scale: 是否估计缩放因子
        eps: 数值稳定性阈值

    Returns:
        s: 缩放因子
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        error: 对齐误差
    """
    assert A.shape == B.shape, "点云A和B必须具有相同的形状"
    assert A.shape[1] == 3, "点云必须是3维的"

    N = A.shape[0]

    # 处理权重
    if weights is None:
        w = np.ones(N, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)  # 确保非负

    Wsum = np.sum(w)
    if Wsum <= eps:
        raise ValueError("权重和太小")

    # 计算加权质心
    mu_A = np.sum(w[:, None] * A, axis=0) / Wsum
    mu_B = np.sum(w[:, None] * B, axis=0) / Wsum

    # 中心化坐标
    A_centered = A - mu_A
    B_centered = B - mu_B

    # 计算加权协方差矩阵
    H = (A_centered.T * w) @ B_centered

    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # 确保旋转矩阵行列式为+1
    det_UV = np.linalg.det(V @ U.T)
    D = np.eye(3)
    if det_UV < 0:
        D[2, 2] = -1.0

    # 计算旋转矩阵
    R = V @ D @ U.T

    # 计算缩放因子
    if not allow_scale:
        s = 1.0
    else:
        # 计算源点云的方差
        var_A = np.sum(w * np.sum(A_centered * A_centered, axis=1))
        if var_A <= eps:
            s = 1.0
        else:
            tr_SD = S[0] * D[0, 0] + S[1] * D[1, 1] + S[2] * D[2, 2]
            s = tr_SD / var_A

    # 计算平移向量
    t = mu_B - s * R @ mu_A

    # 计算对齐误差
    transformed_A = s * (R @ A.T).T + t
    error = np.mean(np.linalg.norm(B - transformed_A, axis=1))

    return s, R, t, error

def compute_pose_sim3_roma(A, B, weights=None, allow_scale=True, eps=None):

    # 转换输入
    x = torch.tensor(A, dtype=torch.float64).unsqueeze(0)  # [1, N, 3]
    y = torch.tensor(B, dtype=torch.float64).unsqueeze(0)
    w = torch.tensor(weights, dtype=torch.float64).unsqueeze(0) if weights is not None else None

    # 调用 roma
    if allow_scale:
        R, t, s = roma.rigid_points_registration(x, y, compute_scaling=True)
        s = s.item()
    else:
        R, t = roma.rigid_points_registration(x, y, weights=w, compute_scaling=False)
        s = 1.0

    # 转 numpy
    R = R[0].cpu().numpy()        # (3,3)
    t = t[0].cpu().numpy().reshape(1, 3)  # 保持形状一致 (1,3)

    # 应用变换
    transformed_A = s * (A @ R.T) + t  # (N,3)
    error = np.mean(np.linalg.norm(B - transformed_A, axis=1))

    return s, R, t.flatten(), error

def compute_pose_sim3_roma_new(A, B, weights=None, allow_scale=True, roma=None):
    """
    A, B: np.ndarray (N,3)
    weights: None or (N,)
    roma: module providing rigid_points_registration (if None, assumes roma is imported globally)
    returns: s, R, t (shape (3,)), error, info_dict
    info_dict 包含选用的候选和若干诊断信息
    """
    if roma is None:
        import roma as roma  # 若你已在其他处 import roma，可去掉这行

    # to torch (batch dim)
    x = torch.tensor(A, dtype=torch.float64).unsqueeze(0)  # [1, N, 3]
    y = torch.tensor(B, dtype=torch.float64).unsqueeze(0)
    w = torch.tensor(weights, dtype=torch.float64).unsqueeze(0) if weights is not None else None

    # 调用 roma
    if allow_scale:
        R_torch, t_torch, s_torch = roma.rigid_points_registration(x, y, weights=w, compute_scaling=True)
        s_val = float(s_torch.item()) if hasattr(s_torch, "item") else float(s_torch)
    else:
        R_torch, t_torch = roma.rigid_points_registration(x, y, weights=w, compute_scaling=False)
        s_val = 1.0

    # 取第0个 batch -> numpy
    R = R_torch[0].cpu().numpy()        # (3,3)
    t = t_torch[0].cpu().numpy().reshape(3,)  # (3,)

    # 一个小工具：对 R 做 SVD 正交化并强制 det = +1
    def orthonormalize_R(R_in):
        U, S, Vt = np.linalg.svd(R_in)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            # 修正 -1 的情况（确保是 proper rotation）
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    # 构造候选变换（常见混淆来源：R 或 R.T，原始 R 可能数值不正交）
    candidates = []
    # 原始 R（未经正交化）
    candidates.append(("R_raw", R))
    # 转置
    candidates.append(("R_T", R.T))
    # 正交化后的（来自原始）
    candidates.append(("R_orth", orthonormalize_R(R)))
    # 正交化后再转置
    candidates.append(("R_orth_T", orthonormalize_R(R).T))

    best = None
    best_err = float("inf")
    best_R = None
    best_name = None
    best_transformed = None

    for name, Rcand in candidates:
        # 两种应用方式：点矩阵左乘 (R @ p^T).T 等价于 p @ R.T
        # 我们尝试用 A @ Rcand.T（即把 Rcand 视作作用在点列的右乘）来变换 A
        transformed = s_val * (A @ Rcand.T) + t.reshape(1, 3)
        err = np.mean(np.linalg.norm(B - transformed, axis=1))
        if err < best_err:
            best_err = err
            best = (s_val, Rcand.copy(), t.copy(), err)
            best_R = Rcand.copy()
            best_name = name
            best_transformed = transformed

    # 额外诊断：若 det(best_R) < 0，则在日志中说明（通常不应发生，因为我们有 orthonormalize 选项）
    det_best = np.linalg.det(best_R)

    info = {
        "candidate_tried": [name for name, _ in candidates],
        "chosen_candidate": best_name,
        "det_chosen_R": float(det_best),
        "scale": float(s_val)
    }

    s_out, R_out, t_out, error_out = best
    return s_out, R_out, t_out.reshape(3,), error_out, info
