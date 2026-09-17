"""Point-cloud I/O, transform serialization, and compatibility utilities."""
import numpy as np
import open3d as o3d
import re
import yaml

#Jack 点云读取
def load_points_from_file(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    return pts

def mat4_from_sim3(R, t, s):
    """Build the homogeneous Sim(3) matrix with blocks [sR, t; 0, 1]."""
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = s * R
    T[:3,3] = t
    return T

#Jack 降采样
def downsample_point_cloud(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)

#Jack 保存最终的 Sim3 变换矩阵到文本
def save_sim3_transforms(transforms, output_path):
    # Each chunk has its own transform; never compose unrelated chunks.
    with open(output_path, 'w') as f:
        for transform in transforms:
            np.savetxt(f, transform, fmt="%.12f")
            f.write("\n")

# Jack 提取文件名中的第一个整数（用于按自然数匹配）
def extract_first_int(filename):
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else None

# Jack natural sort key, 自然数顺序排列
def natural_key(s):
    parts = re.split(r'(\d+)', s)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

# Jack 读取config文件
def load_config(path="../config/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# Jack 保存点云为 PLY 文件
def save_ply(points, colors, filename):
    """
    将点云和颜色数据保存为 .ply 文件（二进制小端格式）。
    - points: (N, 3) float32 点云坐标
    - colors: (N, 3) uint8 颜色值 (RGB)
    - filename: 输出文件路径
    """
    with open(filename, 'wb') as f:
        write_ply_header(f, len(points))   # 写入PLY文件头，包括点数、属性定义
        write_ply_batch(f, points, colors) # 写入点和颜色数据

# Jack 写入 PLY 文件头
def write_ply_header(f, num_vertices):
    """
    写入PLY文件头信息，指定文件格式、顶点数量和属性类型。
    - f: 文件句柄
    - num_vertices: 点的总数量
    """
    header = [
        "ply",
        "format binary_little_endian 1.0",   # 二进制小端格式
        f"element vertex {num_vertices}",   # 点的数量
        "property float x",                 # 点坐标属性
        "property float y",
        "property float z",
        "property uchar red",               # 颜色属性
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    f.write("\n".join(header).encode() + b"\n")

# Jack 批量写入 PLY 点和颜色
def write_ply_batch(f, points, colors):
    """
    将点和颜色批量写入PLY文件（二进制方式）。
    - points: (N, 3) float32 点云坐标
    - colors: (N, 3) uint8 颜色
    """
    structured = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('red', np.uint8),
        ('green', np.uint8),
        ('blue', np.uint8)
    ])
    # 将点和颜色填入结构化数组
    structured['x'] = points[:, 0]
    structured['y'] = points[:, 1]
    structured['z'] = points[:, 2]
    structured['red'] = colors[:, 0]
    structured['green'] = colors[:, 1]
    structured['blue'] = colors[:, 2]

    f.write(structured.tobytes())  # 直接写入字节流

# Jack 向水库采样池中添加新数据（向量化版本）
def vectorized_reservoir_sampling(new_pts, new_cls, current_count, reservoir_pts, reservoir_clr):
    """
    向水库采样池中添加新数据（向量化版本）。
    - new_pts: (M, 3) 新采样点
    - new_cls: (M, 3) 新采样点颜色
    - current_count: 当前已处理的点总数
    - reservoir_pts: (K, 3) 水库中的点
    - reservoir_clr: (K, 3) 水库中的颜色
    返回:
    - 更新后的点总数
    - 更新后的水库点
    - 更新后的水库颜色
    """
    k = len(reservoir_pts)          # 水库大小
    n_new = len(new_pts)            # 新点数量

    # 为新点生成随机索引，范围 [0, current_count+n_new)
    rand_indices = np.random.randint(0, current_count + n_new, size=n_new)

    # 确定哪些新点需要替换水库中的点
    replace_mask = rand_indices < k
    replace_indices = rand_indices[replace_mask]
    replace_pts = new_pts[replace_mask]
    replace_cls = new_cls[replace_mask]

    # 替换操作
    reservoir_pts[replace_indices] = replace_pts
    reservoir_clr[replace_indices] = replace_cls

    return current_count + n_new, reservoir_pts, reservoir_clr

# Jack 根据置信度过滤点云并保存PLY文件，支持全量输出或水库采样
def save_confident_pointcloud_batch(points, colors, confs, output_path, conf_threshold, sample_ratio, batch_size, s, R, t, Rcl, Pcl):
    """
    根据置信度过滤点云并保存PLY文件，支持全量输出或水库采样。
    - points: (b,H,W,3) 或 (N,3)，点云
    - colors: (b,H,W,3) 或 (N,3)，颜色
    - confs: (b,H,W) 或 (N,)，置信度
    - output_path: 输出路径
    - conf_threshold: 置信度阈值
    - sample_ratio: 采样比例（<1 表示下采样，=1 表示全量保存）
    - batch_size: 分批写入大小
    - s, R, t: 尺度、旋转、平移（点云坐标变换）
    - Rcl, Pcl: 相机-激光雷达外参（未使用）
    """
    # 标准化输入为批量形式
    if points.ndim == 2:   # (N,3) 转为 (1,N,3)
        b = 1
        points = points[np.newaxis, ...]
        colors = colors[np.newaxis, ...]
        confs = confs[np.newaxis, ...]
    elif points.ndim == 4: # (b,H,W,3)
        b = points.shape[0]
    else:
        raise ValueError("Unsupported points dimension. Must be 2 (N,3) or 4 (b,H,W,3)")

    # 计算所有批次中置信度合格点的总数
    total_valid = 0
    for i in range(b):
        cfs = confs[i].reshape(-1)
        total_valid += np.count_nonzero((cfs >= conf_threshold) & (cfs > 1e-5))

    # 采样点数
    num_samples = int(total_valid * sample_ratio) if sample_ratio < 1.0 else total_valid

    # 若无有效点，直接保存空点云
    if num_samples == 0:
        save_ply(np.zeros((0, 3), dtype=np.float32),
                 np.zeros((0, 3), dtype=np.uint8),
                 output_path)
        return

    # =============== 全量保存模式 ===============
    if sample_ratio == 1.0:
        with open(output_path, 'wb') as f:
            write_ply_header(f, num_samples)

            for i in range(b):
                pts = points[i].reshape(-1, 3).astype(np.float32)
                cls = colors[i].reshape(-1, 3).astype(np.uint8)
                cfs = confs[i].reshape(-1).astype(np.float32)

                mask = (cfs >= conf_threshold) & (cfs > 1e-5)
                valid_pts = pts[mask]
                valid_cls = cls[mask]

                # 分批写入，避免一次写入过大
                for j in range(0, len(valid_pts), batch_size):
                    batch_pts = s * (R @ valid_pts[j:j+batch_size].T).T + t
                    batch_cls = valid_cls[j:j+batch_size]
                    write_ply_batch(f, batch_pts, batch_cls)

    # =============== 水库采样模式 ===============
    else:
        reservoir_pts = np.zeros((num_samples, 3), dtype=np.float32)
        reservoir_clr = np.zeros((num_samples, 3), dtype=np.uint8)
        count = 0

        for i in range(b):
            pts = points[i].reshape(-1, 3).astype(np.float32)
            cls = colors[i].reshape(-1, 3).astype(np.uint8)
            cfs = confs[i].reshape(-1).astype(np.float32)

            mask = (cfs >= conf_threshold) & (cfs > 1e-5)
            valid_pts = pts[mask]
            valid_cls = cls[mask]
            n_valid = len(valid_pts)

            if count < num_samples:
                fill_count = min(num_samples - count, n_valid)
                reservoir_pts[count:count+fill_count] = valid_pts[:fill_count]
                reservoir_clr[count:count+fill_count] = valid_cls[:fill_count]
                count += fill_count

                if fill_count < n_valid: # 剩余的点用水库采样处理
                    remaining_pts = valid_pts[fill_count:]
                    remaining_cls = valid_cls[fill_count:]
                    count, reservoir_pts, reservoir_clr = vectorized_reservoir_sampling(
                        remaining_pts, remaining_cls, count, reservoir_pts, reservoir_clr
                    )
            else: # 已填满水库，直接进入采样逻辑
                count, reservoir_pts, reservoir_clr = vectorized_reservoir_sampling(
                    valid_pts, valid_cls, count, reservoir_pts, reservoir_clr
                )

        # 对采样后的点应用相似变换 s*R·x + t
        reservoir_pts = s * (R @ reservoir_pts.T).T + t

        # 保存结果点云
        save_ply(reservoir_pts, reservoir_clr, output_path)
