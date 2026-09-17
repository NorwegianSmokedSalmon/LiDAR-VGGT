"""Regularized nearest-neighbor Sim(3) and ICP registration for independent chunks.

The implementation is shared by the baseline CLI and regression tests.
"""
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

from Registration import regularization as regular
from Registration import sim3
from Utils import utils


def register_clouds(lidar_chunk_folder, vggt_chunk_folder, output_txt, output_ply_folder,
         use_regularization_sim3, use_downsample, use_energy_strategy, use_bbox_strategy,
         alpha, beta,
         max_sim3_iterations=200, min_sim3_iterations=10, scale_tolerance=6e-5,
         icp_threshold=2.0, icp_tolerance=1e-5, icp_max_iterations=500,
         vggt_voxel_size = 0.001, lidar_voxel_size = 0.001):

    os.makedirs(output_ply_folder, exist_ok=True)
    final_folder = os.path.join(output_ply_folder, "final")
    os.makedirs(final_folder, exist_ok=True)

    lidar_files = [f for f in os.listdir(lidar_chunk_folder) if f.endswith('.pcd')]
    vggt_files = [f for f in os.listdir(vggt_chunk_folder) if f.endswith('.ply')]

    # 建映射：index -> filename
    lidar_map, vggt_map = {}, {}
    for f in lidar_files:
        idx = utils.extract_first_int(f)
        if idx is not None:
            lidar_map[idx] = f
    for f in vggt_files:
        idx = utils.extract_first_int(f)
        if idx is not None:
            vggt_map[idx] = f

    pairs = []
    if lidar_map and vggt_map:
        common_idxs = sorted(set(lidar_map.keys()) & set(vggt_map.keys()))
        if not common_idxs:
            print("⚠️ 警告：两侧都带数字但没有公共索引，回退到 natural sort 配对。")
        else:
            for idx in common_idxs:
                pairs.append((lidar_map[idx], vggt_map[idx]))
    if not pairs:
        print("⚠️ 使用 natural sort 回退配对。")
        lidar_sorted = sorted(lidar_files, key=utils.natural_key)
        vggt_sorted = sorted(vggt_files, key=utils.natural_key)
        minlen = min(len(lidar_sorted), len(vggt_sorted))
        pairs = list(zip(lidar_sorted[:minlen], vggt_sorted[:minlen]))

    if not pairs:
        print("❌ 未找到可配对的点云文件。")
        return

    all_transforms = []
    metrics = []
    # Jack, 存注册的RMSE，计算平均
    rmse_list = []
    for chunk_count, (lidar_file, vggt_file) in enumerate(pairs):
        print("=" * 60)
        print(f"[{chunk_count}] Lidar: {lidar_file}  <-->  VGGt: {vggt_file}")

        lidar_pcd = o3d.io.read_point_cloud(os.path.join(lidar_chunk_folder, lidar_file))
        vggt_pcd = o3d.io.read_point_cloud(os.path.join(vggt_chunk_folder, vggt_file))

        if use_downsample:
            print("downsample!!!!!!!")
            lidar_pcd = utils.downsample_point_cloud(lidar_pcd, voxel_size=lidar_voxel_size)
            vggt_pcd = utils.downsample_point_cloud(vggt_pcd, voxel_size=vggt_voxel_size)


        lidar_points = np.asarray(lidar_pcd.points)
        vggt_points = np.asarray(vggt_pcd.points)

        print(f"  After downsampling: Lidar {len(lidar_points)}, VGGt {len(vggt_points)}")
        if len(lidar_points) == 0 or len(vggt_points) == 0:
            print("  跳过：某一侧点云为空。")
            continue

        # 迭代求解 Sim3（并保存每次迭代结果）
        trans_list = []
        cumulative_sim3 = np.eye(4, dtype=np.float64)
        current_points = vggt_points.copy()
        prev_rmse = None
        prev_scale = None

        # 为防止每个 chunk 的 iter 文件互相覆盖，按 chunk 单独建文件夹
        chunk_iter_base = os.path.join(output_ply_folder, f"chunk_{chunk_count}")
        os.makedirs(chunk_iter_base, exist_ok=True)

        # lam 估计
        lam = 1e4
        if use_regularization_sim3:
            if use_energy_strategy:
                print("  使用能量策略估计 lambda")
                lam = regular.estimate_lambda_energy(current_points, alpha)
            elif use_bbox_strategy:
                print("  使用包围盒策略估计 lambda")
                lam = regular.estimate_lambda_bbox(current_points, beta)
            print(f"  正则化 Sim3: use_regularization_sim3={use_regularization_sim3}, lambda={lam:.6f}")
        kdtree = KDTree(lidar_points)
        for it in range(max_sim3_iterations):
            print(f"  Iter {it}: KDTree + Umeyama")
            dist, idx = kdtree.query(current_points, workers=min(8, os.cpu_count() or 1))
            matched_lidar_points = lidar_points[idx]

            try:
                if use_regularization_sim3:
                    R, t, s = sim3.estimate_sim3_regularized_pcd(current_points, matched_lidar_points, with_scale=True,
                                                       s0=1.0, lam=lam, max_iters=50, tol=1e-6)
                else:
                    R, t, s = sim3.estimate_sim3_umeyama_pcd(current_points, matched_lidar_points, with_scale=True)
            except Exception as e:
                print("  Umeyama 估计失败：", e)
                break

            transformed_points = sim3.apply_sim3_transform(current_points, R, t, s)
            cumulative_sim3 = utils.mat4_from_sim3(R, t, s) @ cumulative_sim3

            rmse = sim3.compute_rmse(transformed_points, matched_lidar_points)
            print(f"    scale: {s:.6f}, RMSE: {rmse:.6f}")

            trans_list.append((R, t, s, rmse))

            # 保存当前迭代点云（每个 chunk 放在独立子文件夹）
            iter_folder = os.path.join(chunk_iter_base, f"iter_{it}")
            os.makedirs(iter_folder, exist_ok=True)
            pcd_iter = o3d.geometry.PointCloud()
            pcd_iter.points = o3d.utility.Vector3dVector(transformed_points)
            if np.asarray(vggt_pcd.colors).size != 0:
                pcd_iter.colors = vggt_pcd.colors
            o3d.io.write_point_cloud(os.path.join(iter_folder, vggt_file), pcd_iter)

            # 判断 Sim3 收敛：RMSE 和 scale 都稳定
            if prev_rmse is not None and prev_scale is not None and it >= min_sim3_iterations:
                if  abs(float(prev_scale) - float(s)) < scale_tolerance:

                    print(f"    ✅ Sim3 收敛 at iter {it}, ΔRMSE={abs(prev_rmse-rmse):.6f}, Δscale={abs(prev_scale-s):.6f}")
                    converged = True
                    current_points = transformed_points
                    break

            prev_rmse = rmse
            prev_scale = s
            current_points = transformed_points\

        # kdtreree 释放
        del kdtree

        # 如果没收敛，会使用最后一次结果继续做 ICP
        # === 切换到 ICP (SE3) 并打印逐迭代日志 ===
        print("👉 切换到 ICP (SE3 配准) 并打印逐步迭代日志")
        src_icp = o3d.geometry.PointCloud()
        src_icp.points = o3d.utility.Vector3dVector(current_points)
        if np.asarray(vggt_pcd.colors).size != 0:
            src_icp.colors = vggt_pcd.colors

        # ICP 逐步迭代（每次调用 registration_icp 做 1 次迭代），并打印 fitness / inlier_rmse
        init_trans = np.eye(4)
        est_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        prev_inlier_rmse = None
        icp_converged = False

        # for icp_it in range(icp_max_iters):
        for icp_it in range(1):
            res = o3d.pipelines.registration.registration_icp(
                src_icp, lidar_pcd, icp_threshold, init_trans,
                est_method,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations)
            )
            # res.transformation 是当前增量变换（把 src_icp 对齐到 target 的变换）
            # 把增量应用到 src_icp（这样下一次迭代基于更新后的 src）
            try:
                src_icp.transform(res.transformation)
            except Exception as e:
                print("  警告：应用 ICP 增量变换失败：", e)
                break
            print("第{}次 ICP 迭代结果:".format(icp_it))
            print("相对变换：", res.transformation)
            init_trans = res.transformation @ init_trans  # 累计变换
            # init_trans = res.transformation  # 累计变换
            fitness = getattr(res, "fitness", None)
            inlier_rmse = getattr(res, "inlier_rmse", None)
            # 某些 Open3D 版本/绑定可能不返回 fitness/inlier_rmse，做安全检查
            print(f"    ICP iter {icp_it}: fitness={fitness}, inlier_rmse={inlier_rmse}")

            if prev_inlier_rmse is not None and inlier_rmse is not None:
                if abs(prev_inlier_rmse - inlier_rmse) < icp_tolerance:
                    print(f"    ✅ ICP 收敛 at iter {icp_it}, Δinlier_rmse={abs(prev_inlier_rmse - inlier_rmse):.6f}")
                    icp_converged = True
                    break
            prev_inlier_rmse = inlier_rmse

        # ICP 最终信息（安全打印）
        final_fitness = getattr(res, "fitness", None)
        final_inlier_rmse = getattr(res, "inlier_rmse", None)
        print(f"  ICP 结束: fitness={final_fitness}, inlier_rmse={final_inlier_rmse}")
        rmse_list.append(final_inlier_rmse)

        # 保存 ICP 最终点云到 final 文件夹
        final_path = os.path.join(final_folder, vggt_file)
        o3d.io.write_point_cloud(final_path, src_icp)
        print(f"  Final (ICP) saved: {final_path}")

        # all_transforms.append(trans_list)


        sim3_mat = cumulative_sim3

        # ICP矩阵（init_trans保存了最终SE3）
        icp_mat = np.eye(4, dtype=np.float64)
        try:
            icp_mat = init_trans
        except:
            pass

        # 最终变换 = ICP * Sim3
        final_chunk_mat = icp_mat @ sim3_mat
        all_transforms.append(final_chunk_mat)
        utils.save_sim3_transforms(all_transforms, output_txt)
        metrics.append({"lidar_file": lidar_file, "vggt_file": vggt_file,
                        "fitness": float(final_fitness), "inlier_rmse": float(final_inlier_rmse),
                        "transform": final_chunk_mat.tolist(), "output": final_path})

    print("All done.")
    print("all rmse_list:", rmse_list)
    print("平均RMSE:", np.mean(rmse_list))
    return metrics
