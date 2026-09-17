"""Run a selectable reconstruction model -> LiDAR pose Sim3 -> regularized point-cloud Sim3 + ICP."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import yaml

from Registration import sim3
from Registration import initialValue_refine as refine
from Utils import utils
from Registration.pointcloud import register_clouds
from models.inference import chunk_ranges, natural_key, run_inference, saved_model
from models.registry import MODEL_NAMES, model_spec, configured_checkpoint
from models.predictions import load_prediction_manifest
from tools.merge_chunks import merge_result


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, help="Default: vggt, or the model in reused predictions")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--conf-threshold", type=float, help="Override model-specific point confidence threshold")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--max-chunks", type=int, default=1, help="Default: one real data chunk")
    scope.add_argument("--all-chunks", action="store_true", help="Run Chunk.chunk_num chunks from the config")
    scope.add_argument("--all-images", action="store_true", help="Process every input image, including the final partial chunk")
    parser.add_argument("--merge", action="store_true", help="Merge all registered chunks into a colored world map")
    parser.add_argument("--merge-voxel-size", type=float, default=0.1, help="Map voxel size in meters; zero exports only the original merged map")
    parser.add_argument("--chunk-size", type=int, help="Override chunk size; also rebuilds matching LiDAR chunks")
    parser.add_argument("--chunk-step", type=int)
    parser.add_argument("--refine-line", action="store_true", help="Use camera/LiDAR attitudes for nearly straight trajectories")
    parser.add_argument("--sample-ratio", type=float, help="Override point sampling, e.g. 0.05 for a quick run")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--skip-inference", action="store_true", help="Reuse this output directory's npy/manifest.json")
    reuse.add_argument("--prediction-dir", type=Path, help="Reuse canonical model predictions from another directory")
    args = parser.parse_args()
    if args.merge and (not np.isfinite(args.merge_voxel_size) or args.merge_voxel_size < 0):
        parser.error("merge-voxel-size must be finite and nonnegative")
    started = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    if args.refine_line:
        cfg["pose_sim3"]["refine_line"] = True
    data, chunk = cfg["Data"], cfg["Chunk"]
    if args.sample_ratio is not None:
        chunk["sample_ratio"] = args.sample_ratio
    base = (project_path(data["base_path"]) / data["dataset"]).resolve()
    output = (args.output_dir or PROJECT_ROOT / "outputs" / datetime.now().strftime("AMtown01_%Y%m%d_%H%M%S")).resolve()
    if not args.skip_inference and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    if cfg["pose_sim3"]["refine_scale"]:
        raise ValueError("Cross-chunk refine_scale is unsupported; use pose_sim3.refine_line for attitude refinement")
    size = args.chunk_size if args.chunk_size is not None else int(chunk["chunk_size"])
    step = args.chunk_step if args.chunk_step is not None else min(int(chunk["chunk_step"]), size)
    limit = int(chunk["chunk_num"]) if args.all_chunks else args.max_chunks
    if size < 3 or not 0 < step <= size or limit < 1 or int(chunk["start_idx"]) < 0:
        parser.error("Require chunk-size >= 3, 0 < chunk-step <= chunk-size, max-chunks > 0 and start_idx >= 0")
    if not 0 < float(chunk["sample_ratio"]) <= 1:
        raise ValueError("sample_ratio must be in (0, 1]")
    images = sorted((p for p in (base / data["images_dir"]).iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=natural_key)
    images = images[int(chunk["start_idx"]):]
    if args.all_images:
        limit = len(list(chunk_ranges(len(images), size, size - step)))
    matched_images, _, matched_poses = sim3.match_poses(
        str(base), data["images_dir"], data["body_pcd_dir"],
        data["img_time_file"], data["lio_pose_file"], time_tolerance=0.06)
    Rcl = np.asarray(cfg["calibration"]["R"]).reshape(3, 3)
    Pcl = np.asarray(cfg["calibration"]["t"])
    target_positions, _ = sim3.transform_lidar_to_camera(matched_poses, Rcl, Pcl)
    target_by_image = {str(Path(p).resolve()): pos for p, pos in zip(matched_images, target_positions)}
    required = images[:size + (limit - 1) * step]
    missing = [str(p) for p in required if str(p.resolve()) not in target_by_image]
    if missing:
        raise ValueError(f"Images without synchronized LiDAR poses: {missing[:10]}")
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    prediction_dir = args.prediction_dir.resolve() if args.prediction_dir else output / "npy"
    if args.skip_inference or args.prediction_dir:
        manifest = load_prediction_manifest(prediction_dir, images, args.model,
            chunk_size=args.chunk_size, chunk_step=args.chunk_step,
            max_chunks=None if args.all_images or args.skip_inference else limit, require_all=args.all_images)
        model_name = saved_model(manifest, args.model)
        if args.checkpoint:
            parser.error("checkpoint applies only to new inference")
    else:
        model_name = args.model or "vggt"
        manifest = run_inference(images, prediction_dir, configured_checkpoint(model_name, cfg, args.checkpoint),
                                 size, size - step, limit, device=args.device, model_name=model_name)
    if args.conf_threshold is not None:
        chunk["conf_threshold"] = args.conf_threshold
    elif model_name != "vggt":
        chunk["conf_threshold"] = model_spec(model_name)["default_min_confidence"]
    cfg["Reconstruction"] = {"model": model_name, "confidence_type": model_spec(model_name)["confidence"]}
    cfg["Chunk"]["chunk_size"] = manifest["chunk_size"]
    cfg["Chunk"]["chunk_step"] = manifest["chunk_size"] - manifest["overlap"]
    cfg["Chunk"]["chunk_num"] = len(manifest["chunks"])
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for folder in ("vggt_chunk", "lidar_chunk"):
        (output / folder).mkdir(exist_ok=True)
    timestamps = {}
    for line in (base / data["img_time_file"]).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, stamp = line.split()[:2]
            timestamps[name] = float(stamp)
    lidar_rows = np.loadtxt(base / data["lio_pose_file"])
    pose_transforms, pose_metrics = {}, []
    for entry in manifest["chunks"]:
        idx = entry["index"]
        predictions = np.load(prediction_dir / entry["file"], allow_pickle=True).item()
        extrinsics = predictions["extrinsic"]
        source = -np.einsum("nji,nj->ni", extrinsics[:, :, :3], extrinsics[:, :, 3])
        target = np.asarray([target_by_image[p] for p in entry["images"]])
        if len(source) < 3 or np.linalg.norm(source - source.mean(axis=0)) < 1e-8:
            raise ValueError(f"Chunk {idx} needs at least 3 distinct camera positions for Sim3")
        solver = sim3.compute_pose_sim3_roma_new if cfg["pose_sim3"]["use_roma"] else sim3.compute_pose_sim3
        scale, rotation, translation, error, *_ = solver(source, target, allow_scale=True)
        if cfg["pose_sim3"]["refine_line"] and refine.is_points_near_line(target)[0]:
            source_quats = Rotation.from_matrix(extrinsics[:, :, :3].transpose(0, 2, 1)).as_quat()
            _, target_quats_all = sim3.transform_lidar_to_camera(matched_poses, Rcl, Pcl)
            quat_by_image = dict(zip((str(Path(p).resolve()) for p in matched_images), target_quats_all))
            target_quats = np.asarray([quat_by_image[p] for p in entry["images"]])
            rotation = refine.correct_unconstrain_R(source_quats, target_quats, rotation, K=min(5, len(source)))
            translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
        error = float(np.mean(np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)))
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(error):
            raise ValueError(f"Invalid pose alignment for chunk {idx}")
        pose_transforms[idx] = (scale, rotation, translation)
        ply = output / "vggt_chunk" / f"{idx}_pc.ply"
        utils.save_confident_pointcloud_batch(
            predictions["world_points"].reshape(-1, 3),
            (predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8),
            predictions["world_points_conf"].reshape(-1), str(ply),
            float(chunk["conf_threshold"]), float(chunk["sample_ratio"]), 1000000,
            scale, rotation, translation, Rcl, Pcl)
        del predictions
        # Rebuild the LiDAR target for the exact image time interval in the LiDAR
        # world frame; historical chunks may use a different size or coordinate frame.
        first, last = (timestamps[Path(entry["images"][k]).name] for k in (0, -1))
        rows = lidar_rows[(lidar_rows[:, 1] >= first - 0.06) & (lidar_rows[:, 1] <= last + 0.06)]
        clouds = []
        for row in rows:
            path = base / data["body_pcd_dir"] / f"{int(row[0])}.pcd"
            if not path.is_file():
                raise FileNotFoundError(path)
            cloud = o3d.io.read_point_cloud(str(path))
            rot = Rotation.from_quat(row[[6, 7, 8, 5]]).as_matrix()
            clouds.append(np.asarray(cloud.points) @ rot.T + row[2:5])
        if not clouds:
            raise ValueError(f"No LiDAR scans for chunk {idx}")
        target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(clouds)))
        if cfg["PCD_sim3"]["downsample"]:
            target_cloud = target_cloud.voxel_down_sample(float(cfg["Downsample"]["lidar_voxel_size"]))
        if not len(target_cloud.points):
            raise ValueError(f"Empty LiDAR target for chunk {idx}")
        lidar_path = output / "lidar_chunk" / f"chunk_{idx}.pcd"
        if not o3d.io.write_point_cloud(str(lidar_path), target_cloud):
            raise OSError(f"Could not write {lidar_path}")
        pose_metrics.append({"index": idx, "frames": len(source), "scale": float(scale),
                             "mean_pose_error_m": error, "lidar_scans": len(rows),
                             "lidar_points": len(target_cloud.points)})
        print(f"Chunk {idx}: pose mean error {error:.4f}m, scale {scale:.4f}, LiDAR scans {len(rows)}", flush=True)
        del clouds, target_cloud
    sim3.save_sim3_dict_to_txt(pose_transforms, str(output / "sim3_pose_correct.txt"))
    reg = cfg["PCD_sim3"]
    metrics = register_clouds(
        str(output / "lidar_chunk"), str(output / "vggt_chunk"),
        str(output / "sim3_pcd.txt"), str(output / "vggt_chunk_reg"),
        reg["use_regularization_sim3"], reg["downsample"], reg["use_energy_strategy"], reg["use_bbox_strategy"],
        float(cfg["Regularization"]["alpha"]), float(cfg["Regularization"]["beta"]),
        int(reg["max_sim3_iterations"]), int(reg["min_sim3_iterations"]), float(reg["scale_tolerance"]),
        float(reg["icp_threshold"]), float(reg["icp_tolerance"]), int(reg["icp_max_iterations"]),
        float(cfg["Downsample"]["vggt_voxel_size"]), float(cfg["Downsample"]["lidar_voxel_size"]))
    if len(metrics or []) != len(manifest["chunks"]):
        raise RuntimeError("Not all inferred chunks completed point-cloud registration")
    # Complete transforms map raw VGGT world points directly into the LiDAR world.
    complete = []
    for metric in metrics:
        idx = utils.extract_first_int(metric["vggt_file"])
        s, r, t = pose_transforms[idx]
        complete.append(np.asarray(metric["transform"]) @ utils.mat4_from_sim3(r, t, s))
    utils.save_sim3_transforms(complete, str(output / "sim3_total.txt"))
    summary = {"method": "pose_regularized_sim3_icp", "metric_type": "nearest_neighbor_icp",
               "prediction_dir": str(prediction_dir), "inference_reused": bool(args.skip_inference or args.prediction_dir),
               "model": model_name, "dataset": str(base), "output": str(output), "seconds": time.perf_counter() - started,
               "inference": manifest, "pose_alignment": pose_metrics, "registration": metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.merge:
        merge_result(output, args.merge_voxel_size)
    print(f"Completed. Results: {output}\nSummary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
