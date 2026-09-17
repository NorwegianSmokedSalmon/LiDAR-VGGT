"""Color Distance (CD) and Color Fidelity (CF), LiDAR-VGGT Eq. (16)."""
import argparse

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from ._common import colored_arrays, read_cloud, validate_search, write_report
else:
    from _common import colored_arrays, read_cloud, validate_search, write_report


def nearest_color_error(src_points, src_colors, kdtree, tgt_colors, thread_num=8,
                        return_all=False, batch_size=65536):
    """Mean RGB L2 error at spatial nearest neighbors, with no distance cutoff.

    cKDTree provides batched parallel queries. Existing Open3D KDTreeFlann
    callers are also accepted, using its per-point query interface.
    """
    validate_search(thread_num, batch_size)
    if len(src_points) == 0 or len(tgt_colors) == 0:
        raise ValueError("Both point clouds must be nonempty")
    errors = np.empty(len(src_points), dtype=np.float64)
    for start in range(0, len(src_points), batch_size):
        stop = min(start + batch_size, len(src_points))
        points = src_points[start:stop]
        if isinstance(kdtree, cKDTree):
            _, indices = kdtree.query(points, k=1, workers=thread_num)
        else:
            indices = np.fromiter((kdtree.search_knn_vector_3d(p, 1)[1][0] for p in points),
                                  dtype=np.int64, count=len(points))
        errors[start:stop] = np.linalg.norm(src_colors[start:stop] - tgt_colors[indices], axis=1)
    mean = float(errors.mean())
    return (mean, errors) if return_all else mean


def to_db(err):
    """CF = -20 log10(CD). Zero -> +inf; positive errors are not clipped."""
    values = np.asarray(err, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Color errors must be nonnegative and finite")
    with np.errstate(divide="ignore"):
        result = -20.0 * np.log10(values)
    return float(result) if result.ndim == 0 else result


def compute_color_distance(pcd0, pcd1, thread_num=8):
    """Return (CD, mean_0_to_1, mean_1_to_0, errors_0_to_1, errors_1_to_0).

    CD is the average of the two directional means, not their sum. Each
    direction has weight 1/2 even when the cloud sizes differ. RGB is [0, 1].
    """
    points0, colors0 = colored_arrays(pcd0, "source map")
    points1, colors1 = colored_arrays(pcd1, "reference map")
    mean_01, errors_01 = nearest_color_error(points0, colors0, cKDTree(points1), colors1,
                                            thread_num=thread_num, return_all=True)
    mean_10, errors_10 = nearest_color_error(points1, colors1, cKDTree(points0), colors0,
                                            thread_num=thread_num, return_all=True)
    return 0.5 * (mean_01 + mean_10), mean_01, mean_10, errors_01, errors_10


def visualize_distributions(errors_01, errors_10, output_path=None, show=True):
    """Optional diagnostic plots; CF of mean CD is distinct from mean point CF."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for errors, label in ((errors_01, "0→1"), (errors_10, "1→0")):
        db = to_db(errors)
        finite_db = db[np.isfinite(db)]
        axs[0, 0].hist(errors, bins=100, alpha=0.5, label=label)
        if finite_db.size:
            axs[0, 1].hist(finite_db, bins=100, alpha=0.5, label=label)
            axs[1, 1].plot(np.sort(finite_db), np.arange(1, len(finite_db)+1)/len(db), label=label)
        axs[1, 0].plot(np.sort(errors), np.arange(1, len(errors)+1)/len(errors), label=label)
    for ax, title in zip(axs.flat, ("RGB L2 error", "Pointwise fidelity (finite dB)",
                                   "RGB L2 error CDF", "Pointwise fidelity CDF (finite dB)")):
        ax.set_title(title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle("Zero color error has +inf fidelity and is omitted from finite-dB plots")
    fig.tight_layout()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CD and CF (Eq. 16), normalized RGB L2")
    parser.add_argument("pcd0", help="Source colored PCD or PLY")
    parser.add_argument("pcd1", help="Reference colored PCD or PLY in the same frame")
    parser.add_argument("--voxel_size", "--voxel-size", type=float, default=0.0,
                        help="Optional downsampling; default 0 evaluates all input points")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plot", help="Save error distributions to an image")
    parser.add_argument("--show", action="store_true", help="Show interactive error plots")
    parser.add_argument("--json", help="Write metrics, point counts and parameters to JSON")
    args = parser.parse_args()
    if not np.isfinite(args.voxel_size) or args.voxel_size < 0:
        parser.error("voxel size must be finite and nonnegative")
    pcd0, pcd1 = read_cloud(args.pcd0), read_cloud(args.pcd1)
    if args.voxel_size > 0:
        pcd0 = pcd0.voxel_down_sample(args.voxel_size)
        pcd1 = pcd1.voxel_down_sample(args.voxel_size)
    cd, cd01, cd10, errors01, errors10 = compute_color_distance(pcd0, pcd1, args.threads)
    cf = to_db(cd)
    print(f"Mean RGB L2 0->1: {cd01:.10g}; 1->0: {cd10:.10g}")
    print(f"CD: {cd:.10g} (RGB channel scale [0,1]); CD_rgb255: {cd*255:.10g}")
    print(f"CF: {cf:.10g} dB")
    if args.json:
        write_report(args.json, {"source": args.pcd0, "reference": args.pcd1,
                                 "CD": cd, "CD_rgb255": 255.0 * cd, "CF_dB": cf,
                                 "mean_source_to_reference": cd01, "mean_reference_to_source": cd10,
                                 "source_points": len(pcd0.points), "reference_points": len(pcd1.points),
                                 "downsample_voxel_size_m": args.voxel_size})
    if args.plot or args.show:
        visualize_distributions(errors01, errors10, args.plot, args.show)


if __name__ == "__main__":
    main()
