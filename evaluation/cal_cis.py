"""Color Consistency Score (CCS), LiDAR-VGGT Eq. (18).

The historical filename and compute_cis name remain as compatibility aliases.
No voxel downsampling is applied before estimating sample color covariance.
"""
import argparse

import numpy as np

if __package__:
    from ._common import colored_arrays, read_cloud, write_report
else:
    from _common import colored_arrays, read_cloud, write_report


def compute_ccs(cloud, voxel_size=0.1, *, voxel_origin=(0., 0., 0.),
                singleton_policy="zero", return_details=False):
    """Mean sample RGB covariance trace, giving each occupied voxel equal weight.

    Eq. (18) is undefined for singleton voxels. The default extends their trace
    to zero and includes them in N_v. 'exclude' averages only voxels with n>=2;
    'error' refuses singleton voxels. These are explicit implementation choices,
    not settings stated by the paper. Grid origin and size are also configurable.
    """
    points, colors = colored_arrays(cloud)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    origin = np.asarray(voxel_origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("voxel_origin must contain three finite coordinates")
    if singleton_policy not in ("zero", "exclude", "error"):
        raise ValueError("singleton_policy must be zero, exclude or error")

    grid_float = np.floor((points - origin) / voxel_size)
    if not np.isfinite(grid_float).all() or np.any(np.abs(grid_float) >= 2.**63):
        raise ValueError("voxel indices exceed int64; increase voxel_size")
    grid = grid_float.astype(np.int64)
    del grid_float
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    grid = grid[order]
    starts = np.r_[0, np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(points)])
    del grid
    singleton_count = int(np.count_nonzero(counts == 1))
    if singleton_policy == "error" and singleton_count:
        raise ValueError(f"Eq. (18) is undefined for {singleton_count} singleton voxels")
    valid = counts >= 2
    if singleton_policy == "exclude" and not valid.any():
        raise ValueError("No voxels with at least two points")

    # Center before squaring to avoid cancellation for nearly constant colors.
    sorted_colors = colors[order]
    means = np.add.reduceat(sorted_colors, starts, axis=0) / counts[:, None]
    sum_squared = np.zeros(len(counts), dtype=np.float64)
    for channel in range(3):
        centered = sorted_colors[:, channel] - np.repeat(means[:, channel], counts)
        np.square(centered, out=centered)
        sum_squared += np.add.reduceat(centered, starts)
    traces = np.zeros(len(counts), dtype=np.float64)
    traces[valid] = sum_squared[valid] / (counts[valid] - 1)
    evaluated = traces[valid] if singleton_policy == "exclude" else traces
    score = float(evaluated.mean())
    if return_details:
        return {"CCS": score, "points": len(points), "occupied_voxels": len(counts),
                "evaluated_voxels": len(evaluated), "singleton_voxels": singleton_count,
                "singleton_policy": singleton_policy, "voxel_size_m": float(voxel_size),
                "voxel_origin": origin.tolist(), "pre_downsample": False}
    return score


def compute_cis(pcd_file, voxel_size=0.1, **kwargs):
    """Compatibility entry point; now computes the paper's CCS."""
    return compute_ccs(read_cloud(pcd_file), voxel_size, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="CCS (Eq. 18), formerly named CIS")
    parser.add_argument("--pcd", required=True, help="Colored PCD or PLY map")
    parser.add_argument("--voxel_size", "--voxel-size", type=float, default=0.1,
                        help="Voxel edge in meters (default: 0.1; paper does not specify it)")
    parser.add_argument("--voxel-origin", nargs=3, type=float, default=(0., 0., 0.))
    parser.add_argument("--singleton-policy", choices=("zero", "exclude", "error"), default="zero")
    parser.add_argument("--json", help="Write score, voxel counts and conventions to JSON")
    args = parser.parse_args()
    report = compute_cis(args.pcd, args.voxel_size, voxel_origin=args.voxel_origin,
                         singleton_policy=args.singleton_policy, return_details=True)
    print(f"CCS: {report['CCS']:.10g}")
    print(f"Occupied voxels: {report['occupied_voxels']}; singletons: "
          f"{report['singleton_voxels']} (policy: {args.singleton_policy})")
    if args.json:
        write_report(args.json, {"map": str(args.pcd), **report})


if __name__ == "__main__":
    main()
