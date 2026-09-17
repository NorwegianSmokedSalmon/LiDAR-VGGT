"""Reference-to-reconstruction Local Color Recall (LCR), Eq. (17).

The historical filename and compute_LCCR name remain as compatibility aliases.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

if __package__:
    from ._common import colored_arrays, read_cloud, validate_search, write_report
else:
    from _common import colored_arrays, read_cloud, validate_search, write_report


def compute_lcr(map_pcd, truth_pcd, tau=0.1, r_g=0.5, save_pass=False, *,
                output_dir=".", thread_num=8, batch_size=1024,
                show_progress=False, return_details=False):
    """Fraction of reference points with a map neighbor whose RGB L2 <= 3*tau.

    Coordinates must share a metric frame; RGB channels are in [0, 1]. Empty
    neighborhoods count as misses. Defaults follow Fig. 7 / Sec. IV-B.
    """
    map_points, map_colors = colored_arrays(map_pcd, "reconstructed map")
    truth_points, truth_colors = colored_arrays(truth_pcd, "reference map")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be nonnegative and finite")
    if not np.isfinite(r_g) or r_g <= 0:
        raise ValueError("radius must be positive and finite")
    validate_search(thread_num, batch_size)
    tree = cKDTree(map_points)
    hits = np.zeros(len(truth_points), dtype=bool)
    map_hits = np.zeros(len(map_points), dtype=bool) if save_pass else None
    for start in tqdm(range(0, len(truth_points), batch_size), desc="Computing LCR",
                      disable=not show_progress):
        stop = min(start + batch_size, len(truth_points))
        neighbors = tree.query_ball_point(truth_points[start:stop], r_g,
                                         workers=thread_num, return_sorted=False)
        counts = np.fromiter((len(n) for n in neighbors), dtype=np.int64, count=stop-start)
        if not counts.any():
            continue
        map_ids = np.concatenate(neighbors).astype(np.int64, copy=False)
        ref_ids = np.repeat(np.arange(start, stop), counts)
        errors = np.linalg.norm(map_colors[map_ids] - truth_colors[ref_ids], axis=1)
        valid = errors <= 3.0 * tau
        hits[ref_ids[valid]] = True
        if save_pass:
            map_hits[map_ids[valid]] = True

    if save_pass:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save each point once, even if it occurs in several neighborhoods.
        for name, cloud, mask in (("map_pass.pcd", map_pcd, map_hits),
                                  ("truth_pass.pcd", truth_pcd, hits)):
            path = output_dir / name
            if mask.any():
                selected = cloud.select_by_index(np.flatnonzero(mask).tolist())
                if not o3d.io.write_point_cloud(str(path), selected):
                    raise OSError(f"Failed to write {path}")
            elif path.exists():
                path.unlink()  # Do not leave a previous run's passing points behind.
    score = float(hits.mean())
    if return_details:
        return {"LCR": score, "LCR_percent": 100.0 * score,
                "direction": "reference_to_reconstruction", "map_points": len(map_points),
                "reference_points": len(truth_points), "matched_reference_points": int(hits.sum()),
                "tau": float(tau), "rgb_l2_threshold": float(3 * tau), "radius_m": float(r_g)}
    return score


def compute_LCCR(map_pcd, truth_pcd, tau=0.1, r_g=0.5, save_pass=True, **kwargs):
    """Compatibility entry point; direction and threshold now follow paper LCR."""
    return compute_lcr(map_pcd, truth_pcd, tau, r_g, save_pass, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="LCR reference->map (Eq. 17), formerly LCCR")
    parser.add_argument("--map", required=True, help="Reconstructed colored PCD or PLY")
    parser.add_argument("--truth", required=True, help="Reference colored PCD or PLY in the same frame")
    parser.add_argument("--tau", type=float, default=0.1, help="RGB L2 threshold is 3*tau (default: 0.1)")
    parser.add_argument("--radius", type=float, default=0.5, help="Neighborhood radius in meters")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-pass", action="store_true", help="Export unique passing points")
    parser.add_argument("--output-dir", default=".", help="Directory for passing point clouds")
    parser.add_argument("--json", help="Write score, counts and parameters to JSON")
    args = parser.parse_args()
    report = compute_lcr(read_cloud(args.map), read_cloud(args.truth), args.tau, args.radius,
                         args.save_pass, output_dir=args.output_dir, thread_num=args.threads,
                         batch_size=args.batch_size, show_progress=True, return_details=True)
    print(f"LCR (reference -> reconstruction): {report['LCR']:.8f} ({report['LCR_percent']:.4f}%)")
    if args.json:
        write_report(args.json, {"map": args.map, "reference": args.truth, **report})


if __name__ == "__main__":
    main()
