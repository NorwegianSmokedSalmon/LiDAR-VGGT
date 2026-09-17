"""Reference scales and regularization weights for point-cloud Sim(3)."""
import numpy as np

def estimate_s0(src, dst):
    """Estimate target/source scale from bounding boxes, falling back to RMS radii."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    src_min, src_max = src.min(axis=0), src.max(axis=0)
    dst_min, dst_max = dst.min(axis=0), dst.max(axis=0)
    diag_src = np.linalg.norm(src_max - src_min)
    diag_dst = np.linalg.norm(dst_max - dst_min)

    if diag_src > 1e-12:
        s0 = diag_dst / diag_src
    else:
        rms_src = np.sqrt(np.mean(np.sum(src_c**2, axis=1)))
        rms_dst = np.sqrt(np.mean(np.sum(dst_c**2, axis=1)))
        s0 = rms_dst / rms_src if rms_src > 1e-12 else 1.0
    return float(s0)

def estimate_lambda_energy(src, alpha=0.1):
    """Return alpha times the total squared distance to the source centroid."""
    mu_src = src.mean(axis=0)
    src_c = src - mu_src
    A_src = np.sum(np.sum(src_c**2, axis=1))
    lam = max(1e-12, alpha * A_src)
    return float(lam)

def estimate_lambda_bbox(src, beta=0.05):
    """Return beta * N * L², where L is the source bounding-box diagonal."""
    mu_src = src.mean(axis=0)
    src_c = src - mu_src
    n = src.shape[0]

    src_min, src_max = src.min(axis=0), src.max(axis=0)
    diag_src = np.linalg.norm(src_max - src_min)
    if diag_src > 1e-12:
        L = diag_src
    else:
        L = np.sqrt(np.mean(np.sum(src_c**2, axis=1)))

    lam = max(1e-12, beta * n * (L**2))
    return float(lam)
