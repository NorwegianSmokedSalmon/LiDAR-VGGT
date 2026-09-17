"""Sparse SE(3) pose graph using SciPy; no ROS, GTSAM or compiled extension."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from PGO.geometry import invert


def check_connected(count, sources, targets, fixed_nodes=(0,)):
    adjacency = coo_matrix((np.ones(len(sources)), (sources, targets)), shape=(count, count))
    components, labels = connected_components(adjacency, directed=False)
    anchored = set(labels[np.asarray(fixed_nodes, int)])
    if len(anchored) != components:
        raise ValueError(f'Pose graph is disconnected ({components} components) with unanchored components; inspect rejected overlap edges')
    return int(components)


def edge_errors(poses, sources, targets, measurements):
    ri = poses[sources, :3, :3].transpose(0, 2, 1)
    relative_r = ri @ poses[targets, :3, :3]
    relative_t = np.einsum('nij,nj->ni', ri, poses[targets, :3, 3] - poses[sources, :3, 3])
    rotations = Rotation.from_matrix(measurements[:, :3, :3].transpose(0, 2, 1) @ relative_r).as_rotvec()
    translations = relative_t - measurements[:, :3, 3]
    return np.column_stack([rotations, translations])


def warm_start(initial, chunk_ids, sources, targets, measurements, kinds):
    """Rigidly connect each chunk before nonlinear per-frame optimization.

    This preserves every within-chunk odometry measurement exactly.
    """
    poses = initial.copy()
    for chunk in sorted(set(chunk_ids))[1:]:
        edges = np.flatnonzero((kinds == 'icp') & (chunk_ids[targets] == chunk) & (chunk_ids[sources] < chunk))
        if not len(edges):
            raise ValueError(f'Chunk {chunk} has no accepted overlap constraint to an earlier chunk')
        candidates = poses[sources[edges]] @ measurements[edges] @ invert(initial[targets[edges]])
        correction = np.eye(4)
        correction[:3, :3] = Rotation.from_matrix(candidates[:, :3, :3]).mean().as_matrix()
        correction[:3, 3] = np.median(candidates[:, :3, 3], axis=0)
        selected = chunk_ids == chunk
        poses[selected] = correction @ initial[selected]
    poses[0] = initial[0]
    return poses


def optimize(initial, sources, targets, measurements, kinds, config, icp_weight, start=None, fixed_nodes=(0,)):
    initial, measurements = np.asarray(initial), np.asarray(measurements)
    sources, targets, kinds = np.asarray(sources), np.asarray(targets), np.asarray(kinds)
    check_connected(len(initial), sources, targets, fixed_nodes)
    odom_weight = float(config['weights']['odometry'])
    if not 0 < icp_weight < odom_weight:
        raise ValueError('Require 0 < overlap ICP weight < odometry weight; first pose is fixed exactly')
    seed = initial.copy() if start is None else np.asarray(start).copy()
    fixed = np.unique(np.asarray(fixed_nodes, dtype=int))
    if not len(fixed) or fixed[0] != 0 or fixed[-1] >= len(initial):
        raise ValueError('Fixed node set must include node zero and valid node indices')
    free = np.setdiff1d(np.arange(len(initial)), fixed)
    if not len(free):
        raise ValueError('No free pose variables')
    variable = np.full(len(initial), -1, int)
    variable[free] = np.arange(len(free))
    seed[fixed] = initial[fixed]
    noise = config['noise']
    sigmas = np.array([np.deg2rad(noise['rotation_sigma_deg'])] * 3 + [noise['translation_sigma_m']] * 3)
    weights = np.sqrt(np.where(kinds == 'odom', odom_weight, icp_weight))[:, None] / sigmas
    icp_mask = kinds == 'icp'
    robust_delta = float(config['solver']['icp_huber_delta'])

    def unpack(vector):
        poses = seed.copy()
        update = vector.reshape(-1, 6)
        poses[free, :3, :3] = seed[free, :3, :3] @ Rotation.from_rotvec(update[:, :3]).as_matrix()
        poses[free, :3, 3] = seed[free, :3, 3] + update[:, 3:]
        return poses

    def residual(vector):
        values = edge_errors(unpack(vector), sources, targets, measurements) * weights
        # Huber on whole overlap factors only. Odometry remains quadratic.
        norm = np.linalg.norm(values[icp_mask], axis=1)
        scale = np.ones(len(norm))
        outside = norm > robust_delta
        scale[outside] = np.sqrt(2 * robust_delta * norm[outside] - robust_delta**2) / norm[outside]
        values[icp_mask] *= scale[:, None]
        return values.ravel()

    rows, columns = [], []
    for edge, nodes in enumerate(zip(sources, targets)):
        for node in nodes:
            if variable[node] < 0:
                continue  # Eliminating this variable makes the prior exact.
            rows.extend(np.repeat(np.arange(6 * edge, 6 * edge + 6), 6))
            columns.extend(np.tile(np.arange(6 * variable[node], 6 * (variable[node]+1)), 6))
    sparsity = coo_matrix((np.ones(len(rows)), (rows, columns)),
                          shape=(6 * len(sources), 6 * len(free))).tocsr()
    x0 = np.zeros(6 * len(free))
    initial_cost = float(np.sum(residual(x0)**2) / 2)
    solution = least_squares(residual, x0, jac_sparsity=sparsity, method='trf', tr_solver='lsmr',
                             x_scale='jac', max_nfev=int(config['solver']['max_nfev']),
                             ftol=float(config['solver']['ftol']), xtol=1e-6, gtol=1e-5)
    optimized = unpack(solution.x)
    if not np.isfinite(optimized).all() or not np.array_equal(optimized[fixed], initial[fixed]):
        raise ValueError('Non-finite solution or moved fixed first pose')
    info = {'converged': bool(solution.success), 'message': solution.message, 'nfev': solution.nfev,
            'initial_cost': initial_cost, 'final_cost': float(solution.cost),
            'optimality': float(solution.optimality), 'icp_weight': float(icp_weight),
            'odometry_weight': odom_weight, 'first_pose_fixed': True, 'fixed_nodes': fixed.tolist()}
    return optimized, info
