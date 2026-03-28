from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return v / norm


def rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation matrix that aligns unit vector a to unit vector b."""
    a = normalize(a)
    b = normalize(b)
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return np.eye(3)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, axis))
        x, y, z = axis
        k_mat = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * (k_mat @ k_mat)

    v_cross = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + v_cross + v_cross @ v_cross * ((1.0 - c) / (s * s))


def euler_zxy_from_matrix(r_mat: np.ndarray) -> Tuple[float, float, float]:
    """Return Euler angles in degrees for BVH channel order Zrotation Xrotation Yrotation."""
    r21 = max(-1.0, min(1.0, r_mat[2, 1]))
    x = math.asin(r21)
    cx = math.cos(x)

    if abs(cx) < 1e-7:
        z = math.atan2(-r_mat[0, 2], r_mat[0, 0])
        y = 0.0
    else:
        z = math.atan2(-r_mat[0, 1], r_mat[1, 1])
        y = math.atan2(-r_mat[2, 0], r_mat[2, 2])

    return (math.degrees(z), math.degrees(x), math.degrees(y))
