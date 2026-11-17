"""Small kinematics helpers (rotation maths + damped-least-squares IK)."""
from __future__ import annotations

import mujoco
import numpy as np

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def orientation_from_axes(approach_world, finger_world, approach_axis: str, finger_axis: str) -> np.ndarray:
    """Rotation matrix of the TCP frame such that its ``approach_axis`` column points
    along ``approach_world`` and its ``finger_axis`` column along ``finger_world``."""
    a = np.asarray(approach_world, float)
    a /= np.linalg.norm(a)
    f = np.asarray(finger_world, float)
    f = f - a * f.dot(a)                       # Gram-Schmidt
    f /= np.linalg.norm(f)
    ia, if_ = AXIS_INDEX[approach_axis], AXIS_INDEX[finger_axis]
    if ia == if_:
        raise ValueError("approach_axis and finger_axis must differ")
    ik = 3 - ia - if_
    R = np.zeros((3, 3))
    R[:, ia], R[:, if_] = a, f
    # right-handed: col[k] = col[i] x col[j] for cyclic (i, j, k)
    R[:, ik] = np.cross(a, f) if (ia + 1) % 3 == if_ else np.cross(f, a)
    return R


def rotation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Small-angle orientation error (world frame) that rotates R_cur towards R_des."""
    return 0.5 * sum(np.cross(R_cur[:, i], R_des[:, i]) for i in range(3))


def rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class TCP:
    """Tool centre point = a site plus a fixed offset expressed in the site frame."""

    def __init__(self, model: mujoco.MjModel, site_name: str, offset):
        self.site_id = model.site(site_name).id
        self.body_id = model.site_bodyid[self.site_id]
        self.offset = np.asarray(offset, float)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def pos(self, data) -> np.ndarray:
        return data.site_xpos[self.site_id] + data.site_xmat[self.site_id].reshape(3, 3) @ self.offset

    def mat(self, data) -> np.ndarray:
        return data.site_xmat[self.site_id].reshape(3, 3)

    def jacobian(self, model, data, dofs) -> np.ndarray:
        mujoco.mj_jac(model, data, self._jacp, self._jacr, self.pos(data), self.body_id)
        return np.vstack([self._jacp[:, dofs], self._jacr[:, dofs]])


def dls_step(J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    JJt = J @ J.T
    return J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(JJt.shape[0]), err)


def solve_ik(model, data, tcp: TCP, qpos_adr, dof_adr, target_pos, target_R, q_limits,
             q_seed=None, iters=300, damping=0.05, rot_weight=1.0, tol=1e-4):
    """Iterative IK on ``data`` (modified in place).  Returns (q, pos_error_norm)."""
    if q_seed is not None:
        data.qpos[qpos_adr] = q_seed
    err_n = np.inf
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        e = np.concatenate([target_pos - tcp.pos(data), rot_weight * rotation_error(tcp.mat(data), target_R)])
        err_n = np.linalg.norm(e[:3])
        if err_n < tol and np.linalg.norm(e[3:]) < 10 * tol:
            break
        dq = dls_step(tcp.jacobian(model, data, dof_adr), e, damping)
        dq = np.clip(dq, -0.2, 0.2)
        data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr] + dq, q_limits[:, 0], q_limits[:, 1])
    mujoco.mj_kinematics(model, data)
    return data.qpos[qpos_adr].copy(), err_n
