"""State observation helpers for G1 Omnipicker."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_obs_cache = {
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

_g1_robot_dds = None
_dds_initialized = False


def _get_g1_robot_dds_instance():
    global _g1_robot_dds, _dds_initialized
    if _dds_initialized and _g1_robot_dds is not None:
        return _g1_robot_dds
    try:
        from dds.dds_master import dds_manager

        _g1_robot_dds = dds_manager.get_object("g129")
    except Exception:
        _g1_robot_dds = None
    _dds_initialized = True
    return _g1_robot_dds


def get_robot_imu_data(env: "ManagerBasedRLEnv") -> torch.Tensor:
    data = env.scene["robot"].data
    root_state = data.root_state_w
    return root_state[:, :13]


def _pad_or_slice_to_29(x: torch.Tensor) -> torch.Tensor:
    """Return [B, 29] tensor by slicing or zero-padding joint dimension."""
    b, n = x.shape
    if n == 29:
        return x
    if n > 29:
        return x[:, :29]
    out = torch.zeros((b, 29), device=x.device, dtype=x.dtype)
    out[:, :n] = x
    return out


def get_robot_boy_joint_states(
    env: "ManagerBasedRLEnv",
    enable_dds: bool = True,
) -> torch.Tensor:
    """Return [B, 87] = q29 | dq29 | tau29 for Omnipicker.

    For compatibility with Unitree DDS layout, this function always publishes 29-length
    vectors (slice or pad from robot joint arrays).
    """
    robot = env.scene["robot"]
    pos_29 = _pad_or_slice_to_29(robot.data.joint_pos)
    vel_29 = _pad_or_slice_to_29(robot.data.joint_vel)
    tau_29 = _pad_or_slice_to_29(robot.data.applied_torque)

    if enable_dds and pos_29.shape[0] > 0:
        now_ms = int(time.time() * 1000)
        if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
            dds_obj = _get_g1_robot_dds_instance()
            if dds_obj is not None:
                imu_data = get_robot_imu_data(env)
                try:
                    dds_obj.write_robot_state(
                        pos_29[0].detach().contiguous().cpu().numpy(),
                        vel_29[0].detach().contiguous().cpu().numpy(),
                        tau_29[0].detach().contiguous().cpu().numpy(),
                        imu_data[0].detach().contiguous().cpu().numpy(),
                    )
                    _obs_cache["dds_last_ms"] = now_ms
                except Exception:
                    pass

    return torch.cat([pos_29, vel_29, tau_29], dim=1)
