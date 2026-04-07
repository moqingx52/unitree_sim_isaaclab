# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1 Omnipicker 最小加载配置：仅验证 USD 可 spawn，执行器用正则覆盖全部关节。"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 与 unitree.py 一致：优先 PROJECT_ROOT；否则以本仓库根目录推断
_repo_root = os.environ.get("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G1_OMNIPICKER_USD_PATH = os.path.join(_repo_root, "assets", "robots", "G1_omnipicker", "robot.usd")

G1_OMNIPICKER_MINIMAL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_OMNIPICKER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
)


def replace_usd_path(cfg: ArticulationCfg, usd_path: str) -> ArticulationCfg:
    """用绝对路径覆盖 spawn.usd_path（便于命令行指向服务器上的 USD）。"""
    return cfg.replace(spawn=cfg.spawn.replace(usd_path=usd_path))
