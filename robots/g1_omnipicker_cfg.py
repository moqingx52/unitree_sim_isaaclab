# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1 Omnipicker：固定底座 + 双臂 + 双夹爪 USD。首版仅驱动 body + 双臂，不驱动 head / gripper 多关节。"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_repo_root = os.environ.get("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G1_OMNIPICKER_USD_PATH = os.path.join(_repo_root, "assets", "robots", "G1_omnipicker", "robot.usd")

G1_OMNIPICKER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_OMNIPICKER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "idx01_body_joint1": 0.0,
            "idx02_body_joint2": 0.0,
            "idx21_arm_l_joint1": 0.0,
            "idx22_arm_l_joint2": 0.0,
            "idx23_arm_l_joint3": 0.0,
            "idx24_arm_l_joint4": 0.0,
            "idx25_arm_l_joint5": 0.0,
            "idx26_arm_l_joint6": 0.0,
            "idx27_arm_l_joint7": 0.0,
            "idx61_arm_r_joint1": 0.0,
            "idx62_arm_r_joint2": 0.0,
            "idx63_arm_r_joint3": 0.0,
            "idx64_arm_r_joint4": 0.0,
            "idx65_arm_r_joint5": 0.0,
            "idx66_arm_r_joint6": 0.0,
            "idx67_arm_r_joint7": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[
                "idx01_body_joint1",
                "idx02_body_joint2",
            ],
            stiffness=80.0,
            damping=4.0,
            armature=0.01,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[r"idx2[1-7]_arm_l_joint[1-7]"],
            stiffness=60.0,
            damping=2.0,
            armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[r"idx6[1-7]_arm_r_joint[1-7]"],
            stiffness=60.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

# 仅验证 spawn / 列关节名时可用：全关节隐式执行器
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
    return cfg.replace(spawn=cfg.spawn.replace(usd_path=usd_path))


def with_root_z(cfg: ArticulationCfg, z: float) -> ArticulationCfg:
    """覆盖根节点初始高度（用于校准入地 / 离地间隙）。"""
    init = cfg.init_state
    x, y, _ = init.pos
    return cfg.replace(init_state=init.replace(pos=(float(x), float(y), float(z))))


def suggest_root_z(current_init_z: float, lowest_body_z_world: float, ground_clearance: float = 0.02) -> float:
    """使刚体最低点世界 z ≈ ground_clearance 时的建议根高度（假定地面 z=0）。"""
    return float(current_init_z) + (float(ground_clearance) - float(lowest_body_z_world))
