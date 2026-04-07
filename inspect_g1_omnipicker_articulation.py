#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""加载 G1 Omnipicker，列关节/刚体，诊断根高度，可选关节位置控制。

默认使用「body + 双臂」执行器配置 ``G1_OMNIPICKER_CFG``；``--cfg minimal`` 为全关节隐式执行器（仅排查用）。

用法::

    python inspect_g1_omnipicker_articulation.py --device cuda --init_z 1.0

    python inspect_g1_omnipicker_articulation.py --device cuda --cfg minimal --exit_after_print

    python inspect_g1_omnipicker_articulation.py --device cuda --control demo

建议高度（打印 root_z、最低点 z 与推荐 init_z）::

    python inspect_g1_omnipicker_articulation.py --device cuda --headless --exit_after_print --init_z 1.0
"""

from __future__ import annotations

import argparse
import math
import os

# 与 sim_main 一致，供 robots/*.py 内路径拼接
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect / control G1 Omnipicker (spawn, Z diagnostic, joint targets)")
parser.add_argument(
    "--cfg",
    type=str,
    choices=["full", "minimal"],
    default="full",
    help="full = G1_OMNIPICKER_CFG（body+双臂）；minimal = 全关节隐式执行器",
)
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="覆盖默认 robot.usd 路径",
)
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
parser.add_argument("--env_spacing", type=float, default=2.0, help="环境网格间距")
parser.add_argument(
    "--no_ground",
    action="store_true",
    help="不生成默认地面",
)
parser.add_argument(
    "--init_z",
    type=float,
    default=None,
    help="根节点初始世界高度 z；默认 full=1.0，minimal=0.85",
)
parser.add_argument(
    "--ground_clearance",
    type=float,
    default=0.02,
    help="建议 z 时假定目标：刚体最低点离地高度 (m)，默认 0.02",
)
parser.add_argument(
    "--exit_after_print",
    action="store_true",
    help="打印后退出（建议 headless）",
)
parser.add_argument(
    "--no_print_kinematics",
    action="store_true",
    help="不打印完整关节/刚体列表（仍打印 Z 诊断摘要）",
)
parser.add_argument(
    "--control",
    type=str,
    choices=["off", "hold", "demo"],
    default="hold",
    help="off=不写目标；hold=保持默认关节位；demo=双臂叠加小幅正弦（验证 PD）",
)
parser.add_argument(
    "--demo_amp",
    type=float,
    default=0.12,
    help="demo 模式下双臂关节目标正弦幅值 (rad)",
)
parser.add_argument(
    "--demo_hz",
    type=float,
    default=0.3,
    help="demo 模式正弦频率 (Hz)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from robots.g1_omnipicker_cfg import (
    G1_OMNIPICKER_CFG,
    G1_OMNIPICKER_MINIMAL_CFG,
    replace_usd_path,
    suggest_root_z,
    with_root_z,
)


def _default_init_z() -> float:
    if args_cli.init_z is not None:
        return args_cli.init_z
    return 0.85 if args_cli.cfg == "minimal" else 1.0


def _robot_cfg() -> ArticulationCfg:
    base = G1_OMNIPICKER_MINIMAL_CFG if args_cli.cfg == "minimal" else G1_OMNIPICKER_CFG
    cfg = replace_usd_path(base, os.path.abspath(args_cli.usd_path)) if args_cli.usd_path else base
    return with_root_z(cfg, _default_init_z())


def _build_scene_cfg() -> InteractiveSceneCfg:
    rc = _robot_cfg()
    if args_cli.no_ground:

        @configclass
        class _InspectSceneCfg(InteractiveSceneCfg):
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            )
            robot: ArticulationCfg = rc

        return _InspectSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=args_cli.env_spacing,
            replicate_physics=True,
        )

    @configclass
    class _InspectSceneCfg(InteractiveSceneCfg):
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        robot: ArticulationCfg = rc

    return _InspectSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=True,
    )


def _print_articulation_info(robot) -> None:
    print("=" * 72)
    print("G1 Omnipicker — Articulation")
    print("=" * 72)
    print(f"配置: --cfg {args_cli.cfg}")
    ref_usd = (G1_OMNIPICKER_CFG if args_cli.cfg == "full" else G1_OMNIPICKER_MINIMAL_CFG).spawn.usd_path
    print(f"USD (默认): {ref_usd}")
    if args_cli.usd_path:
        print(f"USD (覆盖): {os.path.abspath(args_cli.usd_path)}")
    print()

    if args_cli.no_print_kinematics:
        print("(已 --no_print_kinematics，跳过关节/刚体完整列表)")
        print()
        return

    jn = list(robot.data.joint_names)
    bn = list(robot.data.body_names)
    print(f"关节数量: {len(jn)}")
    for i, name in enumerate(jn):
        print(f"  [{i:4d}] {name}")
    print()
    print(f"刚体数量: {len(bn)}")
    for i, name in enumerate(bn):
        print(f"  [{i:4d}] {name}")
    print()

    try:
        _j_idx, j_found = robot.find_joints(".*", preserve_order=True)
        print(f"find_joints('.*', preserve_order=True) -> {len(j_found)} 个")
    except TypeError:
        _j_idx, j_found = robot.find_joints(".*")
        print(f"find_joints('.*') -> {len(j_found)} 个")
    except Exception as e:
        print(f"find_joints: {e}")

    try:
        _b_idx, b_found = robot.find_bodies(".*")
        print(f"find_bodies('.*') -> {len(b_found)} 个")
    except Exception as e:
        print(f"find_bodies: {e}")

    print("=" * 72)


def _print_z_diagnostic(robot, init_z_used: float) -> None:
    root_z = robot.data.root_pos_w[0, 2].item()
    lowest_z = robot.data.body_pos_w[0, :, 2].min().item()
    z_new = suggest_root_z(init_z_used, lowest_z, args_cli.ground_clearance)
    print("-" * 72)
    print("Z 轴诊断（世界系，默认地面 z≈0）")
    print(f"  当前 cfg init_z = {init_z_used:.4f}")
    print(f"  root_z         = {root_z:.4f}")
    print(f"  lowest_body_z  = {lowest_z:.4f}  (所有 body_link 世界坐标 z 最小值)")
    print(f"  建议 init_z    ≈ {z_new:.4f}  (使最低点离地约 {args_cli.ground_clearance * 100:.1f} cm)")
    print(f"  公式: z_new = init_z + ({args_cli.ground_clearance} - lowest_z)")
    print("-" * 72)


def _resolve_arm_joint_indices(robot):
    """双臂 14 个 arm 关节在 articulation 中的索引（用于 demo）。"""
    idx_l, _ = robot.find_joints(r"idx2[1-7]_arm_l_joint[1-7]")
    idx_r, _ = robot.find_joints(r"idx6[1-7]_arm_r_joint[1-7]")
    if hasattr(idx_l, "tolist"):
        idx_l = idx_l.tolist()
    else:
        idx_l = list(idx_l)
    if hasattr(idx_r, "tolist"):
        idx_r = idx_r.tolist()
    else:
        idx_r = list(idx_r)
    return list(idx_l) + list(idx_r)


def _apply_joint_control(robot, sim_time: float) -> None:
    if args_cli.control == "off":
        return
    target = robot.data.default_joint_pos.clone()
    if args_cli.control == "hold":
        robot.set_joint_position_target(target)
        return
    # demo
    arm_ids = _resolve_arm_joint_indices(robot)
    if not arm_ids:
        robot.set_joint_position_target(target)
        return
    w = 2.0 * math.pi * args_cli.demo_hz
    amp = args_cli.demo_amp
    for k, jid in enumerate(arm_ids):
        phase = sim_time * w + 0.3 * k
        target[:, jid] = target[:, jid] + amp * math.sin(phase)
    robot.set_joint_position_target(target)


def main() -> None:
    init_z_used = _default_init_z()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, 2.2, 1.2], [0.0, 0.0, 0.6])

    scene_cfg = _build_scene_cfg()
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    scene.write_data_to_sim()
    for _ in range(5):
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    _print_articulation_info(robot)
    _print_z_diagnostic(robot, init_z_used)

    if args_cli.exit_after_print:
        simulation_app.close()
        return

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    while simulation_app.is_running():
        _apply_joint_control(robot, sim_time)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        sim_time += sim_dt


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app.is_running():
            simulation_app.close()
