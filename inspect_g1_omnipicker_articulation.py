#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""单独加载 G1 Omnipicker（最小 ArticulationCfg），打印 Isaac Lab 识别的关节与刚体名。

用法（在 unitree_sim_isaaclab 目录、已激活 Isaac Lab / Isaac Sim 环境）::

    python inspect_g1_omnipicker_articulation.py --device cuda

服务器上若资产在固定路径，可显式指定 USD::

    python inspect_g1_omnipicker_articulation.py --device cuda \\
        --usd_path /home/gsy/work/unitree_sim_isaaclab/assets/robots/G1_omnipicker/robot.usd

仅打印后退出（适合无界面/自动化）::

    python inspect_g1_omnipicker_articulation.py --device cuda --headless --exit_after_print
"""

from __future__ import annotations

import argparse
import os

# 与 sim_main 一致，供 robots/*.py 内路径拼接
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect G1 Omnipicker articulation (spawn + joint/body names)")
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="覆盖默认 robot.usd 路径（默认: $PROJECT_ROOT/assets/robots/G1_omnipicker/robot.usd）",
)
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
parser.add_argument("--env_spacing", type=float, default=2.0, help="环境网格间距")
parser.add_argument(
    "--no_ground",
    action="store_true",
    help="不生成地面（默认生成默认地面，便于站立）",
)
parser.add_argument(
    "--exit_after_print",
    action="store_true",
    help="打印关节/刚体名后立刻退出（建议配合 --headless）",
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

from robots.g1_omnipicker_minimal import G1_OMNIPICKER_MINIMAL_CFG, replace_usd_path


def _robot_cfg() -> ArticulationCfg:
    if args_cli.usd_path:
        return replace_usd_path(G1_OMNIPICKER_MINIMAL_CFG, os.path.abspath(args_cli.usd_path))
    return G1_OMNIPICKER_MINIMAL_CFG


def _build_scene_cfg() -> InteractiveSceneCfg:
    """按 CLI 组装 InteractiveSceneCfg（地面可选）。"""
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
    print("G1 Omnipicker — Isaac Lab Articulation 解析结果")
    print("=" * 72)
    default_usd = G1_OMNIPICKER_MINIMAL_CFG.spawn.usd_path
    print(f"USD (配置内默认): {default_usd}")
    if args_cli.usd_path:
        print(f"USD (命令行覆盖): {os.path.abspath(args_cli.usd_path)}")
    print()

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
        print(f"find_joints('.*', preserve_order=True) -> {len(j_found)} 个: {list(j_found)}")
    except TypeError:
        _j_idx, j_found = robot.find_joints(".*")
        print(f"find_joints('.*') -> {len(j_found)} 个: {list(j_found)}")
    except Exception as e:
        print(f"find_joints 调用异常（可忽略）: {e}")

    try:
        _b_idx, b_found = robot.find_bodies(".*")
        print(f"find_bodies('.*') -> {len(b_found)} 个: {list(b_found)}")
    except Exception as e:
        print(f"find_bodies 调用异常（可忽略）: {e}")

    print("=" * 72)


def main() -> None:
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

    if args_cli.exit_after_print:
        simulation_app.close()
        return

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app.is_running():
            simulation_app.close()
