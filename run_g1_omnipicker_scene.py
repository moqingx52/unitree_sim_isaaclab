#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""加载 g1_omnipicker 机器人和指定场景 USD 的最小运行脚本。

默认值已按当前验证结果写死：
- robot_usd: /home/gsy/work/unitree_sim_isaaclab/assets/robots/G1_omnipicker/robot.usda
- scene_usd: /home/gsy/work/unitree_sim_isaaclab/assets/objects/USD3_26/scene.usd
- init_z: 0.0
"""

from __future__ import annotations

import argparse
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run G1 Omnipicker with USD scene.")
parser.add_argument("--robot_usd", type=str, default="/home/gsy/work/unitree_sim_isaaclab/assets/robots/G1_omnipicker/robot.usda")
parser.add_argument("--scene_usd", type=str, default="/home/gsy/work/unitree_sim_isaaclab/assets/objects/USD3_26/scene.usd")
parser.add_argument("--init_z", type=float, default=0.0, help="Robot root init z.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Env spacing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from robots.g1_omnipicker_cfg import G1_OMNIPICKER_CFG, replace_usd_path, with_root_z


def _build_robot_cfg() -> ArticulationCfg:
    cfg = replace_usd_path(G1_OMNIPICKER_CFG, args_cli.robot_usd)
    return with_root_z(cfg, args_cli.init_z)


@configclass
class G1OmnipickerSceneCfg(InteractiveSceneCfg):
    room_scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/RoomScene",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.scene_usd,
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot: ArticulationCfg = _build_robot_cfg()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.0, 1.6], [0.0, 0.0, 0.6])

    scene_cfg = G1OmnipickerSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=True,
    )
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]

    sim.reset()
    scene.reset()

    print("=" * 72)
    print("G1 Omnipicker scene loaded")
    print(f"robot_usd: {args_cli.robot_usd}")
    print(f"scene_usd: {args_cli.scene_usd}")
    print(f"init_z:    {args_cli.init_z:.4f}")
    print(f"joint_cnt: {len(robot.data.joint_names)}")
    print(f"body_cnt:  {len(robot.data.body_names)}")
    print("=" * 72)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
