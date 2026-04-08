import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robots.g1_omnipicker_cfg import G1_OMNIPICKER_CFG, replace_usd_path, with_root_z
from tasks.common_config import CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_cylindercfg import TableCylinderSceneCfg

from . import mdp


@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfg):
    robot: ArticulationCfg = with_root_z(
        replace_usd_path(
            G1_OMNIPICKER_CFG,
            "/home/gsy/work/unitree_sim_isaaclab/assets/robots/G1_omnipicker/robot.usda",
        ),
        0.0,
    )

    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.reset_object_estimate)


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceG1OmnipickerInspireEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_root_state_uniform(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                    pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
                    velocity_range={},
                    asset_cfg=SceneEntityCfg("object"),
                )
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(func=lambda env: base_mdp.reset_scene_to_default(env, torch.arange(env.num_envs, device=env.device))),
        )
