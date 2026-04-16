import torch
import os

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_cylindercfg import TableCylinderSceneCfg
from tasks.g1_tasks.pick_place_cylinder_g1_29dof_inspire import mdp

project_root = os.environ.get("PROJECT_ROOT")
MICROWAVE_USD_PATH = f"{project_root}/assets/objects/microwave/7292/usd/7292.usd"
MICROWAVE_INIT_POS = (-0.08, 0.80, 0.90)
# Isaac Lab uses quaternions in (w, x, y, z). Rotate +90 deg about world x
# so the microwave's top (+z) points toward the robot's back (-y).
MICROWAVE_INIT_ROT = (0.7071, 0.7071, 0.0, 0.0)
MICROWAVE_SCALE = (0.25, 0.25, 0.25)


def _create_microwave_rigid_cfg() -> AssetBaseCfg:
    print("[MicrowaveCfg] Using rigid fallback placement.")
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/Microwave",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=MICROWAVE_INIT_POS,
            rot=MICROWAVE_INIT_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=MICROWAVE_USD_PATH,
            # Keep corrective scale for unit mismatch.
            scale=MICROWAVE_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


def _create_microwave_articulation_cfg(door_joint_names: list[str]) -> ArticulationCfg:
    print(f"[MicrowaveCfg] Using articulated microwave with joints: {door_joint_names}")
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Microwave",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MICROWAVE_INIT_POS,
            rot=MICROWAVE_INIT_ROT,
            joint_pos={joint_name: 0.0 for joint_name in door_joint_names},
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=MICROWAVE_USD_PATH,
            scale=MICROWAVE_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        actuators={
            "microwave_door": ImplicitActuatorCfg(
                joint_names_expr=door_joint_names,
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=40.0,
                damping=5.0,
            ),
        },
    )


def _build_microwave_cfg() -> AssetBaseCfg | ArticulationCfg:
    # Try articulated loading first. If USD is missing articulation metadata/joints,
    # print the reason and fallback to rigid placement automatically.
    if not os.path.exists(MICROWAVE_USD_PATH):
        print(f"[MicrowaveCfg] USD not found: {MICROWAVE_USD_PATH}")
        return _create_microwave_rigid_cfg()

    try:
        from pxr import Usd, UsdPhysics
    except Exception as exc:
        print(f"[MicrowaveCfg] pxr unavailable ({exc}); fallback to rigid placement.")
        return _create_microwave_rigid_cfg()

    try:
        stage = Usd.Stage.Open(MICROWAVE_USD_PATH)
        if stage is None:
            print("[MicrowaveCfg] Failed to open USD stage; fallback to rigid placement.")
            return _create_microwave_rigid_cfg()

        has_articulation_root = any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse())
        revolute_joint_names: list[str] = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                revolute_joint_names.append(prim.GetName())

        if not has_articulation_root:
            print("[MicrowaveCfg] USD has no ArticulationRootAPI; fallback to rigid placement.")
            return _create_microwave_rigid_cfg()
        if not revolute_joint_names:
            print("[MicrowaveCfg] USD has no RevoluteJoint; fallback to rigid placement.")
            return _create_microwave_rigid_cfg()

        return _create_microwave_articulation_cfg(revolute_joint_names)
    except Exception as exc:
        print(f"[MicrowaveCfg] Articulation probe failed ({exc}); fallback to rigid placement.")
        return _create_microwave_rigid_cfg()


@configclass
class ObjectTableMicrowaveSceneCfg(TableCylinderSceneCfg):
    """Base cylinder pick-place scene plus one microwave on main table."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()

    # Place microwave next to the cylinder object on the main table.
    # Try articulation first; automatically fallback to rigid placement.
    microwave = _build_microwave_cfg()

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
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.reset_object_estimate)


@configclass
class TerminationsNoResetCfg:
    pass


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
class PickPlaceG129InspireMicrowaveBaseFixEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableMicrowaveSceneCfg = ObjectTableMicrowaveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
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


@configclass
class PickPlaceG129InspireMicrowaveBaseFixNoResetEnvCfg(PickPlaceG129InspireMicrowaveBaseFixEnvCfg):
    terminations: TerminationsNoResetCfg = TerminationsNoResetCfg()
