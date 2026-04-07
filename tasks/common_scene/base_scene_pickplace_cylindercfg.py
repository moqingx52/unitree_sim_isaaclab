# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")
@configclass
class TableCylinderSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # 房间中心点
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"/home/gsy/work/unitree_sim_isaaclab/assets/objects/USD3_26/scene.usd",  # use simple room model
        ),
    )
    # print(f"ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
    #ISAAC_NUCLEUS_DIR: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac
    # （原 packing_table ~ packing_table_6 已移除：场景中不再加载桌子）
    # Object
    # 2. object configuration (cylinder)     
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",    # object in the scene
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.40, 0.175], # 地面放置：圆柱高 0.35，质心 z=半高（无桌子时） 
                                                  rot=[1, 0, 0, 0]), # initial rotation (rot)
        spawn=sim_utils.CylinderCfg(
            radius=0.018,    # cylinder radius (radius)
            height=0.35,     # cylinder height (height)
 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),    # rigid body properties configuration (rigid_props)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
            collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),    # visual material configuration (visual_material)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",    # friction combine mode
                restitution_combine_mode="min",    # restitution combine mode
                static_friction=1.5,    # static friction coefficient
                dynamic_friction=1.5,    # dynamic friction coefficient
                restitution=0.0,    # restitution coefficient (no restitution)
            ),
        ),
    )
    # Ground plane
    # 3. ground configuration
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",    # ground in the scene
    #     spawn=GroundPlaneCfg( ),    # ground configuration
    # )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-0.1, 3.6, 1.6),
                                                    rot_offset=( -0.00617,0.00617, 0.70708, -0.70708),
                                                    focal_length = 16.5)