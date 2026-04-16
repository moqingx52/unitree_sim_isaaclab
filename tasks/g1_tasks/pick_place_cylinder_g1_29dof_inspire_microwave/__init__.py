import gymnasium as gym

from . import pickplace_cylinder_g1_29dof_inspire_microwave_env_cfg


gym.register(
    id="Isaac-PickPlace-Cylinder-G129-Inspire-Microwave-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_cylinder_g1_29dof_inspire_microwave_env_cfg.PickPlaceG129InspireMicrowaveBaseFixEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cylinder-G129-Inspire-Microwave-Joint-NoReset",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_cylinder_g1_29dof_inspire_microwave_env_cfg.PickPlaceG129InspireMicrowaveBaseFixNoResetEnvCfg,
    },
    disable_env_checker=True,
)
