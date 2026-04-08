import gymnasium as gym

from . import pickplace_cylinder_g1_omnipicker_inspire_env_cfg


gym.register(
    id="Isaac-PickPlace-Cylinder-G1-Omnipicker-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_cylinder_g1_omnipicker_inspire_env_cfg.PickPlaceG1OmnipickerInspireEnvCfg,
    },
    disable_env_checker=True,
)
