import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# 用于训练的环境（保持自适应采样）
gym.register(
    id="Tracking-Flat-e1_19dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.E1_19DOFFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)

gym.register(
    id="Tracking-Flat-e1_19dof-Wo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.E1_19DOFFlatWoEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)
