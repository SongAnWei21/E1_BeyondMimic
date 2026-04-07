import gymnasium as gym

from . import agents, flat_env_cfg
from . import env_cfg_stage1, env_cfg_stage2, env_cfg_stage3
from . import env_cfg,env_cfg_nohis

##
# Register Gym environments.
##

gym.register(
    id="e1_21dof_bm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.E1_21DOF_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)

gym.register(
    id="e1_21dof",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.E1_21DOF_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)


gym.register(
    id="e1_21dof_nohis",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_nohis.E1_21DOF_NOHIS_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)

# 阶段1  
gym.register(
    id="e1_21dof_bm_stage1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_stage1.E1_21DOF_Stage1_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)

# 阶段2
gym.register(
    id="e1_21dof_bm_stage2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_stage2.E1_21DOF_Stage2_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)

# 阶段3
gym.register(
    id="e1_21dof_bm_stage3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_stage3.E1_21DOF_Stage3_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1FlatPPORunnerCfg",
    }, 
)

