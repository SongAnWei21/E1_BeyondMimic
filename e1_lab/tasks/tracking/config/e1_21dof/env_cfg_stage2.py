from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import e1_lab.tasks.tracking.mdp as mdp

from e1_lab.robots.e1_21dof import E1_21DOF_CFG, E1_21DOF_ACTION_SCALE
# 直接继承 Stage1，只覆盖需要改的部分
from .env_cfg_stage1 import E1_21DOF_Stage1_EnvCfg

@configclass
class E1_21DOF_Stage2_EnvCfg(E1_21DOF_Stage1_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ================= 阶段 2：稍微放宽动作惩罚 =================
        self.rewards.action_rate_l2.weight = -0.5 

        # ================= 阶段 2：恢复轻微的目标目标扰动 (微弱眼花) =================
        self.commands.motion.velocity_range = {
            "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.1, 0.1),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)
        }
        self.commands.motion.pose_range = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.1, 0.1)
        }

        # ================= 阶段 2：引入轻度域随机化与推力 =================
        STAGE2_PUSH_VEL = {
            "x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.2, 0.2),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4),
        }

        # 1. 恢复推力 (轻推)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval", interval_range_s=(1.0, 3.0),
            params={"velocity_range": STAGE2_PUSH_VEL},
        )

        # 2. 轻度质量随机化 (±5%)
        self.events.randomize_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
        )

        # 3. 轻度摩擦力随机化 (最大 0.02)
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "friction_distribution_params": (0.0, 0.02), "operation": "add"},
        )
        
        # 4. 恢复质心微调
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "com_range": {"x": (-0.01, 0.01), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}},
        )