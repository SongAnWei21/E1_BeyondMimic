from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import e1_lab.tasks.tracking.mdp as mdp

from e1_lab.robots.e1_21dof import E1_21DOF_CFG, E1_21DOF_ACTION_SCALE
# 继承 Stage2，全面加码
from .env_cfg_stage2 import E1_21DOF_Stage2_EnvCfg

@configclass
class E1_21DOF_Stage3_EnvCfg(E1_21DOF_Stage2_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ================= 阶段 3：动作惩罚回调到正常水平 =================
        self.rewards.action_rate_l2.weight = -0.1 
        self.rewards.joint_acc_l2.weight = -1e-6

        # ================= 阶段 3：全火力全开的跟踪扰动 =================
        self.commands.motion.velocity_range = {
            "x": (-1.2, 1.2), "y": (-1.2, 1.2), "z": (-0.6, 0.6),
            "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-1.05, 1.05)
        }
        self.commands.motion.pose_range = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)
        }
        self.commands.motion.joint_position_range = (-0.1, 0.1)

        # ================= 阶段 3：全火力全开的域随机化 =================
        STAGE3_PUSH_VEL = {
            "x": (-1.2, 1.2), "y": (-1.2, 1.2), "z": (-0.6, 0.6),
            "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-1.05, 1.05),
        }

        # 1. 狂暴推力
        self.events.push_robot.params["velocity_range"] = STAGE3_PUSH_VEL

        # 2. 深度质量随机化 (±15%)
        self.events.randomize_mass.params["mass_distribution_params"] = (0.85, 1.15)
        
        # 3. 更大的质心偏移
        self.events.base_com.params["com_range"] = {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

        # 4. 加入地面材质随机化 (打滑测试)
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.5, 3), "dynamic_friction_range": (0.5, 3), "restitution_range": (0.0, 0.5), "num_buckets": 64},
        )

        # 5. 加入 PD 参数大盲盒
        self.events.randomize_pd = EventTerm(
            func=mdp.randomize_actuator_gains, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "stiffness_distribution_params": (0.85, 1.15), "damping_distribution_params": (0.85, 1.15), "operation": "scale"},
        )

        # 6. 加入转子惯量随机化
        self.events.randomize_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "armature_distribution_params": (0.8, 1.2), "operation": "scale"},
        )
        
        # 7. 增大摩擦力干扰
        self.events.randomize_joint_friction.params["friction_distribution_params"] = (0.0, 0.05)