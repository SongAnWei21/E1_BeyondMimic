# from isaaclab.utils import configclass
# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import SceneEntityCfg
# import e1_lab.tasks.tracking.mdp as mdp

# from e1_lab.robots.e1_21dof import E1_21DOF_CFG, E1_21DOF_ACTION_SCALE
# from e1_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

# @configclass
# class E1_21DOF_Stage1_EnvCfg(TrackingEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # ================= 基础机器人与动作配置 =================
#         self.scene.robot = E1_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#         self.actions.joint_pos.scale = E1_21DOF_ACTION_SCALE

#         common_joints = [
#             'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
#             'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
#             "waist_yaw_joint",
#             "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
#             "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
#         ]
#         self.actions.joint_pos.joint_names = common_joints
#         self.commands.motion.joint_names = common_joints

#         self.commands.motion.anchor_body_name = "torso_link"
#         self.commands.motion.body_names = [
#             "pelvis", "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
#             "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
#             "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
#             "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
#         ]

#         # ================= 查看器与终止条件 =================
#         self.viewer.eye = (3.0, 3.0, 2.0) 
#         self.viewer.lookat = (0.0, 0.0, 1.0) 
#         self.viewer.origin_type = "world" 
#         self.viewer.asset_name = None 
#         self.scene.contact_forces.debug_vis = False 
#         self.commands.motion.debug_vis = False 

#         self.terminations.ee_body_pos.params["body_names"] = [
#             "left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link",
#         ]

#         # ================= 奖励塑形  =================
#         self.rewards.motion_body_pos.weight = 2.0          
#         self.rewards.motion_body_pos.params["std"] = 0.15  
#         self.rewards.joint_limit.weight = -2.0    # 关节限位惩罚小，后续增大
#         self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
#             "contact_forces", body_names=[r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_elbow_link$)(?!right_elbow_link$).+$"],
#         )
        
#         # 核心：重度惩罚不连贯动作，让舞姿柔和
#         self.rewards.action_rate_l2.weight = -0.2  #后续再增大      
#         self.rewards.joint_acc_l2.weight = -1e-5       
#         self.rewards.dof_torques_l2.weight = -1e-5     

#         # ================= 目标命令的随机扰动 (小扰动) =================
#         self.commands.motion.velocity_range = {
#             "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.1, 0.1),
#             "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)
#         }

#         self.commands.motion.pose_range = {
#             "x": (-0.0, 0.0),              # X方向位置偏移(m)
#             "y": (-0.0, 0.0),              # Y方向位置偏移(m)
#             "z": (-0.0, 0.0),              # Z方向位置偏移(m)
#             "roll": (-0.0, 0.0),           # 翻滚角偏移(rad)
#             "pitch": (-0.0, 0.0),          # 俯仰角偏移(rad)
#             "yaw": (-0.0, 0.0),            # 偏航角偏移(rad)
#         }
        
#         self.commands.motion.joint_position_range = (0.0, 0.0)

#         STAGE1_PUSH_VEL = {
#             "x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.2, 0.2),
#             "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4),
#         }

#         # 推力 (轻推)
#         self.events.push_robot = EventTerm(
#             func=mdp.push_by_setting_velocity,
#             mode="interval", interval_range_s=(1.0, 3.0),
#             params={"velocity_range": STAGE1_PUSH_VEL},
#         )

#         # 质量随机化 (±5%)
#         self.events.randomize_mass = EventTerm(
#             func=mdp.randomize_rigid_body_mass,
#             mode="startup",
#             params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
#         )
        
#         # 质心微调随机化
#         self.events.base_com = EventTerm(
#             func=mdp.randomize_rigid_body_com, mode="startup",
#             params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "com_range": {"x": (-0.01, 0.01), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}},
#         )

#         # 地面材质随机化 (打滑测试)
#         self.events.physics_material = EventTerm(
#             func=mdp.randomize_rigid_body_material, mode="startup",
#             params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.5, 3), "dynamic_friction_range": (0.5, 3), "restitution_range": (0.0, 0.5), "num_buckets": 64},
#         )

#         # PD参数随机化
#         self.events.randomize_pd = EventTerm(
#             func=mdp.randomize_actuator_gains, mode="startup",
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "stiffness_distribution_params": (0.85, 1.15), "damping_distribution_params": (0.85, 1.15), "operation": "scale"},
#         )

#         # 摩擦力随机化 (最大 0.02)
#         self.events.randomize_joint_friction = EventTerm(
#             func=mdp.randomize_joint_parameters,
#             mode="startup",
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "friction_distribution_params": (0.0, 0.02), "operation": "add"},
#         )

#         # 转子惯量随机化
#         self.events.randomize_joint_armature = EventTerm(
#             func=mdp.randomize_joint_parameters, mode="startup",
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "armature_distribution_params": (0.8, 1.2), "operation": "scale"},
#         )
        
#         # 裁剪无用观测
#         self.observations.policy.motion_anchor_pos_b = None
#         self.observations.policy.base_lin_vel = None



from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import e1_lab.tasks.tracking.mdp as mdp

from e1_lab.robots.e1_21dof import E1_21DOF_CFG, E1_21DOF_ACTION_SCALE
from e1_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

@configclass
class E1_21DOF_Stage1_EnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ================= 1. 基础机器人与动作配置 =================
        self.scene.robot = E1_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_21DOF_ACTION_SCALE

        common_joints = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
            "waist_yaw_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        ]
        self.actions.joint_pos.joint_names = common_joints
        self.commands.motion.joint_names = common_joints

        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis", "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
            "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
        ]

        # ================= 2. 查看器与终止条件 =================
        self.viewer.eye = (3.0, 3.0, 2.0) 
        self.viewer.lookat = (0.0, 0.0, 1.0) 
        self.viewer.origin_type = "world" 
        self.viewer.asset_name = None 
        self.scene.contact_forces.debug_vis = False 
        self.commands.motion.debug_vis = False 

        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link",
        ]

        # ================= 3. 奖励塑形 (温室期宽容模式) =================
        self.rewards.motion_body_pos.weight = 2.0          
        self.rewards.motion_body_pos.params["std"] = 0.15  
        
        # 【关键修复】大幅降低关节限位罚款，鼓励婴儿期自由探索
        self.rewards.joint_limit.weight = -0.1   
        
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_elbow_link$)(?!right_elbow_link$).+$"],
        )
        
        # 【关键修复】降低平滑惩罚，不要让 AI 不敢动
        self.rewards.action_rate_l2.weight = -0.05      
        self.rewards.joint_acc_l2.weight = -1e-5       
        self.rewards.dof_torques_l2.weight = -1e-5     

        # ================= 4. 目标命令的随机扰动 (绝对零扰动/不眼花) =================
        # 【关键修复】老老实实写 (0.0, 0.0)，防止报错，同时确保靶子绝对静止
        self.commands.motion.velocity_range = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.commands.motion.pose_range = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)

        # ================= 5. 物理域随机化 (直接物理断电，全部设为 None) =================
        # 【关键修复】关掉所有的风霜雨雪，让它安心跳舞！
        self.events.push_robot = None
        self.events.randomize_mass = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.randomize_pd = None
        self.events.randomize_joint_friction = None
        self.events.randomize_joint_armature = None
        
        # ================= 6. 裁剪无用观测 =================
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None