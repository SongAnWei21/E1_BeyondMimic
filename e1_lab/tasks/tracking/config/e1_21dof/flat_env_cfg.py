from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import e1_lab.tasks.tracking.mdp as mdp

from e1_lab.robots.e1_21dof import E1_21DOF_CFG, E1_21DOF_ACTION_SCALE


from e1_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm


# 机器人速度扰动范围（用于域随机化）
# 单位: 线速度(m/s), 角速度(rad/s)
# E1_VELOCITY_RANGE = {
#     "x": (-0.5, 0.5),      # 前后方向线速度
#     "y": (-0.5, 0.5),      # 左右方向线速度
#     "z": (-0.2, 0.2),      # 上下方向线速度
#     "roll": (-0.52, 0.52), # 翻滚角速度(约30度/秒)
#     "pitch": (-0.52, 0.52),# 俯仰角速度(约30度/秒)
#     "yaw": (-0.78, 0.78),  # 偏航角速度(约45度/秒)
# }

# E1_VELOCITY_RANGE = {
#     "x": (-1.2, 1.2),      # 前后方向线速度
#     "y": (-1.2, 1.2),      # 左右方向线速度
#     "z": (-0.6, 0.6),      # 上下方向线速度
#     "roll": (-0.8, 0.8), # 翻滚角速度(约30度/秒)
#     "pitch": (-0.8, 0.8),# 俯仰角速度(约30度/秒)
#     "yaw": (-1.05, 1.05),  # 偏航角速度(约45度/秒)
# }

E1_VELOCITY_RANGE = {
    "x": (-0.3, 0.3),      # 前后方向线速度
    "y": (-0.3, 0.3),      # 左右方向线速度
    "z": (-0.1, 0.1),      # 上下方向线速度
    "roll": (-0.2, 0.2), # 翻滚角速度
    "pitch": (-0.2, 0.2),# 俯仰角速度
    "yaw": (-0.3, 0.3),  # 偏航角速度
}

@configclass
class E1_21DOF_EnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = E1_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_21DOF_ACTION_SCALE

        # 指定参与运动跟踪的关节（排除锁定的头部关节）
        self.actions.joint_pos.joint_names = [
            'left_hip_pitch_joint',
            'left_hip_roll_joint',
            'left_hip_yaw_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',

            'right_hip_pitch_joint',
            'right_hip_roll_joint',
            'right_hip_yaw_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint', 

            "waist_yaw_joint",
            
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",

        ]
        
        # motion command也需要知道参与跟踪的关节（与动作空间一致）
        self.commands.motion.joint_names = [
            'left_hip_pitch_joint',
            'left_hip_roll_joint',
            'left_hip_yaw_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',

            'right_hip_pitch_joint',
            'right_hip_roll_joint',
            'right_hip_yaw_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint', 

            "waist_yaw_joint",

            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",

        ]

        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",

            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_roll_link",

            "torso_link",

            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",

            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",

        ]
        
        # 相机设置：自由视角，不跟随机器人
        self.viewer.eye = (3.0, 3.0, 2.0)  # 相机位置
        self.viewer.lookat = (0.0, 0.0, 1.0)  # 看向位置
        self.viewer.origin_type = "world"  # 世界坐标系，不跟随机器人
        self.viewer.asset_name = None  # 不绑定到特定资产
        
        self.scene.contact_forces.debug_vis = False  # 关闭接触力可视化, 不想处理这些可视化报错，不关心界面上画出来的红红绿绿的调试线

        self.commands.motion.debug_vis = False # 关闭可视化
        self.commands.motion.velocity_range = E1_VELOCITY_RANGE # 加大随机扰动

        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",  
            "right_ankle_roll_link", 
            "left_elbow_link",      
            "right_elbow_link",
        ]

        self.rewards.motion_body_pos.weight = 1.5          
        self.rewards.motion_body_pos.params["std"] = 0.15  
        self.rewards.action_rate_l2.weight = -1.0  
        self.rewards.dof_torques_l2.weight = -1e-5    
        self.rewards.joint_acc_l2.weight = -2.5e-7 
        self.rewards.joint_limit.weight = -10.0 
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_elbow_link$)(?!right_elbow_link$).+$"
            ],
        )
        # self.rewards.feet_slide = RewTerm(
        #     func=mdp.feet_slide,
        #     weight=-0.5,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        #     },
        # )
        # self.rewards.feet_orientation_l2 = RewTerm(
        #     func=mdp.feet_orientation_l2,
        #     weight=-0.5,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        #     },
        # )

        # torso_link质心随机化
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 3),
                "dynamic_friction_range": (0.5, 3),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )
            # 间隔事件 - 传统推动（对所有环境）
        self.events.push_robot = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",                         # 间隔执行模式
                interval_range_s=(1.0, 3.0),             # 执行间隔范围(s)
                params={"velocity_range": E1_VELOCITY_RANGE}, # 推力速度范围
        )

        # 质量随机化 
        self.events.randomize_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.95, 1.05),  # 全身各个连杆质量
                "operation": "scale",
            },
        )

        # PD 参数随机化 (防止策略对当前的 Kp/Kd 过拟合)
        self.events.randomize_pd = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.85, 1.15), # Kp 浮动 15%
                "damping_distribution_params": (0.85, 1.15),   # Kd 浮动 15%
                "operation": "scale",
            },
        )

        # 关节摩擦力随机化 
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": (0.0, 0.01), # 增加 0 到 0.01 N.m 的随机干摩擦力 (根据电机量级微调)
                "operation": "add",
            },
        )

        # 转子惯量随机化 
        self.events.randomize_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.8, 1.2), # 转子惯量上下浮动 20%
                "operation": "scale",
            },
        )


        # 去除motion_anchor_pos_b和base_lin_vel
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        

