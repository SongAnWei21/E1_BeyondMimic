from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import e1_lab.tasks.tracking.mdp as mdp

from e1_lab.robots.e1_19dof import E1_19DOF_CFG, E1_19DOF_ACTION_SCALE


from e1_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm


# 机器人速度扰动范围（用于域随机化）
# 单位: 线速度(m/s), 角速度(rad/s)
# E1_VELOCITY_RANGE = {
#     "x": (-1.0, 1.0),      # 前后方向线速度
#     "y": (-1.0, 1.0),      # 左右方向线速度
#     "z": (-0.5, 0.5),      # 上下方向线速度
#     "roll": (-0.82, 0.82), # 翻滚角速度(约30度/秒)
#     "pitch": (-0.82, 0.82),# 俯仰角速度(约30度/秒)
#     "yaw": (-0.98, 0.98),  # 偏航角速度(约45度/秒)
# }

E1_19DOF_VELOCITY_RANGE = {
    "x": (-1.5, 1.5),      # 前后方向线速度
    "y": (-1.5, 1.5),      # 左右方向线速度
    "z": (-0.8, 0.8),      # 上下方向线速度
    "roll": (-1.0, 1.0), # 翻滚角速度(约30度/秒)
    "pitch": (-1.0, 1.0),# 俯仰角速度(约30度/秒)
    "yaw": (-1.2, 1.2),  # 偏航角速度(约45度/秒)
}


@configclass
class E1_19DOFFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = E1_19DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_19DOF_ACTION_SCALE

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

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",

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

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
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
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
        ]
        
        # 相机设置：自由视角，不跟随机器人
        self.viewer.eye = (3.0, 3.0, 2.0)  # 相机位置
        self.viewer.lookat = (0.0, 0.0, 1.0)  # 看向位置
        self.viewer.origin_type = "world"  # 世界坐标系，不跟随机器人
        self.viewer.asset_name = None  # 不绑定到特定资产
        
        # 关闭调试可视化显示
        # self.commands.motion.debug_vis = False  # 关闭motion命令的调试可视化

        self.scene.contact_forces.debug_vis = False  # 关闭接触力可视化, 不想处理这些可视化报错，或者你不关心界面上画出来的红红绿绿的调试线

        self.commands.motion.debug_vis = False # 关闭可视化

        self.commands.motion.velocity_range = E1_19DOF_VELOCITY_RANGE # 加大随机扰动

        # 保持训练时的默认设置（自适应采样等）\
        self.rewards.motion_body_pos.weight = 1.5  # 从1.0增加到1.5
        self.rewards.motion_body_pos.params["std"] = 0.15  # 从1.0增加到1.5

        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",  # HI机器人的脚踝
            "right_ankle_roll_link", 
            "left_shoulder_yaw_link",      # HI机器人的手腕（注意名称不同）
            "right_shoulder_yaw_link",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_shoulder_yaw_link$)(?!right_shoulder_yaw_link$).+$"
            ],
        )
        # 如需演示模式，请使用 Tracking-Flat-PI-Plus-Play-v0
        
        # 修复base_com事件配置，使用base_link而不是torso_link
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

@configclass
class E1_19DOFFlatWoEnvCfg(E1_19DOFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
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

