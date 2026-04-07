
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from e1_lab.assets import ASSET_DIR

# 灵足各型号电机的转动惯量参数 kg*m^2
ARMATURE_RS00 = 0.001   # 14Nm  
ARMATURE_RS01 = 0.0042  
ARMATURE_RS02 = 0.0042
ARMATURE_RS03 = 0.02   # 60Nm
ARMATURE_RS04 = 0.04
ARMATURE_RS05 = 0.0007
ARMATURE_RS06 = 0.012  # 36Nm

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz的自然频率(转换为弧度)
DAMPING_RATIO = 2.0 # 阻尼比

# 根据转动惯量和自然频率计算各型号电机的刚度和阻尼
# STIFFNESS_RS00 = ARMATURE_RS00 * NATURAL_FREQ**2  # 3.9478417602100686
# DAMPING_RS00 = 2.0 * DAMPING_RATIO * ARMATURE_RS00 * NATURAL_FREQ # 0.25132741228
# STIFFNESS_RS03 = ARMATURE_RS03 * NATURAL_FREQ**2  # 78.95683520420137
# DAMPING_RS03 = 2.0 * DAMPING_RATIO * ARMATURE_RS03 * NATURAL_FREQ # 5.0265482456
# STIFFNESS_RS06 = ARMATURE_RS06 * NATURAL_FREQ**2 # 47.37410112252082
# DAMPING_RS06 = 2.0 * DAMPING_RATIO * ARMATURE_RS06 * NATURAL_FREQ # 3.01592894736

# print(STIFFNESS_RS00)
# print(STIFFNESS_RS03)
# print(STIFFNESS_RS06)
# print(DAMPING_RS00)
# print(DAMPING_RS03)
# print(DAMPING_RS06)

STIFFNESS_RS00 = 20 
DAMPING_RS00 = 1
STIFFNESS_RS03 = 100 
DAMPING_RS03 = 5
STIFFNESS_RS06 = 50 
DAMPING_RS06 = 3

EFFORT_LIMIT_RS00 = 14
VELOCITY_LIMIT_RS00 = 32.99
EFFORT_LIMIT_RS03 = 60
VELOCITY_LIMIT_RS03 = 20.42
EFFORT_LIMIT_RS06 = 36
VELOCITY_LIMIT_RS06 = 50.27
EFFORT_LIMIT_RS06_17_28 = 59.3  # RS06同步带传动减速比17/28 
VELOCITY_LIMIT_RS06_17_28 = 30.52 # RS06同步带传动减速比17/28 


# E1_19DOF机器人模型配置
E1_19DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/e1/urdf/E1_19dof.urdf", 
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint":-0.3,
            ".*_hip_roll_joint":0.0,
            ".*_hip_yaw_joint":0.0,
            ".*_knee_joint":0.6,
            ".*_ankle_pitch_joint":-0.3,
            ".*_ankle_roll_joint":0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": EFFORT_LIMIT_RS03,
                ".*_hip_roll_joint": EFFORT_LIMIT_RS03,
                ".*_hip_yaw_joint": EFFORT_LIMIT_RS06,
                ".*_knee_joint": EFFORT_LIMIT_RS06_17_28,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": VELOCITY_LIMIT_RS03,
                ".*_hip_roll_joint": VELOCITY_LIMIT_RS03,
                ".*_hip_yaw_joint": VELOCITY_LIMIT_RS06,
                ".*_knee_joint": VELOCITY_LIMIT_RS06_17_28,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_RS03,
                ".*_hip_roll_joint": STIFFNESS_RS03,
                ".*_hip_yaw_joint": STIFFNESS_RS06,
                ".*_knee_joint": STIFFNESS_RS06,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_RS03,
                ".*_hip_roll_joint": DAMPING_RS03,
                ".*_hip_yaw_joint": DAMPING_RS06,
                ".*_knee_joint": DAMPING_RS06,
            },
            armature={
                ".*_hip_pitch_joint":  ARMATURE_RS03,
                ".*_hip_roll_joint": ARMATURE_RS03,
                ".*_hip_yaw_joint": ARMATURE_RS06,
                ".*_knee_joint": ARMATURE_RS06,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint", 
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": EFFORT_LIMIT_RS06_17_28,
                ".*_ankle_roll_joint": EFFORT_LIMIT_RS00,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": VELOCITY_LIMIT_RS06_17_28,
                ".*_ankle_roll_joint": VELOCITY_LIMIT_RS00,
            },
            stiffness={
                ".*_ankle_pitch_joint": STIFFNESS_RS06,
                ".*_ankle_roll_joint": STIFFNESS_RS00,
            },
            damping={
                ".*_ankle_pitch_joint": DAMPING_RS06, 
                ".*_ankle_roll_joint": DAMPING_RS00,
            },
            armature={
                ".*_ankle_pitch_joint": ARMATURE_RS06,
                ".*_ankle_roll_joint": ARMATURE_RS00,
            },
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=EFFORT_LIMIT_RS03,
            velocity_limit_sim=VELOCITY_LIMIT_RS03,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_RS03,
            damping=DAMPING_RS03, 
            armature=ARMATURE_RS03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",

            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": EFFORT_LIMIT_RS06,
                ".*_shoulder_roll_joint": EFFORT_LIMIT_RS06,
                ".*_shoulder_yaw_joint": EFFORT_LIMIT_RS00,

            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": VELOCITY_LIMIT_RS06,
                ".*_shoulder_roll_joint": VELOCITY_LIMIT_RS06,
                ".*_shoulder_yaw_joint": VELOCITY_LIMIT_RS00,

            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_RS06,
                ".*_shoulder_roll_joint": STIFFNESS_RS06,
                ".*_shoulder_yaw_joint": STIFFNESS_RS00,

            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_RS06,
                ".*_shoulder_roll_joint": DAMPING_RS06,
                ".*_shoulder_yaw_joint": DAMPING_RS00,

            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_RS06,
                ".*_shoulder_roll_joint": ARMATURE_RS06,
                ".*_shoulder_yaw_joint": ARMATURE_RS00,

            },
        ),
    },
)

E1_19DOF_ACTION_SCALE = {}
for a in E1_19DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            E1_19DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
