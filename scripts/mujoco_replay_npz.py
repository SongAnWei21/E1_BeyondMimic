import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
import sys

# ================= 配置区域 =================

# 1. XML 中的关节顺序 (MuJoCo Physical DFS Order)
# MuJoCo 解析 XML 通常是深度优先：左腿 -> 右腿 -> 腰 -> 左臂 -> 右臂
XML_JOINT_NAMES = [
    # Left Leg (6)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    
    # Right Leg (6)
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    
    # Waist (1)
    'waist_yaw_joint',
    
    # Left Arm (4)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
    'left_elbow_joint',
    
    # Right Arm (4)
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
    'right_elbow_joint'
]

# 2. NPZ 文件中的关节顺序 (IsaacLab Training Order)
LOG_JOINT_ORDER = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint',
    'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint',
    'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
]

# ===========================================

def play_motion(xml_path, motion_file):
    print(f"Loading XML: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print(f"Loading Motion: {motion_file}")
    motion = np.load(motion_file)
    
    # 读取关节数据
    if 'joint_pos' in motion:
        npz_qpos = motion['joint_pos'] # Shape: (Frames, Num_Joints)
    elif 'qpos' in motion:
        npz_qpos = motion['qpos']
    else:
        print("Error: Cannot find 'joint_pos' or 'qpos' in NPZ file.")
        print(f"Available keys: {list(motion.keys())}")
        return

    # 读取 Root 位置/姿态
    npz_root_pos = motion.get('body_pos_w', None)
    npz_root_quat = motion.get('body_quat_w', None)

    # 检查维度
    num_frames = npz_qpos.shape[0]
    num_joints_npz = npz_qpos.shape[1]
    num_joints_xml = len(XML_JOINT_NAMES)

    print(f"Frames: {num_frames}, NPZ Joints: {num_joints_npz}, XML Joints: {num_joints_xml}")

    if num_joints_npz != num_joints_xml:
        print(f"[WARNING] Joint count mismatch! NPZ has {num_joints_npz}, but script expects {num_joints_xml}.")

    # ---------------------------------------------------------
    # 核心逻辑：建立映射索引
    # XML[i] 应该读取 NPZ[ map_indices[i] ]
    # ---------------------------------------------------------
    map_indices = []
    
    print("\n[MAPPING] Mapping XML joints to NPZ columns based on Log Order...")
    for i, xml_name in enumerate(XML_JOINT_NAMES):
        try:
            # 在 Log 顺序中找到 xml_name 对应的索引
            npz_idx = LOG_JOINT_ORDER.index(xml_name)
            map_indices.append(npz_idx)
            print(f"  XML[{i:02d}] {xml_name:<30} <--- NPZ[{npz_idx:02d}]")
        except ValueError:
            print(f"  [ERROR] XML Joint '{xml_name}' not found in NPZ Joint List!")
            return

    # ---------------------------------------------------------

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("\nStarting Playback...")
        frame_idx = 0
        
        while viewer.is_running():
            step_start = time.time()

            # --- 实时打印帧数与进度 ---
            progress = (frame_idx + 1) / num_frames * 100
            # 使用 \r 回车符和 flush=True 实现原地流畅刷新
            print(f"\r[Playback] Frame: {frame_idx + 1:04d} / {num_frames} | Progress: {progress:6.2f}%", end="", flush=True)
            # --------------------------

            # 1. 设置 Root (Base)
            # MuJoCo qpos[0:3] = pos, qpos[3:7] = quat (w,x,y,z)
            if npz_root_pos is not None:
                if npz_root_pos.ndim == 3: 
                    d.qpos[0:3] = npz_root_pos[frame_idx, 0, :]
                else:
                    d.qpos[0:3] = npz_root_pos[frame_idx]

            if npz_root_quat is not None:
                if npz_root_quat.ndim == 3:
                    d.qpos[3:7] = npz_root_quat[frame_idx, 0, :]
                else:
                    d.qpos[3:7] = npz_root_quat[frame_idx]
            
            # 2. 设置 关节角度 (使用映射)
            current_frame_joints = npz_qpos[frame_idx] 
            
            # 根据映射抓取正确的顺序
            remapped_joints = current_frame_joints[map_indices]
            
            # 填入 MuJoCo (浮动基座占了前 7 个位置，所以从 index 7 开始)
            d.qpos[7 : 7+num_joints_xml] = remapped_joints

            # 3. 刷新运动学 (不进行物理时间步进，仅刷新画面)
            mujoco.mj_forward(m, d)

            # 4. 更新画面
            viewer.sync()

            # 循环播放
            frame_idx = (frame_idx + 1) % num_frames
            
            # 循环结束时换行，防止终端字符重叠
            if frame_idx == 0:
                print("\n[Info] Loop finished, restarting...")
            
            # 控制播放速度 (0.02s per frame for 50Hz)
            # 为了抵消代码执行的时间，可以让 sleep 时间更精准
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="e1_lab/assets/e1_21dof/mjcf/E1_21dof.xml")
    parser.add_argument("--npz", type=str, default="motion/e1_21dof/MJ_dance.npz")
    args = parser.parse_args()

    play_motion(args.xml, args.npz)