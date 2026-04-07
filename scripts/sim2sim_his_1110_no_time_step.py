"""Unified sim2sim script for E1 robot (10-frame Term-wise History, 1110 obs, 重力投影硬编码版)."""

import argparse
import time
import sys
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
from scipy.spatial.transform import Rotation as R

# ================= Simulation Parameters =================
simulation_duration = 300.0
simulation_dt = 0.005
control_decimation = 4

# Robot configuration (E1 21-DOF)
ROBOT_CONFIG = {
    "num_actions": 21,
    "num_obs": 1110,  # 1110维 (替换为 3D 重力投影)
    "reference_body": "torso_link",
    "joint_names": [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    ],
    "motion_body_index": 0,
}

def get_obs(data):
    """获取 MuJoCo 观测值"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    
    # 提取四元数并转为 Scipy 格式计算重力投影
    quat_mj = data.sensor("orientation").data.copy().astype(np.double)
    quat_scipy = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])
    r = R.from_quat(quat_scipy)
    
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    
    return (quat_mj, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# ================= Main Simulation =================
def run_simulation(motion_file: str, xml_path: str, policy_path: str):
    config = ROBOT_CONFIG
    print(f"[INFO]: E1 (1110 obs, 完美初态降落防炸机版, 硬编码参数无TimeStep)")

    # ── Load motion data ──
    motion = np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0],
                     motionpos.shape[0], motionquat.shape[0])

    def frame_idx(t): return t % num_frames if num_frames > 0 else 0

    # ==================== 硬编码网络参数 ====================
    joint_seq = [
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
        'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint',
        'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
        'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint',
        'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
    ]
    
    default_pos_onnx = np.array([-0.100, -0.100, 0.000, 0.000, 0.000, 0.180, 0.180, 0.000, 0.000, 0.060, 0.060, 0.200, 0.200, 0.060, 0.060, -0.100, -0.100, 0.780, 0.780, 0.000, 0.000], dtype=np.float32)
    stiffness_onnx = np.array([200.0, 200.0, 200.0, 100.0, 100.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 200.0, 200.0, 30.0, 30.0, 20.0, 20.0, 50.0, 50.0, 10.0, 10.0], dtype=np.float32)
    damping_onnx = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0], dtype=np.float32)
    action_scale_onnx = np.array([0.075, 0.075, 0.075, 0.150, 0.150, 0.180, 0.180, 0.090, 0.090, 0.180, 0.180, 0.074, 0.074, 0.117, 0.117, 0.741, 0.741, 0.180, 0.180, 0.350, 0.350], dtype=np.float32)
    # ========================================================

    joint_xml = config["joint_names"]
    idx_to_onnx = [joint_xml.index(j) for j in joint_seq]
    idx_to_xml = [joint_seq.index(j) for j in joint_xml]

    stiffness_xml = stiffness_onnx[idx_to_xml]
    damping_xml = damping_onnx[idx_to_xml]
    default_pos_xml = default_pos_onnx[idx_to_xml]
    num_actions = config["num_actions"]
    
    # ── Initialize MuJoCo ──
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # ==================== 动态获取底层安全索引 ====================
    qpos_indices, dof_indices, actuator_indices = [], [], []
    for name in joint_xml:
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id == -1: sys.exit(f"[ERROR]: XML 中找不到关节 '{name}'！")
        qpos_indices.append(m.jnt_qposadr[jnt_id])
        dof_indices.append(m.jnt_dofadr[jnt_id])
        
        act_idx = -1
        for i in range(m.nu):
            if m.actuator_trnid[i, 0] == jnt_id:
                act_idx = i; break
        if act_idx == -1: sys.exit(f"[ERROR]: 找不到电机 '{name}'！")
        actuator_indices.append(act_idx)

    # ==================== 核心修复：完美初态降落 ====================
    print("[INFO]: 正在执行安全预热 (Warm-up)...")
    mujoco.mj_resetData(m, d)
    
    initial_motion_pos = motionpos[0, config["motion_body_index"], :].copy()
    initial_motion_quat = motionquat[0, config["motion_body_index"], :].copy()
    
    first_pos_onnx = motioninputpos[0, :].copy()
    first_pos_xml = first_pos_onnx[idx_to_xml]
    
    # 强制掰成第一帧的舞姿并悬空 5 厘米防穿模(核心关键)
    for i, idx in enumerate(qpos_indices):
        d.qpos[idx] = first_pos_xml[i]
        
    d.qpos[0:2] = initial_motion_pos[0:2]
    d.qpos[2] = initial_motion_pos[2] + 0.05 
    d.qpos[3:7] = initial_motion_quat 
    
    mujoco.mj_forward(m, d)
    
    # 跑 0.5 秒让机器人稳稳落到地上
    for _ in range(100):
        q_current = np.array([d.qpos[i] for i in qpos_indices], dtype=np.float32)
        dq_current = np.array([d.qvel[i] for i in dof_indices], dtype=np.float32)
        
        tau = pd_control(first_pos_xml, q_current, stiffness_xml,
                         np.zeros_like(damping_xml), dq_current, damping_xml)
        tau = np.clip(tau, -60.0, 60.0)
        
        for i, act_idx in enumerate(actuator_indices):
            d.ctrl[act_idx] = tau[i]
            
        mujoco.mj_step(m, d)
    
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    print("[INFO]: 预热完成，机器人完美就绪！")
    # ============================================================
    
    policy = onnxruntime.InferenceSession(policy_path)

    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0

    # ==================== 初始化 1110 维历史堆叠 (投影重力) ====================
    H = 10
    hist_cmd       = np.zeros((H, 42), dtype=np.float32)
    hist_proj_grav = np.zeros((H, 3),  dtype=np.float32)  # 3维重力特征
    hist_ang_vel   = np.zeros((H, 3),  dtype=np.float32)
    hist_joint_pos = np.zeros((H, 21), dtype=np.float32)
    hist_joint_vel = np.zeros((H, 21), dtype=np.float32)
    hist_actions   = np.zeros((H, 21), dtype=np.float32)
    is_first_frame = True

    test_step_count = 0
    test_start_time = time.time()

    # ── Simulation Loop ──
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        counter = 0
        
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            quat, v, omega, gvec = get_obs(d)
            
            q_current = np.array([d.qpos[i] for i in qpos_indices], dtype=np.float32)
            dq_current = np.array([d.qvel[i] for i in dof_indices], dtype=np.float32)

            if counter % control_decimation == 0:
                idx = frame_idx(timestep)
                motioninput = np.concatenate((motioninputpos[idx, :], motioninputvel[idx, :]), axis=0)

                qpos_onnx = q_current[idx_to_onnx]
                qvel_onnx = dq_current[idx_to_onnx]

                cur_cmd        = motioninput[:42].astype(np.float32)
                cur_proj_grav  = gvec.astype(np.float32)              # 填入 3D 重力投影
                cur_ang_vel    = omega.astype(np.float32)
                cur_joint_pos  = (qpos_onnx - default_pos_onnx).astype(np.float32)
                cur_joint_vel  = qvel_onnx.astype(np.float32)
                cur_action     = action_buffer.astype(np.float32)

                if is_first_frame:
                    hist_cmd[:]        = cur_cmd
                    hist_proj_grav[:]  = cur_proj_grav
                    hist_ang_vel[:]    = cur_ang_vel
                    hist_joint_pos[:]  = cur_joint_pos
                    hist_joint_vel[:]  = cur_joint_vel
                    hist_actions[:]    = cur_action
                    is_first_frame = False
                else:
                    hist_cmd       = np.roll(hist_cmd,       -1, axis=0); hist_cmd[-1]       = cur_cmd
                    hist_proj_grav = np.roll(hist_proj_grav, -1, axis=0); hist_proj_grav[-1] = cur_proj_grav
                    hist_ang_vel   = np.roll(hist_ang_vel,   -1, axis=0); hist_ang_vel[-1]   = cur_ang_vel
                    hist_joint_pos = np.roll(hist_joint_pos, -1, axis=0); hist_joint_pos[-1] = cur_joint_pos
                    hist_joint_vel = np.roll(hist_joint_vel, -1, axis=0); hist_joint_vel[-1] = cur_joint_vel
                    hist_actions   = np.roll(hist_actions,   -1, axis=0); hist_actions[-1]   = cur_action

                # 拼接展平为 1110 维 (420 + 30 + 30 + 210 + 210 + 210)
                obs = np.concatenate([
                    hist_cmd.reshape(-1),
                    hist_proj_grav.reshape(-1),
                    hist_ang_vel.reshape(-1),
                    hist_joint_pos.reshape(-1),
                    hist_joint_vel.reshape(-1),
                    hist_actions.reshape(-1)
                ]).reshape(1, -1).astype(np.float32)

                # ==================== 彻底移除 TimeStep ====================
                feed_dict = {'obs': obs}
                
                action = policy.run(['actions'], feed_dict)[0][0]
                action_buffer = np.asarray(action).reshape(-1)

                target_pos_onnx = action_buffer * action_scale_onnx + default_pos_onnx
                target_dof_pos_xml = target_pos_onnx[idx_to_xml]
                timestep = (timestep + 1) % num_frames

            # 3. PD 控制与 NaN 拦截护盾
            tau = pd_control(target_dof_pos_xml, q_current, stiffness_xml,
                             np.zeros_like(damping_xml), dq_current, damping_xml)
            
            # 【终极拦截】：防 NaN 崩溃
            if np.any(np.isnan(tau)):
                print("\n[CRITICAL ERROR]: 神经网络输出了 NaN (计算崩溃)。仿真安全终止。")
                break
                
            tau = np.clip(tau, -60.0, 60.0)
            
            for i, act_idx in enumerate(actuator_indices):
                d.ctrl[act_idx] = tau[i]
            
            mujoco.mj_step(m, d)
            counter += 1
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0: time.sleep(time_until_next_step)

            test_step_count += 1
            if test_step_count % 500 == 0:
                elapsed = time.time() - test_start_time
                print(f"FPS - Sim: {500/elapsed:.1f} Hz | Ctrl: {(500/elapsed)/control_decimation:.1f} Hz")
                test_start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, default="motion/e1_21dof/MJ_dance.npz")
    parser.add_argument("--xml_path", type=str, default="e1_lab/assets/e1_21dof/mjcf/E1_21dof.xml")
    parser.add_argument("--policy_path", type=str, default="policy/his_1110/model_50k_action.onnx")
    args = parser.parse_args()
    run_simulation(args.motion_file, args.xml_path, args.policy_path)