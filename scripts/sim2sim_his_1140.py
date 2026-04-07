"""Unified sim2sim script for E1 robot (10-frame Term-wise History, 1140 obs, 完美防炸机版)."""

import argparse
import time
import sys
import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
from scipy.spatial.transform import Rotation as R

# ================= Simulation Parameters =================
simulation_duration = 300.0
simulation_dt = 0.005
control_decimation = 4

# Robot configuration (E1 21-DOF)
ROBOT_CONFIG = {
    "num_actions": 21,
    "num_obs": 1140,
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

def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_obs(data):
    """获取 MuJoCo 观测值"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    
    quat_mj = data.sensor("orientation").data.copy().astype(np.double)
    quat_scipy = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])
    r = R.from_quat(quat_scipy)
    
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    
    return (quat_mj, v, omega, gvec)

def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    conj = np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1)
    return conj / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# ================= Main Simulation =================
def run_simulation(motion_file: str, xml_path: str, policy_path: str):
    config = ROBOT_CONFIG
    print(f"[INFO]: E1 (1140 obs, 完美初态降落防炸机版)")

    # ── Load motion data ──
    motion = np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0],
                     motionpos.shape[0], motionquat.shape[0])

    def frame_idx(t): return t % num_frames if num_frames > 0 else 0

    # ── Load ONNX and metadata ──
    model = onnx.load(policy_path)
    for prop in model.metadata_props:
        if prop.key == "joint_names": joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos": default_pos_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)
        elif prop.key == "joint_stiffness": stiffness_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)
        elif prop.key == "joint_damping": damping_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)
        elif prop.key == "action_scale": action_scale_onnx = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)

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
    
    # 1. 提取动捕第 0 帧的所有精确姿态
    initial_motion_pos = motionpos[0, config["motion_body_index"], :].copy()
    initial_motion_quat = motionquat[0, config["motion_body_index"], :].copy()
    
    # 将手脚的初始角度，转化为第 0 帧的真实舞蹈动作 (XML顺序)
    first_pos_onnx = motioninputpos[0, :].copy()
    first_pos_xml = first_pos_onnx[idx_to_xml]
    
    # 2. 将机器人的每个关节都强行掰成第一帧的舞姿！
    for i, idx in enumerate(qpos_indices):
        d.qpos[idx] = first_pos_xml[i]
        
    # 3. 把摆好姿势的机器人放在离地 5 厘米的地方
    d.qpos[0:2] = initial_motion_pos[0:2]
    d.qpos[2] = initial_motion_pos[2] + 0.05 
    d.qpos[3:7] = initial_motion_quat 
    
    mujoco.mj_forward(m, d)
    
    # 4. 让机器人用第一帧的舞姿，平稳落到地上 (0.5秒)
    for _ in range(100):
        q_current = np.array([d.qpos[i] for i in qpos_indices], dtype=np.float32)
        dq_current = np.array([d.qvel[i] for i in dof_indices], dtype=np.float32)
        
        # 注意！这里跟踪的是 first_pos_xml，而不是 default_pos！这样落地才不会摔倒！
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
    has_time_step = 'time_step' in [inp.name for inp in policy.get_inputs()]

    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0
    motion_body_idx = config["motion_body_index"]

    # ==================== 恢复正确的 1140 维历史堆叠 (ref_ori) ====================
    H = 10
    hist_cmd       = np.zeros((H, 42), dtype=np.float32)
    hist_ref_ori   = np.zeros((H, 6),  dtype=np.float32)  # 6维朝向特征
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
                motionquatcurrent = motionquat[idx, motion_body_idx, :]

                # 计算 6D 相对朝向
                q01 = quat
                q02 = motionquatcurrent
                q10 = quat_inv_np(q01)
                q12 = quat_mul_np(q10, q02) if q02 is not None else q10
                mat = matrix_from_quat(torch.from_numpy(q12).unsqueeze(0))
                motion_ref_ori_b = mat[0, :, :2].reshape(6).numpy().astype(np.float32)

                qpos_onnx = q_current[idx_to_onnx]
                qvel_onnx = dq_current[idx_to_onnx]

                cur_cmd        = motioninput[:42].astype(np.float32)
                cur_ref_ori    = motion_ref_ori_b.astype(np.float32)  # 填入 6D 特征
                cur_ang_vel    = omega.astype(np.float32)
                cur_joint_pos  = (qpos_onnx - default_pos_onnx).astype(np.float32)
                cur_joint_vel  = qvel_onnx.astype(np.float32)
                cur_action     = action_buffer.astype(np.float32)

                if is_first_frame:
                    hist_cmd[:]        = cur_cmd
                    hist_ref_ori[:]    = cur_ref_ori
                    hist_ang_vel[:]    = cur_ang_vel
                    hist_joint_pos[:]  = cur_joint_pos
                    hist_joint_vel[:]  = cur_joint_vel
                    hist_actions[:]    = cur_action
                    is_first_frame = False
                else:
                    hist_cmd       = np.roll(hist_cmd,       -1, axis=0); hist_cmd[-1]       = cur_cmd
                    hist_ref_ori   = np.roll(hist_ref_ori,   -1, axis=0); hist_ref_ori[-1]   = cur_ref_ori
                    hist_ang_vel   = np.roll(hist_ang_vel,   -1, axis=0); hist_ang_vel[-1]   = cur_ang_vel
                    hist_joint_pos = np.roll(hist_joint_pos, -1, axis=0); hist_joint_pos[-1] = cur_joint_pos
                    hist_joint_vel = np.roll(hist_joint_vel, -1, axis=0); hist_joint_vel[-1] = cur_joint_vel
                    hist_actions   = np.roll(hist_actions,   -1, axis=0); hist_actions[-1]   = cur_action

                obs = np.concatenate([
                    hist_cmd.reshape(-1),         # 420
                    hist_ref_ori.reshape(-1),     # 60
                    hist_ang_vel.reshape(-1),     # 30
                    hist_joint_pos.reshape(-1),   # 210
                    hist_joint_vel.reshape(-1),   # 210
                    hist_actions.reshape(-1)      # 210
                ]).reshape(1, -1).astype(np.float32)

                feed_dict = {'obs': obs}
                if has_time_step: feed_dict['time_step'] = np.array([[idx]], dtype=np.float32)
                
                action = policy.run(['actions'], feed_dict)[0][0]
                action_buffer = np.asarray(action).reshape(-1)

                target_pos_onnx = action_buffer * action_scale_onnx + default_pos_onnx
                target_dof_pos_xml = target_pos_onnx[idx_to_xml]
                timestep = (timestep + 1) % num_frames

            # 3. PD 控制与 NaN 拦截护盾
            tau = pd_control(target_dof_pos_xml, q_current, stiffness_xml,
                             np.zeros_like(damping_xml), dq_current, damping_xml)
            
            # 【终极拦截】：如果有任何 NaN 产生，立刻停机报错，阻止 MuJoCo 闪退
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
    parser.add_argument("--policy_path", type=str, default="policy/his_1140/model_50k.onnx")
    args = parser.parse_args()
    run_simulation(args.motion_file, args.xml_path, args.policy_path)