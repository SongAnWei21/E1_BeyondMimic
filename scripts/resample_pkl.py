import argparse
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def interpolate_linear(old_times, old_data, new_times):
    """对普通的多维数组（位置、速度）进行线性插值"""
    if old_data is None:
        return None
    # axis=0 表示沿着时间帧的维度进行插值
    interpolator = interp1d(old_times, old_data, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(new_times).astype(np.float32)

def interpolate_quaternion(old_times, old_quats_xyzw, new_times):
    """对四元数进行球面线性插值 (Slerp)"""
    if old_quats_xyzw is None:
        return None
    
    # 确保四元数是规范化的
    norms = np.linalg.norm(old_quats_xyzw, axis=1, keepdims=True)
    old_quats_xyzw = old_quats_xyzw / np.clip(norms, 1e-10, None)

    # Scipy 的 Rotation 默认接受的就是 [x, y, z, w] 格式
    rotations = R.from_quat(old_quats_xyzw)
    
    # 构建 Slerp 插值器
    slerp = Slerp(old_times, rotations)
    
    # 生成新时间点的旋转，并转换回 xyzw 格式的 numpy 数组
    new_rotations = slerp(new_times)
    return new_rotations.as_quat().astype(np.float32)

def resample_pkl(input_file, output_file, target_fps, input_fps_override=None):
    print(f"Loading PKL: {input_file}")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # ================= 核心修改：输入帧率判定逻辑 =================
    if input_fps_override is not None:
        original_fps = float(input_fps_override)
        print(f"[INFO] Using manually specified input FPS: {original_fps} Hz")
    else:
        original_fps = data.get('fps', None)
        if original_fps is None:
            raise ValueError("[ERROR] No 'fps' key found in PKL and --input_fps was not provided. Please specify --input_fps!")
        else:
            print(f"[INFO] Auto-detected input FPS from PKL metadata: {original_fps} Hz")
    # ==============================================================

    num_frames = data['root_pos'].shape[0]
    duration = (num_frames - 1) / original_fps

    print(f"Original: {num_frames} frames @ {original_fps} Hz (Duration: {duration:.3f}s)")
    
    # 允许有一定的浮点数误差
    if abs(original_fps - target_fps) < 1e-3:
        print("Target FPS is same as Original FPS. Just copying the file...")
        # 直接存一份副本即可
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved exact copy to: {output_file}")
        return

    # 构建旧的时间轴和新的时间轴
    old_times = np.linspace(0, duration, num_frames)
    new_num_frames = int(duration * target_fps) + 1
    new_times = np.linspace(0, duration, new_num_frames)

    print(f"Target: {new_num_frames} frames @ {target_fps} Hz")

    # 创建新的数据字典
    new_data = data.copy()
    new_data['fps'] = float(target_fps)

    print("Interpolating linear data (positions & velocities)...")
    # 对所有线性数据进行插值
    linear_keys = ['root_pos', 'dof_pos', 'root_vel', 'root_vel_body', 'root_rot_vel', 'dof_vel']
    for key in linear_keys:
        if key in data and data[key] is not None:
            new_data[key] = interpolate_linear(old_times, data[key], new_times)

    print("Interpolating quaternions (Slerp)...")
    # 对四元数进行球面线性插值
    if 'root_rot' in data:
        new_data['root_rot'] = interpolate_quaternion(old_times, data['root_rot'], new_times)

    # 修正元数据
    if 'meta' in new_data:
        if 'frame_dt_per_step' in new_data['meta']:
            new_data['meta']['frame_dt_per_step'] = np.full(new_num_frames, 1.0 / target_fps, dtype=np.float32)

    # 保存文件
    print(f"Saving to PKL: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(new_data, f)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change the FPS of a motion PKL file.")
    parser.add_argument("--input", type=str, required=True, help="Input PKL file path")
    parser.add_argument("--output", type=str, required=True, help="Output PKL file path")
    parser.add_argument("--output_fps", type=float, required=True, help="Target output FPS (e.g., 50)")
    
    # 新增的输入帧率参数（可选）
    parser.add_argument("--input_fps", type=float, default=None, 
                        help="Override the input FPS. If not provided, it will try to read from the PKL file.")
    
    args = parser.parse_args()

    resample_pkl(args.input, args.output, args.output_fps, args.input_fps)