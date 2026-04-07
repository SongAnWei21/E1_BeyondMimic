import numpy as np
import os

def trim_npz(input_path, output_path, start_frame, end_frame):
    """
    input_path: 原始 npz 文件路径
    output_path: 截取后的保存路径
    start_frame: 开始帧索引 (从 0 开始)
    end_frame: 结束帧索引 (不包含该帧，类似 Python slice)
    """
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    data = np.load(input_path)
    trimmed_dict = {}

    print(f"--- 正在处理: {os.path.basename(input_path)} ---")
    print(f"截取范围: [{start_frame} : {end_frame}]")

    for key in data.files:
        array = data[key]
        
        # 检查是否为可裁剪的数组（通常第一维是帧数）
        if len(array.shape) > 0:
            # 执行切片
            current_trimmed = array[start_frame:end_frame]
            trimmed_dict[key] = current_trimmed
            print(f"Key: {key:15} | 原始形状: {str(array.shape):15} | 裁剪后: {current_trimmed.shape}")
        else:
            # 如果是标量或空数组，直接保留
            trimmed_dict[key] = array

    # 使用 savez_compressed 节省空间
    np.savez_compressed(output_path, **trimmed_dict)
    print(f"--- 处理完成！保存至: {output_path} ---")

# --- 使用示例 ---
# 比如截取 200 到 500 帧
trim_npz(
    input_path='/home/saw/droidup/E1_BeyondMimic/motion/e1_21dof/MJ_dance.npz', 
    output_path='/home/saw/droidup/E1_BeyondMimic/motion/e1_21dof/MJ_dance_100.npz', 
    start_frame=200, 
    end_frame=500
)