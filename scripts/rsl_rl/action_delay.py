import torch
import gymnasium as gym
from collections import deque


class ActionDelayWrapper(gym.Wrapper):
    """
    动作延迟包装器 (支持向量化环境)
    """
    def __init__(self, env, delay_steps=1):
        """
        :param env: IsaacLab 向量化环境
        :param delay_steps: 延迟的策略步数。
                            如果 decimation=4 (50Hz), dt=0.02s。
                            delay_steps=1 意味着 20ms 的延迟；
                            delay_steps=2 意味着 40ms 的延迟。
        """
        super().__init__(env)
        self.delay_steps = delay_steps
        self.action_buffer = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 使用 self.unwrapped 穿透外壳直达底层
        device = self.unwrapped.device
        num_envs = self.unwrapped.num_envs
        num_actions = self.action_space.shape[0]  # action_space 默认会暴露，用 self.action_space 即可

        # 初始化一个队列，里面塞满全 0 的动作 tensor
        zero_actions = torch.zeros((num_envs, num_actions), device=device, dtype=torch.float32)
        self.action_buffer = deque([zero_actions.clone() for _ in range(self.delay_steps)], maxlen=self.delay_steps)
        
        return obs, info

    def step(self, action):
        if self.delay_steps > 0:
            # 1. 把网络当前输出的新动作塞进队列尾部
            self.action_buffer.append(action)
            # 2. 把队列最头部的老动作（几步之前的）取出来
            delayed_action = self.action_buffer.popleft()
        else:
            delayed_action = action
            
        # 3. 把“老动作”喂给底层物理引擎执行
        return self.env.step(delayed_action)