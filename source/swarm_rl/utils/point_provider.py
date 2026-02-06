# point_provider.py
from __future__ import annotations

from isaaclab.envs import DirectRLEnv

import torch

from isaaclab.utils import configclass


@configclass
class PointProviderCfg:

    spawn_noise_range: float = 0.0         # 初始点采样噪声范围 (m)
    target_noise_range:float = 0.15       # 目标点采样噪声范围 (m)

    target_hold_distance_threshold: float = 0.5     # 目标点抵达判定阈值-距离 (m)
    target_hold_speed_threshold: float = 0.5        # 目标点抵达判定阈值-速度 (m/s)
    target_hold_threshold_s: float = 0.5            # 目标保持时间阈值 (s)


class PointProvider:
    def __init__(self, cfg: PointProviderCfg, env: DirectRLEnv, free_points: torch.Tensor):

        self.cfg = cfg
        self._robot = env._robot
        
        self.num_envs = env.num_envs
        self.step_dt = env.step_dt
        self.device = env.device

        self._free_points = torch.as_tensor(free_points, dtype=torch.float32, device=self.device)

        self._spawn_points = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_points = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_hold_counter_s = torch.zeros(self.num_envs, device=self.device)


    def update(self) -> tuple[torch.Tensor, torch.Tensor]:

        achieve_target, distance_to_target = self._check_target_hold()
        
        # 当前步抵达的环境目标点保持计时递增
        self._target_hold_counter_s = torch.where(
            achieve_target,
            self._target_hold_counter_s + self.step_dt,
            torch.zeros_like(self._target_hold_counter_s),
        )

        # 重新生成目标点
        target_hold = (self._target_hold_counter_s >= self.cfg.target_hold_threshold_s)
        target_hold_env_ids = target_hold.nonzero(as_tuple=False).squeeze(-1)
        if len(target_hold_env_ids) > 0:
            self._regenerate_target(target_hold_env_ids)

        return achieve_target, distance_to_target


    def resample(self, env_ids: torch.Tensor):

        num_points = len(env_ids)

        # 在 free points 中随机采样初始点与目标点
        self._spawn_points[env_ids] = self._sample_from_free_points(num_points, self.cfg.spawn_noise_range)
        self._regenerate_target(env_ids)
    

    def get_spawn_points(self) -> torch.Tensor:
        return self._spawn_points


    def get_target_points(self) -> torch.Tensor:
        return self._target_points


    def _sample_from_free_points(self, num_points: int, noise_range: float = 0.0) -> torch.Tensor:
        """ 从 free points 中批量随机采样点，并加入三轴均服从 [-noise_range, noise_range] 均匀分布的随机偏置 """
        points_index = torch.randint(0, len(self._free_points), (num_points,), device=self.device)
        points = self._free_points[points_index]
        noise = (torch.rand((num_points, 3), device=self.device) - 0.5) * 2 * noise_range
        return (points + noise)
    

    def _regenerate_target(self, env_ids: torch.Tensor):
        # 在 free points 中重采样目标点
        self._target_points[env_ids] = self._sample_from_free_points(len(env_ids), self.cfg.target_noise_range)
        # 清零目标点保持计时
        self._target_hold_counter_s[env_ids].zero_()
    

    def _check_target_hold(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance_to_target = (self._target_points - self._robot.data.root_state_w[:, :3]).norm(dim=1)
        speed = self._robot.data.root_lin_vel_b.norm(dim=1)
        achieve_target = torch.logical_and(
            distance_to_target < self.cfg.target_hold_distance_threshold,
            speed < self.cfg.target_hold_speed_threshold,
        )
        return achieve_target, distance_to_target