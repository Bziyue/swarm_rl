import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import math


class CurriculumMetric(Enum):
    """Metrics that can be used to drive curriculum progression."""
    SUCCESS_RATE = "success_rate"
    EPISODE_LENGTH = "episode_length" 
    REWARD_THRESHOLD = "reward_threshold"
    COMBINED = "combined"


@dataclass
class CurriculumStage:
    """Defines a single stage in the curriculum with parameter ranges."""
    name: str
    # Environment difficulty parameters
    wind_tau_range: Tuple[float, float] = (0.5, 0.5)
    wind_sigma_range: Tuple[float, float] = (0.05, 0.05)
    thrust_uncertainty_preset: str = "none"
    # Goal positioning parameters
    goal_distance_range: Tuple[float, float] = (5.0, 15.0)
    goal_height_range: Tuple[float, float] = (0.5, 1.5)
    # Obstacle density (for future use)
    obstacle_density: float = 0.0
    # Performance thresholds to advance to next stage
    success_rate_threshold: float = 0.7
    min_episodes: int = 1000
    # Episode timeout scaling
    episode_timeout_scale: float = 1.0


class CurriculumManager:
    """
    Manages curriculum learning progression for the quadcopter environment.
    
    The curriculum progresses through predefined stages based on performance metrics,
    gradually increasing task difficulty across multiple dimensions including:
    - Wind disturbances
    - Thrust uncertainty 
    - Goal positioning difficulty
    - Episode constraints
    """
    
    def __init__(
        self,
        device: torch.device,
        num_envs: int,
        curriculum_metric: CurriculumMetric = CurriculumMetric.SUCCESS_RATE,
        evaluation_window: int = 500,
        enable_curriculum: bool = True,
        manual_stage: Optional[int] = None
    ):
        """
        Initialize the curriculum manager.
        
        Parameters
        ----------
        device : torch.device
            Device for tensor operations
        num_envs : int
            Number of parallel environments
        curriculum_metric : CurriculumMetric
            Primary metric for curriculum progression
        evaluation_window : int
            Number of episodes to evaluate for progression decisions
        enable_curriculum : bool
            Whether curriculum learning is enabled
        manual_stage : Optional[int]
            If specified, locks curriculum to this stage (for testing)
        """
        self.device = device
        self.num_envs = num_envs
        self.curriculum_metric = curriculum_metric
        self.evaluation_window = evaluation_window
        self.enable_curriculum = enable_curriculum
        self.manual_stage = manual_stage
        
        # Define curriculum stages
        self.stages = self._define_curriculum_stages()
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        
        # Performance tracking
        self.episode_history = []
        self.episodes_in_current_stage = 0
        self.total_episodes = 0
        
        # Stage progression tracking
        self.stage_start_episode = 0
        self.progression_history = []
        
        # Environment-specific parameter caches
        self._cached_wind_params = {}
        self._cached_thrust_params = {}
        self._cached_goal_params = {}
        
        # Add smoothing for performance metrics
        self._performance_smoother = 0.9  # Exponential moving average factor
        self._smoothed_success_rate = 0.0
        self._smoothed_episode_length = 0.0
        self._smoothed_reward = 0.0
        
        # Add minimum performance consistency tracking
        self._consistent_performance_count = 0
        self._min_consistent_episodes = 200  # Require consistent performance before advancing
        
        print(f"Curriculum Manager initialized with {len(self.stages)} stages")
        print(f"Starting stage: {self.current_stage.name}")
        
    def _define_curriculum_stages(self) -> List[CurriculumStage]:
        """Define the curriculum progression stages."""
        return [
            CurriculumStage(
                name="Stage 1: Calm Indoor",
                wind_tau_range=(0.05, 0.1),
                wind_sigma_range=(0.0, 0.0),
                thrust_uncertainty_preset="none",
                goal_distance_range=(4.0, 8.0),
                goal_height_range=(0.5, 1.0),
                episode_timeout_scale=1.2,
                success_rate_threshold=0.80,
                min_episodes=800
            ),
            CurriculumStage(
                name="Stage 2: Light Breeze",
                wind_tau_range=(0.4, 0.6),
                wind_sigma_range=(0.05, 0.1),
                thrust_uncertainty_preset="none",
                goal_distance_range=(8.0, 16.0),
                goal_height_range=(0.4, 1.2),
                episode_timeout_scale=1.0,
                success_rate_threshold=0.80,
                min_episodes=1000
            ),
            CurriculumStage(
                name="Stage 3: Mild Thrust Uncertainty",
                wind_tau_range=(0.8, 1.2),
                wind_sigma_range=(0.05, 0.1),
                thrust_uncertainty_preset="mild",
                goal_distance_range=(16.0, 32.0),
                goal_height_range=(0.4, 1.3),
                episode_timeout_scale=1.0,
                success_rate_threshold=0.80,
                min_episodes=1200
            ),
            CurriculumStage(
                name="Stage 4: Moderate Conditions",
                wind_tau_range=(0.6, 1.5),
                wind_sigma_range=(0.1, 0.2),
                thrust_uncertainty_preset="moderate",
                goal_distance_range=(32.0, 64.0),
                goal_height_range=(0.3, 1.5),
                episode_timeout_scale=1.0,
                success_rate_threshold=0.80,
                min_episodes=1500
            ),
            CurriculumStage(
                name="Stage 5: Challenging Wind",
                wind_tau_range=(0.4, 1.8),
                wind_sigma_range=(0.15, 0.3),
                thrust_uncertainty_preset="moderate", 
                goal_distance_range=(32.0, 64.0),
                goal_height_range=(0.3, 1.8),
                episode_timeout_scale=0.9,
                success_rate_threshold=0.80,
                min_episodes=2000
            ),
            CurriculumStage(
                name="Stage 6: Expert Level",
                wind_tau_range=(0.3, 2.0),
                wind_sigma_range=(0.2, 0.5),
                thrust_uncertainty_preset="severe",
                goal_distance_range=(32.0, 64.0),
                goal_height_range=(0.3, 2.0),
                episode_timeout_scale=0.8,
                success_rate_threshold=0.80,
                min_episodes=float('inf')  # Final stage
            )
        ]
    
    def update_episode_outcome(self, success_rate: float, avg_episode_length: float, avg_reward: float) -> bool:
        """
        Update curriculum based on episode outcomes.
        
        Parameters
        ----------
        success_rate : float
            Recent success rate (0.0 to 1.0)
        avg_episode_length : float
            Average episode length in recent window
        avg_reward : float
            Average episode reward in recent window
            
        Returns
        -------
        bool
            True if curriculum stage advanced
        """
        if not self.enable_curriculum or self.manual_stage is not None:
            return False
            
        # Apply exponential smoothing to reduce noise
        if self.total_episodes == 0:
            self._smoothed_success_rate = success_rate
            self._smoothed_episode_length = avg_episode_length
            self._smoothed_reward = avg_reward
        else:
            self._smoothed_success_rate = (self._performance_smoother * self._smoothed_success_rate + 
                                         (1 - self._performance_smoother) * success_rate)
            self._smoothed_episode_length = (self._performance_smoother * self._smoothed_episode_length + 
                                           (1 - self._performance_smoother) * avg_episode_length)
            self._smoothed_reward = (self._performance_smoother * self._smoothed_reward + 
                                   (1 - self._performance_smoother) * avg_reward)
        
        # Record episode metrics
        episode_data = {
            'episode': self.total_episodes,
            'stage': self.current_stage_idx,
            'success_rate': success_rate,
            'smoothed_success_rate': self._smoothed_success_rate,
            'avg_episode_length': avg_episode_length,
            'smoothed_episode_length': self._smoothed_episode_length,
            'avg_reward': avg_reward,
            'smoothed_reward': self._smoothed_reward
        }
        self.episode_history.append(episode_data)
        
        # Keep only recent history
        if len(self.episode_history) > self.evaluation_window * 2:
            self.episode_history = self.episode_history[-self.evaluation_window:]
        
        self.episodes_in_current_stage += 1
        self.total_episodes += 1
        
        # Check if we should advance to next stage
        return self._check_stage_advancement(success_rate, avg_episode_length, avg_reward)
    
    def _check_stage_advancement(self, success_rate: float, avg_episode_length: float, avg_reward: float) -> bool:
        """Check if conditions are met to advance to the next curriculum stage."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already at final stage
            
        # Must meet minimum episode requirement
        if self.episodes_in_current_stage < self.current_stage.min_episodes:
            return False
            
        # Check primary advancement criterion using smoothed metrics
        advancement_ready = False
        
        if self.curriculum_metric == CurriculumMetric.SUCCESS_RATE:
            advancement_ready = self._smoothed_success_rate >= self.current_stage.success_rate_threshold
        elif self.curriculum_metric == CurriculumMetric.EPISODE_LENGTH:
            # Advance if episodes are getting longer (better exploration)
            target_length = 200  # Adjust based on your environment
            advancement_ready = self._smoothed_episode_length >= target_length
        elif self.curriculum_metric == CurriculumMetric.REWARD_THRESHOLD:
            # Define target reward based on stage
            target_reward = -10.0 + (self.current_stage_idx * 5.0)
            advancement_ready = self._smoothed_reward >= target_reward
        elif self.curriculum_metric == CurriculumMetric.COMBINED:
            # Combined criteria - all must be satisfied
            success_ok = self._smoothed_success_rate >= self.current_stage.success_rate_threshold
            length_ok = self._smoothed_episode_length >= 150
            reward_ok = self._smoothed_reward >= -20.0
            advancement_ready = success_ok and length_ok and reward_ok
            
        # Require consistent performance before advancing
        if advancement_ready:
            self._consistent_performance_count += 1
            if self._consistent_performance_count >= self._min_consistent_episodes:
                return self._advance_stage()
        else:
            self._consistent_performance_count = 0  # Reset if performance drops
            
        return False
    
    def _advance_stage(self) -> bool:
        """Advance to the next curriculum stage."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
            
        # Record progression
        progression_data = {
            'from_stage': self.current_stage_idx,
            'to_stage': self.current_stage_idx + 1,
            'episode': self.total_episodes,
            'episodes_in_stage': self.episodes_in_current_stage,
            'success_rate_at_advancement': self._smoothed_success_rate,
            'episode_length_at_advancement': self._smoothed_episode_length,
            'reward_at_advancement': self._smoothed_reward
        }
        self.progression_history.append(progression_data)
        
        # Advance stage
        self.current_stage_idx += 1
        self.current_stage = self.stages[self.current_stage_idx]
        self.episodes_in_current_stage = 0
        self.stage_start_episode = self.total_episodes
        self._consistent_performance_count = 0  # Reset consistency counter
        
        # Clear cached parameters to force regeneration
        self._cached_wind_params.clear()
        self._cached_thrust_params.clear()
        self._cached_goal_params.clear()
        
        print(f"Curriculum advanced to {self.current_stage.name} at episode {self.total_episodes}")
        print(f"  Success rate: {self._smoothed_success_rate:.3f}")
        print(f"  Episode length: {self._smoothed_episode_length:.1f}")
        print(f"  Reward: {self._smoothed_reward:.2f}")
        return True
    
    def get_wind_parameters(self) -> Dict[str, float]:
        """Get current wind parameters based on curriculum stage."""
        if 'wind' not in self._cached_wind_params:
            stage = self.stages[self.manual_stage] if self.manual_stage is not None else self.current_stage
            
            # Sample parameters within stage ranges
            tau = np.random.uniform(stage.wind_tau_range[0], stage.wind_tau_range[1])
            sigma = np.random.uniform(stage.wind_sigma_range[0], stage.wind_sigma_range[1])
            
            self._cached_wind_params['wind'] = {'tau': tau, 'sigma': sigma}
            
        return self._cached_wind_params['wind']
    
    def get_thrust_uncertainty_preset(self) -> str:
        """Get current thrust uncertainty preset based on curriculum stage."""
        stage = self.stages[self.manual_stage] if self.manual_stage is not None else self.current_stage
        return stage.thrust_uncertainty_preset
    
    def get_goal_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Get goal positioning parameters based on curriculum stage."""
        if 'goal' not in self._cached_goal_params:
            stage = self.stages[self.manual_stage] if self.manual_stage is not None else self.current_stage
            
            self._cached_goal_params['goal'] = {
                'distance_range': stage.goal_distance_range,
                'height_range': stage.goal_height_range
            }
            
        return self._cached_goal_params['goal']
    
    def get_episode_timeout_scale(self) -> float:
        """Get episode timeout scaling factor based on curriculum stage."""
        stage = self.stages[self.manual_stage] if self.manual_stage is not None else self.current_stage
        return stage.episode_timeout_scale
    
    def set_manual_stage(self, stage_idx: Optional[int]) -> None:
        """Manually set curriculum stage (for testing/debugging)."""
        if stage_idx is not None and (stage_idx < 0 or stage_idx >= len(self.stages)):
            raise ValueError(f"Stage index {stage_idx} out of range [0, {len(self.stages)-1}]")
            
        self.manual_stage = stage_idx
        if stage_idx is not None:
            print(f"Curriculum manually set to stage {stage_idx}: {self.stages[stage_idx].name}")
        else:
            print("Manual curriculum override disabled")
            
        # Clear caches to force parameter regeneration
        self._cached_wind_params.clear()
        self._cached_thrust_params.clear()
        self._cached_goal_params.clear()
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about the current curriculum stage."""
        active_stage = self.stages[self.manual_stage] if self.manual_stage is not None else self.current_stage
        active_stage_idx = self.manual_stage if self.manual_stage is not None else self.current_stage_idx
        
        return {
            'stage_index': active_stage_idx,
            'stage_name': active_stage.name,
            'total_stages': len(self.stages),
            'episodes_in_stage': self.episodes_in_current_stage,
            'total_episodes': self.total_episodes,
            'manual_override': self.manual_stage is not None,
            'wind_params': self.get_wind_parameters(),
            'thrust_preset': self.get_thrust_uncertainty_preset(),
            'goal_params': self.get_goal_parameters(),
            'timeout_scale': self.get_episode_timeout_scale(),
            'smoothed_success_rate': self._smoothed_success_rate,
            'smoothed_episode_length': self._smoothed_episode_length,
            'smoothed_reward': self._smoothed_reward,
            'consistent_performance_count': self._consistent_performance_count,
            'advancement_progress': min(1.0, self._consistent_performance_count / self._min_consistent_episodes)
        }
    
    def get_progression_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum progression history."""
        return {
            'current_stage': self.current_stage_idx,
            'total_episodes': self.total_episodes,
            'progression_history': self.progression_history,
            'episodes_per_stage': [
                len([e for e in self.episode_history if e['stage'] == i]) 
                for i in range(len(self.stages))
            ]
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save curriculum manager state for checkpointing."""
        return {
            'current_stage_idx': self.current_stage_idx,
            'episodes_in_current_stage': self.episodes_in_current_stage,
            'total_episodes': self.total_episodes,
            'stage_start_episode': self.stage_start_episode,
            'progression_history': self.progression_history,
            'episode_history': self.episode_history[-self.evaluation_window:],  # Save recent history only
            'manual_stage': self.manual_stage
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load curriculum manager state from checkpoint."""
        self.current_stage_idx = state['current_stage_idx']
        self.current_stage = self.stages[self.current_stage_idx]
        self.episodes_in_current_stage = state['episodes_in_current_stage']
        self.total_episodes = state['total_episodes']
        self.stage_start_episode = state['stage_start_episode']
        self.progression_history = state['progression_history']
        self.episode_history = state['episode_history']
        self.manual_stage = state.get('manual_stage', None)
        
        # Clear caches
        self._cached_wind_params.clear()
        self._cached_thrust_params.clear()
        self._cached_goal_params.clear()
        
        print(f"Curriculum state loaded. Current stage: {self.current_stage.name}")
