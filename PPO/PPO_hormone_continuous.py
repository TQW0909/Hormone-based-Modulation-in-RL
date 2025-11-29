import os
import sys
import gymnasium as gym
import numpy as np
import math, random
import wandb
import torch
from collections import deque
from typing import Dict, Optional, Tuple, List

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import wandb_context

# ============================================================================
# KEEP ALL YOUR EXISTING HELPER CLASSES UNCHANGED
# ============================================================================

class EfficientNoveltyDetector:
    """
    Tracks observation space statistics incrementally to detect novelty for Adrenaline
    """
    
    def __init__(self, obs_dim: int, alpha: float = 0.01):
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)
        self.alpha = alpha
        self.count = 0
        
        # Track recent novelty values for normalization
        self.novelty_history = deque(maxlen=1000)
        self.novelty_baseline = None
        
    def update(self, obs_batch: np.ndarray) -> float:
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
            
        # Compute novelty before updating (important!)
        novelties = []
        for obs in obs_batch:
            # Mahalanobis-like distance
            diff = obs - self.mean
            nov = np.sqrt(np.mean(diff**2 / (self.var + 1e-8)))
            novelties.append(nov)
            
        avg_novelty = np.mean(novelties)
        
        # Update running statistics
        for obs in obs_batch:
            delta = obs - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta**2)
            
        self.count += len(obs_batch)
        
        # Update novelty baseline (slower adaptation)
        self.novelty_history.append(avg_novelty)
        if self.novelty_baseline is None:
            self.novelty_baseline = avg_novelty
        else:
            self.novelty_baseline = 0.99 * self.novelty_baseline + 0.01 * avg_novelty
            
        # Return normalized novelty
        return avg_novelty / (self.novelty_baseline + 1e-8)
    
    def get_intrinsic_reward(self, obs: np.ndarray, scale: float = 0.01) -> float:
        diff = obs - self.mean
        novelty = np.sqrt(np.mean(diff**2 / (self.var + 1e-8)))
        
        # Normalize and scale
        if self.novelty_baseline is not None:
            novelty = novelty / (self.novelty_baseline + 1e-8)
        
        # Apply diminishing returns
        return scale * np.tanh(novelty - 1.0)


class AdvancedHormoneCoupler:
    """
    Implements sophisticated hormone interactions with memory and adaptation.
    This prevents oscillations and provides more biological realism.
    """
    
    def __init__(self, memory_window: int = 50):
        # Hormone history for trend analysis
        self.A_history = deque(maxlen=memory_window)
        self.C_history = deque(maxlen=memory_window)
        self.D_history = deque(maxlen=memory_window)
        
        # Coupling strength parameters
        self.coupling_strengths = {
            'CA': 0.7,   # Cortisol suppresses Adrenaline
            'CD': 0.6,   # Cortisol suppresses Dopamine
            'DC': 0.3,   # Dopamine inhibits Cortisol
            'AD': 0.2,   # Adrenaline excites Dopamine
            'AC': 0.1,   # Adrenaline can increase Cortisol (stress)
        }
        
        # Receptor sensitivity (adapts over time to prevent saturation)
        self.receptor_sensitivity = {
            'A': 1.0,
            'C': 1.0, 
            'D': 1.0,
        }
        
    def couple_hormones(
        self,
        A: float,
        C: float,
        D: float,
        external_stress: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Advanced hormone coupling with memory effects.
        
        Args:
            A, C, D: Current hormone levels
            external_stress: Additional stress signal (e.g., high KL divergence)
            
        Returns:
            Coupled hormone levels (A_eff, C_eff, D_eff)
        """
        # Store history
        self.A_history.append(A)
        self.C_history.append(C)
        self.D_history.append(D)
        
        # Calculate trends (positive = increasing, negative = decreasing)
        A_trend = self._get_trend('A')
        C_trend = self._get_trend('C')
        D_trend = self._get_trend('D')
        
        # Adapt receptor sensitivity (habituation/sensitization)
        self._adapt_receptor_sensitivity()
        
        # Core coupling equations with memory effects
        
        # Adrenaline: suppressed by cortisol, but resists if trending up
        A_suppression = self.coupling_strengths['CA'] * C
        if A_trend > 0:  # If A is increasing, resist suppression
            A_suppression *= (1.0 - 0.3 * min(1.0, A_trend))
        A_eff = A * (1.0 - A_suppression) * self.receptor_sensitivity['A']
        
        # Dopamine: excited by adrenaline, suppressed by cortisol
        D_excitation = self.coupling_strengths['AD'] * A
        D_suppression = self.coupling_strengths['CD'] * C
        D_eff = (D + D_excitation) * (1.0 - D_suppression) * self.receptor_sensitivity['D']
        
        # Cortisol: inhibited by dopamine, excited by stress
        C_base_inhibition = self.coupling_strengths['DC'] * D
        C_stress_boost = self.coupling_strengths['AC'] * A + external_stress
        
        # Cortisol has inertia (slower to decrease than increase)
        if C_trend < 0:  # Decreasing - add resistance
            C_eff = C * (1.0 - 0.5 * C_base_inhibition) + C_stress_boost
        else:  # Increasing or stable
            C_eff = C * (1.0 - C_base_inhibition) + C_stress_boost
            
        C_eff *= self.receptor_sensitivity['C']
        
        # Soft clamping to [0, 1]
        A_eff = self._soft_clamp(A_eff, 0.0, 1.0)
        C_eff = self._soft_clamp(C_eff, 0.0, 1.0)
        D_eff = self._soft_clamp(D_eff, 0.0, 1.0)
        
        return A_eff, C_eff, D_eff
    
    def _get_trend(self, hormone: str) -> float:
        """Calculate normalized trend for a hormone."""
        history = getattr(self, f'{hormone}_history')
        if len(history) < 10:
            return 0.0
            
        # Compare recent average to past average
        recent = np.mean(list(history)[-5:])
        past = np.mean(list(history)[-20:-10]) if len(history) >= 20 else np.mean(list(history)[:5])
        trend = (recent - past) / (past + 1e-8)
        return np.clip(trend, -1.0, 1.0)
    
    def _adapt_receptor_sensitivity(self):
        """
        Implement receptor adaptation (habituation/sensitization).
        High sustained levels → decreased sensitivity (habituation)
        Low sustained levels → increased sensitivity (sensitization)
        """
        if len(self.A_history) < 20:
            return
            
        # Calculate average levels over recent history
        A_avg = np.mean(list(self.A_history)[-20:])
        C_avg = np.mean(list(self.C_history)[-20:])
        D_avg = np.mean(list(self.D_history)[-20:])
        
        # Adapt sensitivity (inverse relationship with average level)
        # High average → lower sensitivity (habituation)
        # Low average → higher sensitivity (sensitization)
        self.receptor_sensitivity['A'] = 0.8 + 0.4 * (1.0 - A_avg)
        self.receptor_sensitivity['C'] = 0.8 + 0.4 * (1.0 - C_avg)
        self.receptor_sensitivity['D'] = 0.8 + 0.4 * (1.0 - D_avg)
        
        # Clamp to reasonable range
        for key in self.receptor_sensitivity:
            self.receptor_sensitivity[key] = np.clip(
                self.receptor_sensitivity[key], 0.5, 1.5
            )
    
    def _soft_clamp(self, x: float, min_val: float, max_val: float) -> float:
        """Soft clamping using smooth transitions near boundaries."""
        if x <= min_val:
            return min_val
        elif x >= max_val:
            return max_val
        else:
            # Smooth transition near boundaries (within 10% of range)
            range_val = max_val - min_val
            margin = 0.1 * range_val
            
            if x < min_val + margin:
                # Near minimum - smooth transition
                t = (x - min_val) / margin
                return min_val + margin * np.tanh(2 * t) / np.tanh(2)
            elif x > max_val - margin:
                # Near maximum - smooth transition
                t = (max_val - x) / margin
                return max_val - margin * np.tanh(2 * t) / np.tanh(2)
            else:
                return x

class MultiScaleProgressTracker:
    """
    Tracks learning progress at multiple temporal scales for robust dopamine signals.
    """
    
    def __init__(self):
        # Different time scales (in episodes)
        self.scales = {
            'immediate': deque(maxlen=10),
            'short': deque(maxlen=50),
            'medium': deque(maxlen=200),
            'long': deque(maxlen=1000),
        }
        
        # Track different metrics
        self.metrics = {
            'reward': {},
            'success_rate': {},
            'td_error': {},
            'episode_length': {},
            'value_accuracy': {},  # How well value function predicts returns
        }
        
        # Initialize metric storage for each scale
        for metric in self.metrics:
            for scale in self.scales:
                self.metrics[metric][scale] = deque(maxlen=self.scales[scale].maxlen)
        
        # Best performance tracking
        self.best_reward = -float('inf')
        self.best_success_rate = 0.0
        self.plateau_counter = 0
        
    def update(
        self,
        episode_rewards: List[float],
        td_errors: np.ndarray,
        values: np.ndarray,
        returns: np.ndarray,
        dones: np.ndarray,
        success_threshold: Optional[float] = None
    ):
        """Update progress metrics from rollout data."""
        
        # Calculate episode-level metrics
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            
            # Success rate (environment-specific)
            if success_threshold is not None:
                success_rate = np.mean([r >= success_threshold for r in episode_rewards])
            else:
                # Default: positive reward = success
                success_rate = np.mean([r > 0 for r in episode_rewards])
            
            # Update best tracking
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.plateau_counter = 0
            else:
                self.plateau_counter += 1
        else:
            mean_reward = 0.0
            success_rate = 0.0
            
        # Calculate value function accuracy (correlation between values and returns)
        if len(values) > 0 and len(returns) > 0:
            correlation = np.corrcoef(values.flatten(), returns.flatten())[0, 1]
            value_accuracy = correlation if not np.isnan(correlation) else 0.0
        else:
            value_accuracy = 0.0
            
        # Calculate mean absolute TD error
        mean_td_error = np.mean(np.abs(td_errors)) if len(td_errors) > 0 else 0.0
        
        # Calculate mean episode length from dones
        episode_lengths = []
        current_length = 0
        for done in dones.flatten():
            current_length += 1
            if done:
                episode_lengths.append(current_length)
                current_length = 0
        mean_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        # Update all scales
        for scale in self.scales:
            self.metrics['reward'][scale].append(mean_reward)
            self.metrics['success_rate'][scale].append(success_rate)
            self.metrics['td_error'][scale].append(mean_td_error)
            self.metrics['episode_length'][scale].append(mean_episode_length)
            self.metrics['value_accuracy'][scale].append(value_accuracy)
    
    def get_progress_signal(self) -> Dict[str, float]:
        """
        Calculate multi-scale progress signals for dopamine modulation.
        
        Returns:
            Dictionary of progress indicators
        """
        signals = {}
        
        # Reward improvement across scales
        for scale in ['immediate', 'short', 'medium']:
            if len(self.metrics['reward'][scale]) >= 5:
                recent = np.mean(list(self.metrics['reward'][scale])[-5:])
                if len(self.metrics['reward'][scale]) >= 10:
                    past = np.mean(list(self.metrics['reward'][scale])[-10:-5])
                    improvement = (recent - past) / (abs(past) + 1e-8)
                    signals[f'reward_trend_{scale}'] = np.clip(improvement, -1.0, 1.0)
                else:
                    signals[f'reward_trend_{scale}'] = 0.0
            else:
                signals[f'reward_trend_{scale}'] = 0.0
                
        # TD error reduction (indicates learning progress)
        if len(self.metrics['td_error']['short']) >= 10:
            recent_td = np.mean(list(self.metrics['td_error']['short'])[-5:])
            past_td = np.mean(list(self.metrics['td_error']['short'])[-20:-10]) \
                if len(self.metrics['td_error']['short']) >= 20 else \
                np.mean(list(self.metrics['td_error']['short'])[:5])
            td_improvement = (past_td - recent_td) / (past_td + 1e-8)
            signals['td_improvement'] = np.clip(td_improvement, -1.0, 1.0)
        else:
            signals['td_improvement'] = 0.0
            
        # Value function accuracy
        if len(self.metrics['value_accuracy']['immediate']) >= 5:
            signals['value_accuracy'] = np.mean(list(self.metrics['value_accuracy']['immediate'])[-5:])
        else:
            signals['value_accuracy'] = 0.0
            
        # Success rate trend
        if len(self.metrics['success_rate']['short']) >= 10:
            recent_success = np.mean(list(self.metrics['success_rate']['short'])[-5:])
            past_success = np.mean(list(self.metrics['success_rate']['short'])[-10:-5])
            signals['success_trend'] = recent_success - past_success
        else:
            signals['success_trend'] = 0.0
            
        # Plateau detection (inverse progress)
        signals['plateau_severity'] = min(1.0, self.plateau_counter / 50.0)
        
        # Combined progress score (weighted average)
        weights = {
            'reward_trend_immediate': 0.15,
            'reward_trend_short': 0.20,
            'reward_trend_medium': 0.15,
            'td_improvement': 0.20,
            'value_accuracy': 0.15,
            'success_trend': 0.15,
        }
        
        combined_progress = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            if key in signals:
                combined_progress += weight * signals[key]
                total_weight += weight
                
        if total_weight > 0:
            combined_progress /= total_weight
            
        # Apply plateau penalty
        combined_progress *= (1.0 - 0.5 * signals['plateau_severity'])
        
        signals['combined_progress'] = np.clip(combined_progress, -1.0, 1.0)
        
        return signals

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def squash(x, k=1.5):
    """Exponential moving average with alpha in (0,1]."""
    return 1.0 / (1.0 + math.exp(-k*(x - 1.0)))

def EMA(prev, now, alpha=0.2):
    """Smooth sigmoid-like mapping: x≈1 -> ~0.5 ; larger x -> closer to 1, smaller -> closer to 0."""
    return alpha*now + (1-alpha)*prev if prev is not None else now

def schedule_value(sched, progress: float, default_progress: float = 1.0) -> float:
    """Return numeric value from an SB3 schedule or a float."""
    try:
        return float(sched(progress)) if callable(sched) else float(sched)
    except Exception:
        return float(sched(default_progress)) if callable(sched) else float(sched)


# ============================================================================
# HORMONE PPO CALLBACK - SIMPLIFIED FOR CONTINUOUS ONLY
# ============================================================================

class HormonePPOCallback(BaseCallback):
    def __init__(
        self,
        warmup_rollouts: int = 20,
        verbose: int = 0,
        # Base hormone levels
        base_A: float = 0.5,
        base_C: float = 0.5,
        base_D: float = 0.5,
        # Initial hyperparameters - CONTINUOUS DEFAULTS
        ent0: float = 0.0,  # Continuous control doesn't use entropy
        lr0: float = 3e-4,
        clip0: float = 0.2,
        # Clamping ranges - WIDER FOR CONTINUOUS
        clamp_ent: Tuple[float, float] = (0.0, 0.0),  # Keep at 0
        clamp_lr: Tuple[float, float] = (2.5e-4, 3.5e-4),  # ±17%
        clamp_clip: Tuple[float, float] = (0.18, 0.22),  # ±10%
        # Feature flags
        use_A: bool = True,
        use_C: bool = True,
        use_D: bool = True,
        # Pulse-decay parameters
        T12_A: float = 12.0,
        T12_C: float = 8.0,
        T12_D: float = 15.0,
        k_A: float = 0.10,
        k_C: float = 0.08,
        k_D: float = 0.10,
        # Thresholds
        th_A: float = 0.30,
        th_C: float = 0.40,
        th_D: float = 0.20,
    ):
        super().__init__(verbose)

        # Core parameters
        self.rollout_idx = 0
        self.warmup_rollouts = warmup_rollouts   

        self.A = base_A
        self.C = base_C
        self.D = base_D
        self.base_A = base_A
        self.base_C = base_C
        self.base_D = base_D

        self.ent0 = ent0
        self.lr0 = lr0
        self.clip0 = clip0
        self.clamp_ent = clamp_ent
        self.clamp_lr = clamp_lr
        self.clamp_clip = clamp_clip

        self.use_A = use_A
        self.use_C = use_C
        self.use_D = use_D

        self.novelty_detector = None
        self.hormone_coupler = None
        self.progress_tracker = MultiScaleProgressTracker()
        
        self.H0_A = base_A
        self.H0_C = base_C
        self.H0_D = base_D
        
        self.T12 = {'A': T12_A, 'C': T12_C, 'D': T12_D}
        self.lam = {h: np.log(2) / t12 for h, t12 in self.T12.items()}

        self.k = {'A': k_A, 'C': k_C, 'D': k_D}
        self.th = {'A': th_A, 'C': th_C, 'D': th_D}

        self.refrac = {'A': 0, 'C': 0, 'D': 0}
        self.refrac_len = {'A': 1, 'C': 2, 'D': 1}
        
        self.ema_adv = None
        self.ema_td = None
        self.ema_vol = None
        self.ref_adv = None
        self.ref_td = None
        
        self.reward_window = deque(maxlen=200)
        self.ema_reward_fast = None
        self.ema_reward_slow = None
        
        self.best_td = None
        self.td_slow = None

        self.signal_history = {}
        self.signal_smooth_alpha = 0.5
        
        self.hormone_momentum = 0.7
        self.prev_A = self.A
        self.prev_C = self.C
        self.prev_D = self.D
        
        self._last_episode_rewards = []
        self._last_approx_kl = None
        self.target_kl = None
        
        self._last_signals = {}
        self._last_pulses = {'A': 0, 'C': 0, 'D': 0}

    def _init_callback(self) -> None:
        """Initialize callback with model."""
        super()._init_callback()
        
        if self.novelty_detector is None:
            obs_shape = self.model.observation_space.shape
            obs_dim = np.prod(obs_shape)
            self.novelty_detector = EfficientNoveltyDetector(obs_dim)
            if self.verbose > 0:
                print(f"[HormonePPO] Initialized novelty detector for obs_dim={obs_dim}")
            
        self.target_kl = getattr(self.model, "target_kl", None)

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print("[HormonePPO] Training started.")

    def _on_rollout_start(self) -> None:
        if self.verbose > 1:
            print(f"[HormonePPO] Rollout {self.rollout_idx}: start")
        self._apply_modulation()

    def _on_rollout_end(self) -> None:
        # Emergency reset for catastrophic failures
        if self.use_C and len(self._last_episode_rewards) > 0:
            recent_reward = np.mean(self._last_episode_rewards)
            if recent_reward < -50 and self.C > 0.6:
                if self.verbose:
                    print(f"🚨 EMERGENCY RESET: Reward={recent_reward:.1f}, C={self.C:.3f}")
                    print(f"   Resetting C to baseline and restoring LR")
                self.C = 0.5
                self.refrac['C'] = 10
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = self.lr0
        
        signals = self._collect_signals()
        
        if self.progress_tracker is not None:
            self._update_progress_tracker()
            
        if self.rollout_idx >= self.warmup_rollouts:
            normalized = self._normalize_signals(signals)
            self._update_hormones(normalized)
            
        self.rollout_idx += 1

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("[HormonePPO] Training finished.")

    def _collect_signals(self) -> Dict[str, float]:
        """Collect all signals with improvements."""
        buf = self.model.rollout_buffer
        signals = {}
        
        adv = buf.advantages.flatten()
        rets = buf.returns.flatten()
        vals = buf.values.flatten()
        td = rets - vals
        
        signals['adv_magnitude'] = float(np.mean(np.abs(adv)))
        signals['td_error'] = float(np.mean(np.abs(td)))

        td_med = float(np.median(td)) if td.size > 0 else 0.0
        signals['td_mad'] = float(np.median(np.abs(td - td_med))) if td.size > 0 else 0.0
        
        if self.novelty_detector is not None:
            obs = buf.observations.reshape((-1,) + buf.observations.shape[2:])
            if len(obs) > 256:
                idx = np.random.choice(len(obs), 256, replace=False)
                obs = obs[idx]
            signals['novelty'] = self.novelty_detector.update(obs)
        else:
            signals['novelty'] = 1.0 + 0.5 * np.std(adv) / (np.mean(np.abs(adv)) + 1e-8)
            
        episodes = list(self.model.ep_info_buffer)
        rewards = [float(e['r']) for e in episodes if 'r' in e]
        signals['mean_reward'] = np.mean(rewards) if rewards else 0.0
        self._last_episode_rewards = rewards
        
        for r in rewards:
            self.reward_window.append(r)
    
        for key, value in signals.items():
            if key not in self.signal_history:
                self.signal_history[key] = value
            else:
                self.signal_history[key] = (self.signal_smooth_alpha * value + 
                                        (1 - self.signal_smooth_alpha) * self.signal_history[key])
            signals[key] = self.signal_history[key]
        
        return signals

    def _update_progress_tracker(self):
        """Update multi-scale progress tracking."""
        if not self.progress_tracker:
            return
            
        buf = self.model.rollout_buffer
        
        td_errors = buf.returns.flatten() - buf.values.flatten()
        values = buf.values.flatten()
        returns = buf.returns.flatten()
        dones = buf.episode_starts.flatten()
        
        self.progress_tracker.update(
            self._last_episode_rewards,
            td_errors,
            values,
            returns,
            dones
        )

    def _normalize_signals(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Normalize signals relative to baselines."""
        
        alpha = 0.2
        if self.ema_adv is None:
            self.ema_adv = signals['adv_magnitude']
            self.ema_td = signals['td_error']
            self.ema_vol = signals['td_mad']
        else:
            self.ema_adv = alpha * signals['adv_magnitude'] + (1 - alpha) * self.ema_adv
            self.ema_td = alpha * signals['td_error'] + (1 - alpha) * self.ema_td
            self.ema_vol = alpha * signals['td_mad'] + (1 - alpha) * self.ema_vol

        if self.ref_adv is None:
            self.ref_adv = self.ema_adv
            self.ref_td = self.ema_td
            self.best_td = self.ema_td
            self.td_slow = self.ema_td
            
        self.td_slow = 0.95 * self.td_slow + 0.05 * self.ema_td if self.td_slow else self.ema_td
        self.best_td = min(self.best_td, self.ema_td) if self.best_td else self.ema_td
        
        if len(self.reward_window) >= 5:
            current_reward = np.mean(list(self.reward_window)[-5:])
            self.ema_reward_fast = 0.3 * current_reward + 0.7 * self.ema_reward_fast \
                if self.ema_reward_fast else current_reward
            self.ema_reward_slow = 0.05 * current_reward + 0.95 * self.ema_reward_slow \
                if self.ema_reward_slow else current_reward
            
        normalized = {
            'adv_hat': self.ema_adv / (self.ref_adv + 1e-8),
            'td_hat': self.ema_td / (self.ref_td + 1e-8),
            'vol_hat': signals['td_mad'] / (self.ema_vol + 1e-8),
            'nov_hat': signals['novelty'],
        }
        
        if self.progress_tracker:
            progress_signals = self.progress_tracker.get_progress_signal()
            normalized['progress'] = progress_signals.get('combined_progress', 0.0)
        else:
            td_improvement = max(0, (self.td_slow - self.ema_td) / (self.td_slow + 1e-8))
            reward_trend = 0.0
            if self.ema_reward_fast and self.ema_reward_slow:
                reward_trend = np.clip((self.ema_reward_fast - self.ema_reward_slow) / 
                                      (abs(self.ema_reward_slow) + 1e-8), -1, 1)
            normalized['progress'] = 0.6 * td_improvement + 0.4 * max(0, reward_trend)
            
        self._last_signals = normalized
            
        return normalized

    def _update_hormones(self, normalized: Dict[str, float]):
        """Update hormone levels with pulse-decay dynamics."""
        
        pulses = self._calculate_pulses(normalized)
        self._last_pulses = pulses
        
        for hormone in ['A', 'C', 'D']:
            use_flag = getattr(self, f'use_{hormone}')
            
            if use_flag:
                current = getattr(self, hormone)
                center = getattr(self, f'H0_{hormone}')
                
                new_level = center + (current - center) * (1.0 - self.lam[hormone])
                new_level += pulses[hormone]
                new_level = np.clip(new_level, 0.0, 1.0)
                
                setattr(self, hormone, new_level)
                
                if pulses[hormone] > 0.15:
                    self.refrac[hormone] = self.refrac_len[hormone]
                else:
                    self.refrac[hormone] = max(0, self.refrac[hormone] - 1)
            else:
                setattr(self, hormone, getattr(self, f'base_{hormone}'))
                
        if self.hormone_coupler:
            num_enabled = sum([self.use_A, self.use_C, self.use_D])
            
            if num_enabled > 1:
                external_stress = 0.0
                if self.target_kl and self._last_approx_kl:
                    kl_ratio = self._last_approx_kl / (self.target_kl + 1e-8)
                    if kl_ratio > 2.0:
                        external_stress = min(0.3, 0.1 * (kl_ratio - 2.0))
                
                A_for_coupling = self.A if self.use_A else 0.0
                C_for_coupling = self.C if self.use_C else 0.0
                D_for_coupling = self.D if self.use_D else 0.0
                
                A_coupled, C_coupled, D_coupled = self.hormone_coupler.couple_hormones(
                    A_for_coupling, C_for_coupling, D_for_coupling, external_stress
                )
                
                if self.use_A:
                    self.A = A_coupled
                if self.use_C:
                    self.C = C_coupled
                if self.use_D:
                    self.D = D_coupled
            
    def _calculate_pulses(self, normalized: Dict[str, float]) -> Dict[str, float]:
        """Calculate hormone pulses from normalized signals."""
        pulses = {}
        
        # Adrenaline
        if self.use_A:
            signal_A = normalized['adv_hat'] * normalized['nov_hat']
            
            adaptive_threshold = self.th['A']
            if self.A < 0.6:
                adaptive_threshold = self.th['A'] * 0.7
            
            if self.refrac['A'] == 0 and signal_A > (1.0 + adaptive_threshold):
                base_pulse = self.k['A'] * (signal_A - 1.0)
                saturation_factor = max(0.0, 1.0 - 2.0 * (self.A - 0.5))
                pulses['A'] = base_pulse * saturation_factor
            else:
                pulses['A'] = 0.0
        else:
            pulses['A'] = 0.0
            
        # Cortisol
        if self.use_C:
            signal_C = normalized['vol_hat']
            
            if not hasattr(self, 'recent_c_pulses'):
                self.recent_c_pulses = deque(maxlen=20)
            
            pulse_density = sum(1 for p in self.recent_c_pulses if p > 0.01)
            
            adaptive_threshold = self.th['C']
            if pulse_density >= 3:
                adaptive_threshold = self.th['C'] + 0.3
                if self.verbose and self.rollout_idx % 5 == 0:
                    print(f"⚠️ C cascade detected! Raising threshold to {adaptive_threshold:.2f}")
            
            if self.refrac['C'] == 0 and signal_C > (1.0 + adaptive_threshold):
                base_pulse = self.k['C'] * (signal_C - 1.0)
                
                if self.use_D:
                    base_pulse *= (1.0 - 0.3 * self.D)
                
                saturation_factor = max(0.3, 1.0 - (self.C - 0.5) * 1.5)
                pulses['C'] = base_pulse * saturation_factor
                
                self.recent_c_pulses.append(pulses['C'])
            else:
                pulses['C'] = 0.0
                self.recent_c_pulses.append(0.0)
        else:
            pulses['C'] = 0.0

        # Dopamine
        if self.use_D:
            signal_D = normalized['progress']
            td_bonus = max(0, 1.0 - normalized['td_hat']) * 0.3
            signal_D = signal_D + td_bonus
            
            if self.refrac['D'] == 0 and signal_D > self.th['D']:
                base_pulse = self.k['D'] * signal_D
                saturation_factor = max(0.0, 1.0 - 2.0 * (self.D - 0.5))
                pulses['D'] = base_pulse * saturation_factor
            else:
                pulses['D'] = 0.0
        else:
            pulses['D'] = 0.0
            
        return pulses
        
    def _apply_modulation(self):
        """Apply hormone-modulated hyperparameters - CONTINUOUS CONTROL ONLY."""
        
        if self.rollout_idx < self.warmup_rollouts:
            return
            
        A_eff = self.A if self.use_A else self.base_A
        C_eff = self.C if self.use_C else self.base_C
        D_eff = self.D if self.use_D else self.base_D
        
        # CONTINUOUS CONTROL MODULATION
        # Entropy stays at 0 (not used for continuous control)
        ent = self.ent0
        
        # LR is primary mechanism (all hormones affect it)
        lr_multiplier = (1.0 
                        + 0.15 * (D_eff - 0.5)    # Progress → faster
                        + 0.05 * (A_eff - 0.5)    # Novelty → moderate speedup
                        - 0.10 * (C_eff - 0.5))   # Stress → slower
        lr = self.lr0 * lr_multiplier
        
        # Clip is secondary mechanism (all hormones affect it)
        clip_multiplier = (1.0 
                          + 0.05 * (A_eff - 0.5)    # Novelty → larger updates
                          + 0.04 * (D_eff - 0.5)    # Progress → larger updates
                          - 0.06 * (C_eff - 0.5))   # Stress → smaller updates
        clip = self.clip0 * clip_multiplier
        
        # Apply KL-aware scaling
        kl_scale = self._get_kl_scale()
        if kl_scale != 1.0:
            ent = self.ent0 + (ent - self.ent0) * kl_scale
            lr = self.lr0 + (lr - self.lr0) * kl_scale
            clip = self.clip0 + (clip - self.clip0) * kl_scale
            
        # Apply safety clamping
        ent = np.clip(ent, self.clamp_ent[0], self.clamp_ent[1])
        lr = np.clip(lr, self.clamp_lr[0], self.clamp_lr[1])

        absolute_lr_floor = 2.5e-4
        if lr < absolute_lr_floor:
            lr = absolute_lr_floor
            if self.verbose and self.rollout_idx % 10 == 0:
                print(f"⚠️ LR at safety floor! C={self.C:.3f}")

        clip = np.clip(clip, self.clamp_clip[0], self.clamp_clip[1])
        
        # Update model hyperparameters
        self.model.ent_coef = ent
        self.model.lr_schedule = get_schedule_fn(lr)
        
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.model.clip_range = get_schedule_fn(clip)
        if hasattr(self.model, 'clip_range_vf'):
            self.model.clip_range_vf = get_schedule_fn(clip)
            
        self._log_state()

    def _get_kl_scale(self) -> float:
        """Get KL-based scaling factor for safety."""
        if self.target_kl is None:
            return 1.0
            
        try:
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                kl = self.model.logger.name_to_value.get('train/approx_kl', None)
                if kl is not None:
                    self._last_approx_kl = float(kl)
        except:
            pass
            
        if self._last_approx_kl is None or self._last_approx_kl <= 0:
            return 1.0
            
        ratio = self.target_kl / self._last_approx_kl
        return np.clip(ratio, 0.5, 2.0)
        
    def _log_state(self):
        """Log all relevant metrics."""
        self.logger.record("hormones/A", self.A)
        self.logger.record("hormones/C", self.C)
        self.logger.record("hormones/D", self.D)
        
        self.logger.record("hormones/pulse_A", self._last_pulses.get('A', 0))
        self.logger.record("hormones/pulse_C", self._last_pulses.get('C', 0))
        self.logger.record("hormones/pulse_D", self._last_pulses.get('D', 0))
        
        for key, value in self._last_signals.items():
            self.logger.record(f"signals/{key}", value)
            
        if self.progress_tracker:
            signals = self.progress_tracker.get_progress_signal()
            for key, value in signals.items():
                self.logger.record(f"progress/{key}", value)
                
        if self.novelty_detector:
            self.logger.record("novelty/baseline", self.novelty_detector.novelty_baseline or 0.0)
            self.logger.record("novelty/count", self.novelty_detector.count)
            
        if self.hormone_coupler:
            for hormone in ['A', 'C', 'D']:
                sensitivity = self.hormone_coupler.receptor_sensitivity.get(hormone, 1.0)
                self.logger.record(f"coupling/sensitivity_{hormone}", sensitivity)
                trend = self.hormone_coupler._get_trend(hormone)
                self.logger.record(f"coupling/trend_{hormone}", trend)
                
        self.logger.record("hparams/entropy_coef", self.model.ent_coef)
        self.logger.record("hparams/learning_rate", 
                          self.model.policy.optimizer.param_groups[0]['lr'])
        self.logger.record("hparams/clip_range", 
                          schedule_value(self.model.clip_range, 1.0))
        
        if self._last_approx_kl is not None:
            self.logger.record("guards/approx_kl", self._last_approx_kl)
            self.logger.record("guards/kl_scale", self._get_kl_scale())


# ============================================================================
# MAIN TRAINING SCRIPT - CONTINUOUS CONTROL ONLY
# ============================================================================

if __name__ == "__main__":

    # ---------- Config ----------
    PROJECT = "hormonal-rl"
    ENV_ID  = "Ant-v5"  # "Ant-v5" or "BipedalWalker-v3"
    ALGO    = "PPO"
    VARIANT = "hormones"
    SEED    = 42

    # Environment-specific settings
    ENV_CONFIGS = {
        "Ant-v5": {
            "total_timesteps": 1_000_000,
            "n_envs": 8,
            "eval_freq": 25_000,
        },
        "BipedalWalker-v3": {
            "total_timesteps": 1_000_000,
            "n_envs": 16,
            "eval_freq": 25_000,
        },
    }

    env_config = ENV_CONFIGS[ENV_ID]
    TOTAL_TIMESTEPS = env_config["total_timesteps"]
    N_ENVS = env_config["n_envs"]
    EVAL_FREQ = env_config["eval_freq"]
    N_EVAL_EPISODES = 10

    # Hyperparameters for continuous control
    hyperparameters = {
        "Ant-v5": dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,  # Continuous control - no entropy bonus
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            )
        ),
        "BipedalWalker-v3": dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,  # High gamma for long episodes
            gae_lambda=0.95,
            clip_range=0.18,  # Tighter clip (SB3 Zoo tuned)
            ent_coef=0.0,  # Continuous control
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            )
        ),
    }

    ppo_kwargs = hyperparameters[ENV_ID]
    
    # ---------- W&B init ----------
    try:
        wandb.login()
    except Exception:
        pass

    ctx = wandb_context(
        env_id=ENV_ID,
        algo=ALGO,
        variant=VARIANT,
        seed=SEED,
        hormone_enabled=True,
        project=PROJECT,
    )

    wandb_run = wandb.init(
        **ctx,
        config=ppo_kwargs | {
            "env_id": ENV_ID,
            "seed": SEED,
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_envs": N_ENVS,
            
            # Hormone parameters
            "warmup_rollouts": 20,
            "base_A": 0.5, "base_C": 0.5, "base_D": 0.5,
            
            "T12_A": 12.0, "T12_C": 8.0, "T12_D": 15.0,
            "k_A": 0.10, "k_C": 0.08, "k_D": 0.10,
            "th_A": 0.30, "th_C": 0.40, "th_D": 0.20,
            
            "ent0": 0.0, "lr0": 3e-4, "clip0": 0.2,
            "clamp_ent": (0.0, 0.0),
            "clamp_lr": (2.5e-4, 3.5e-4),
            "clamp_clip": (0.18, 0.22),
            
            "use_A": False, "use_C": False, "use_D": True,
        },
        save_code=True,
        sync_tensorboard=True,
    )

    # ---------- Environments with VecNormalize ----------
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    vec_env = VecMonitor(vec_env)

    # CRITICAL: VecNormalize for continuous control
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=ppo_kwargs["gamma"],
    )

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED+1)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=ppo_kwargs["gamma"],
        training=False,
    )

    # ---------- Model ----------
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=SEED,
        verbose=1,
        tensorboard_log="tb",
        **ppo_kwargs,
    )

    # ---------- Callbacks ----------
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints",
        log_path="eval_logs",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path="checkpoints",
        model_save_freq=EVAL_FREQ,
        log="all",
        verbose=2,
    )

    horm_cb = HormonePPOCallback(
        warmup_rollouts=20,
        base_A=0.5, base_C=0.5, base_D=0.5,
        ent0=ppo_kwargs["ent_coef"], lr0=ppo_kwargs["learning_rate"], clip0=ppo_kwargs["clip_range"],
        
        clamp_ent=(0.0, 0.0),
        clamp_lr=(2.5e-4, 3.5e-4),
        clamp_clip=(0.18, 0.22),
        
        use_A=True, use_C=True, use_D=True,
        
        T12_A=12.0, T12_C=8.0, T12_D=15.0,
        k_A=0.10, k_C=0.08, k_D=0.10,
        th_A=0.30, th_C=0.40, th_D=0.20,
        
        verbose=1,
    )

    callbacks = CallbackList([eval_cb, wandb_cb, horm_cb])

    # ---------- Train ----------
    print(f"\n{'='*60}")
    print(f"Training {ENV_ID}")
    print(f"Variant: {VARIANT}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=1,
        tb_log_name=f"{ENV_ID}-{VARIANT}-seed{SEED}",
    )

    # ---------- Save model and normalization ----------
    final_path = "checkpoints/final_model.zip"
    model.save(final_path)
    
    vec_env.save("checkpoints/vec_normalize.pkl")
    print(f"[INFO] Saved VecNormalize stats")

    wandb.save(final_path)
    wandb.save("checkpoints/vec_normalize.pkl")
    
    artifact = wandb.Artifact(f"ppo_{ENV_ID.lower().replace('-', '_')}_model", type="model")
    artifact.add_file(final_path)
    artifact.add_file("checkpoints/vec_normalize.pkl")
    wandb.log_artifact(artifact)

    # ---------- Final evaluation ----------
    print(f"\n{'='*60}")
    print(f"Running final evaluation ({N_EVAL_EPISODES} episodes)...")
    print(f"{'='*60}\n")

    returns, lengths = [], []
    
    for i in range(N_EVAL_EPISODES):
        obs = eval_env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            if done[0]:
                if 'episode' in info[0]:
                    ep_ret = info[0]['episode']['r']
                    ep_len = info[0]['episode']['l']
                else:
                    ep_ret += float(reward[0])
                    ep_len += 1
                break
            
            ep_ret += float(reward[0])
            ep_len += 1
        
        returns.append(ep_ret)
        lengths.append(ep_len)

    print(f"\nFinal Evaluation Results:")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Median Return: {np.median(returns):.2f}")
    print(f"  Min/Max: {np.min(returns):.2f} / {np.max(returns):.2f}")

    wandb.log({
        "final_eval/mean_return": float(np.mean(returns)),
        "final_eval/std_return": float(np.std(returns)),
        "final_eval/median_return": float(np.median(returns)),
        "final_eval/min_return": float(np.min(returns)),
        "final_eval/max_return": float(np.max(returns)),
        "final_eval/mean_ep_len": float(np.mean(lengths)),
        "final_eval/episodes": wandb.Table(
            columns=["episode", "return", "length"],
            data=[[i, float(r), int(l)] for i, (r, l) in enumerate(zip(returns, lengths))]
        )
    })

    wandb.finish()
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}\n")