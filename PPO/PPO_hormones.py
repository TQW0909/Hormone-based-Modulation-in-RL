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
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import wandb_context

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


class HormonePPOCallback(BaseCallback):
    def __init__(
        self,
        warmup_rollouts: int = 5,
        verbose: int = 0,
        # Base hormone levels
        base_A: float = 0.5,
        base_C: float = 0.5,
        base_D: float = 0.5,
        # Initial hyperparameters
        ent0: float = 0.01,
        lr0: float = 3e-4,
        clip0: float = 0.2,
        # Clamping ranges
        clamp_ent: Tuple[float, float] = (0.008, 0.012), # (0.005, 0.015) for CartPole, (0.001, 0.005) for LunarLander
        clamp_lr: Tuple[float, float] = (2.8e-4, 3.2e-4), # (2.5e-4, 3.5e-4) for CartPole, (2.8e-4, 3.2e-4) for LunarLander
        clamp_clip: Tuple[float, float] = (0.195, 0.205), # (0.18, 0.3) for CartPole, (0.195, 0.205) for LunarLander
        # Feature flags
        use_A: bool = True,
        use_C: bool = True,
        use_D: bool = True,
        # Pulse-decay parameters
        T12_A: float = 4.0,  # Adrenaline half-life (rollouts)
        T12_C: float = 10.0,  # Cortisol half-life
        T12_D: float = 6.0,  # Dopamine half-life
        k_A: float = 0.15,   # Adrenaline pulse gain
        k_C: float = 0.10,   # Cortisol pulse gain
        k_D: float = 0.10,   # Dopamine pulse gain (0.20 for CartPole, 0.10 for LunarLander)
        # Thresholds for hormone activation
        th_A: float = 0.20,  # Adrenaline threshold (above baseline)
        th_C: float = 0.30,  # Cortisol threshold
        th_D: float = 0.20,  # Dopamine threshold (0.10 for CartPole, 0.20 for LunarLander)
    ):
        super().__init__(verbose)

        # Core parameters
        self.rollout_idx = 0  # count rollouts
        self.warmup_rollouts = warmup_rollouts   

        # Current hormone levels
        self.A = base_A
        self.C = base_C
        self.D = base_D
        self.base_A = base_A
        self.base_C = base_C
        self.base_D = base_D

        # Hyperparameter settings
        self.ent0 = ent0
        self.lr0 = lr0
        self.clip0 = clip0
        self.clamp_ent = clamp_ent
        self.clamp_lr = clamp_lr
        self.clamp_clip = clamp_clip

        # Feature flags
        self.use_A = use_A
        self.use_C = use_C
        self.use_D = use_D

       # Initialize improved components
        self.novelty_detector = None  # Will be initialized when we know obs_dim
        self.hormone_coupler = AdvancedHormoneCoupler()
        self.progress_tracker = MultiScaleProgressTracker()
        
        # Pulse-decay dynamics
        self.H0_A = base_A  # Homeostatic center for A
        self.H0_C = base_C  # Homeostatic center for C  
        self.H0_D = base_D  # Homeostatic center for D
        
        # Half-lives and decay rates
        self.T12 = {'A': T12_A, 'C': T12_C, 'D': T12_D}
        self.lam = {h: np.log(2) / t12 for h, t12 in self.T12.items()}

        # Pulse parameters
        self.k = {'A': k_A, 'C': k_C, 'D': k_D}
        self.th = {'A': th_A, 'C': th_C, 'D': th_D}

        # Refractory periods
        self.refrac = {'A': 0, 'C': 0, 'D': 0}
        self.refrac_len = {'A': 1, 'C': 2, 'D': 1}
        
        # Signal tracking
        self.ema_adv = None
        self.ema_td = None
        self.ema_vol = None  # For volatility baseline
        self.ref_adv = None
        self.ref_td = None
        
        # Reward tracking (for dopamine)
        self.reward_window = deque(maxlen=200)
        self.ema_reward_fast = None
        self.ema_reward_slow = None
        
        # TD tracking for improvement signals
        self.best_td = None
        self.td_slow = None

        # Signal smoothing
        self.signal_history = {}
        self.signal_smooth_alpha = 0.5  # Lower = more smoothing (0.3 for CartPole, 0.5 for LunarLander)
        
        # Hormone momentum
        self.hormone_momentum = 0.7  # Higher = more inertia
        self.prev_A = self.A
        self.prev_C = self.C
        self.prev_D = self.D
        
        # Additional tracking
        self._last_episode_rewards = []
        self._last_approx_kl = None
        self.target_kl = None
        
        # For logging
        self._last_signals = {}
        self._last_pulses = {'A': 0, 'C': 0, 'D': 0}


    def _init_callback(self) -> None:
        """Initialize callback with model."""
        super()._init_callback()
        
        # Get observation dimension for novelty detector
        if self.novelty_detector is None:
            obs_shape = self.model.observation_space.shape
            obs_dim = np.prod(obs_shape)
            self.novelty_detector = EfficientNoveltyDetector(obs_dim)
            if self.verbose > 0:
                print(f"[HormonePPO] Initialized efficient novelty detector for obs_dim={obs_dim}")
            
        # Get target KL if available
        self.target_kl = getattr(self.model, "target_kl", None)

    def _on_step(self) -> bool:
        """
        Called at every environment step during rollout collection and/or after each training iteration.
        We don't need per-step logic, but we must return True to continue training.
        Return False to early-stop training.
        """
        return True

    def _on_training_start(self) -> None:
        # Called once when .learn() begins
        if self.verbose > 0:
            print("[HormonePPO] Training started.")

    def _on_rollout_start(self) -> None:
        """Apply hormone-modulated hyperparameters before rollout."""
        if self.verbose > 1:
            print(f"[HormonePPO] Rollout {self.rollout_idx}: start")
        self._apply_modulation()

    def _on_rollout_end(self) -> None:
        """Process rollout data and update hormones."""
        
        # Collect signals from rollout
        signals = self._collect_signals()
        
        # Update progress tracker
        if self.progress_tracker is not None:
            self._update_progress_tracker()
            
        # Normalize signals (skip during warmup)
        if self.rollout_idx >= self.warmup_rollouts:
            normalized = self._normalize_signals(signals)
            
            # Update hormones
            self._update_hormones(normalized)
            
        self.rollout_idx += 1

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("[HormonePPO] Training finished.")


    def _collect_signals(self) -> Dict[str, float]:
        """Collect all signals with improvements."""
        buf = self.model.rollout_buffer
        signals = {}
        
        # Basic signals
        adv = buf.advantages.flatten()
        rets = buf.returns.flatten()
        vals = buf.values.flatten()
        td = rets - vals
        
        signals['adv_magnitude'] = float(np.mean(np.abs(adv)))
        signals['td_error'] = float(np.mean(np.abs(td)))

        # TD volatility (MAD - median absolute deviation)
        td_med = float(np.median(td)) if td.size > 0 else 0.0
        signals['td_mad'] = float(np.median(np.abs(td - td_med))) if td.size > 0 else 0.0
        
        # Novelty signal (efficient or fallback)
        if self.novelty_detector is not None:
            obs = buf.observations.reshape((-1,) + buf.observations.shape[2:])
            # Subsample for efficiency
            if len(obs) > 256:
                idx = np.random.choice(len(obs), 256, replace=False)
                obs = obs[idx]
            signals['novelty'] = self.novelty_detector.update(obs)
        else:
            # Simple fallback: use advantage variance as proxy for novelty
            signals['novelty'] = 1.0 + 0.5 * np.std(adv) / (np.mean(np.abs(adv)) + 1e-8)
            
        # Episode statistics for reward tracking
        episodes = list(self.model.ep_info_buffer)
        rewards = [float(e['r']) for e in episodes if 'r' in e]
        signals['mean_reward'] = np.mean(rewards) if rewards else 0.0
        self._last_episode_rewards = rewards
        
        # Update reward window for trend calculation
        for r in rewards:
            self.reward_window.append(r)

    
        # Smooth each signal
        for key, value in signals.items():
            if key not in self.signal_history:
                self.signal_history[key] = value
            else:
                # Exponential moving average
                self.signal_history[key] = (self.signal_smooth_alpha * value + 
                                        (1 - self.signal_smooth_alpha) * self.signal_history[key])
            signals[key] = self.signal_history[key]
        
        return signals


    def _update_progress_tracker(self):
        """Update multi-scale progress tracking."""
        if not self.progress_tracker:
            return
            
        buf = self.model.rollout_buffer
        
        # Get required data
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
        
        # Update EMAs for advantage and TD error
        alpha = 0.2
        if self.ema_adv is None:
            self.ema_adv = signals['adv_magnitude']
            self.ema_td = signals['td_error']
            self.ema_vol = signals['td_mad']
        else:
            self.ema_adv = alpha * signals['adv_magnitude'] + (1 - alpha) * self.ema_adv
            self.ema_td = alpha * signals['td_error'] + (1 - alpha) * self.ema_td
            self.ema_vol = alpha * signals['td_mad'] + (1 - alpha) * self.ema_vol
            
        # Set references after warmup
        if self.ref_adv is None:
            self.ref_adv = self.ema_adv
            self.ref_td = self.ema_td
            self.best_td = self.ema_td
            self.td_slow = self.ema_td
            
        # Update TD tracking for improvement signals
        self.td_slow = 0.95 * self.td_slow + 0.05 * self.ema_td if self.td_slow else self.ema_td
        self.best_td = min(self.best_td, self.ema_td) if self.best_td else self.ema_td
        
        # Update reward EMAs for trend
        if len(self.reward_window) >= 5:
            current_reward = np.mean(list(self.reward_window)[-5:])
            self.ema_reward_fast = 0.3 * current_reward + 0.7 * self.ema_reward_fast \
                if self.ema_reward_fast else current_reward
            self.ema_reward_slow = 0.05 * current_reward + 0.95 * self.ema_reward_slow \
                if self.ema_reward_slow else current_reward
            
        # Calculate normalized signals (hat values)
        normalized = {
            'adv_hat': self.ema_adv / (self.ref_adv + 1e-8),
            'td_hat': self.ema_td / (self.ref_td + 1e-8),
            'vol_hat': signals['td_mad'] / (self.ema_vol + 1e-8),
            'nov_hat': signals['novelty'],  # Already normalized by detector
        }
        
        # Calculate progress signals
        if self.progress_tracker:
            progress_signals = self.progress_tracker.get_progress_signal()
            normalized['progress'] = progress_signals.get('combined_progress', 0.0)
        else:
            # Simple progress based on TD improvement and reward trend
            td_improvement = max(0, (self.td_slow - self.ema_td) / (self.td_slow + 1e-8))
            reward_trend = 0.0
            if self.ema_reward_fast and self.ema_reward_slow:
                reward_trend = np.clip((self.ema_reward_fast - self.ema_reward_slow) / 
                                      (abs(self.ema_reward_slow) + 1e-8), -1, 1)
            normalized['progress'] = 0.6 * td_improvement + 0.4 * max(0, reward_trend)
            
        # Store for logging
        self._last_signals = normalized
            
        return normalized

    def _update_hormones(self, normalized: Dict[str, float]):
        """Update hormone levels with pulse-decay dynamics."""
        
        # Calculate pulses for each hormone
        pulses = self._calculate_pulses(normalized)
        self._last_pulses = pulses
        
        # Apply decay toward homeostatic centers and add pulses (only for enabled hormones)
        for hormone in ['A', 'C', 'D']:
            use_flag = getattr(self, f'use_{hormone}')
            
            if use_flag:
                # Get current level and homeostatic center
                current = getattr(self, hormone)
                center = getattr(self, f'H0_{hormone}')
                
                # Apply exponential decay toward center
                new_level = center + (current - center) * (1.0 - self.lam[hormone])
                
                # Add pulse
                new_level += pulses[hormone]
                
                # Clamp to [0, 1]
                new_level = np.clip(new_level, 0.0, 1.0)
                
                # Update level
                setattr(self, hormone, new_level)
                
                # Update refractory period if pulse was large
                if pulses[hormone] > 0.15:
                    self.refrac[hormone] = self.refrac_len[hormone]
                else:
                    self.refrac[hormone] = max(0, self.refrac[hormone] - 1)
            else:
                # Disabled hormone: maintain baseline value
                setattr(self, hormone, getattr(self, f'base_{hormone}'))
                
        # Apply advanced coupling if enabled (only affects enabled hormones)
        if self.hormone_coupler:
            # Calculate external stress from KL divergence if available
            external_stress = 0.0
            if self.target_kl and self._last_approx_kl:
                kl_ratio = self._last_approx_kl / (self.target_kl + 1e-8)
                if kl_ratio > 2.0:
                    external_stress = min(0.3, 0.1 * (kl_ratio - 2.0))
            
            # Only apply coupling if at least one hormone is enabled
            if self.use_A or self.use_C or self.use_D:
                # Use enabled hormones for coupling, disabled ones use 0.0 (no coupling effect)
                # This prevents disabled hormones from affecting enabled ones through coupling
                A_for_coupling = self.A if self.use_A else 0.0
                C_for_coupling = self.C if self.use_C else 0.0
                D_for_coupling = self.D if self.use_D else 0.0
                
                A_coupled, C_coupled, D_coupled = self.hormone_coupler.couple_hormones(
                    A_for_coupling, C_for_coupling, D_for_coupling, external_stress
                )
                
                # Only update enabled hormones with coupling results
                if self.use_A:
                    self.A = A_coupled
                if self.use_C:
                    self.C = C_coupled
                if self.use_D:
                    self.D = D_coupled
            
    def _calculate_pulses(self, normalized: Dict[str, float]) -> Dict[str, float]:
        """Calculate hormone pulses from normalized signals."""
        pulses = {}
        
        # Adrenaline pulse: novelty × advantage (exploration need)
        if self.use_A:
            # Signal combines novelty and advantage magnitude
            signal_A = normalized['adv_hat'] * normalized['nov_hat']
            # Pulse if signal exceeds threshold and not in refractory
            if self.refrac['A'] == 0 and signal_A > (1.0 + self.th['A']):
                pulses['A'] = self.k['A'] * (signal_A - 1.0)
            else:
                pulses['A'] = 0.0
        else:
            pulses['A'] = 0.0
            
        # Cortisol pulse: volatility/uncertainty (stress response)
        if self.use_C:
            # Signal is volatility (TD MAD)
            signal_C = normalized['vol_hat']
            # Pulse if high volatility and not in refractory
            if self.refrac['C'] == 0 and signal_C > (1.0 + self.th['C']):
                pulses['C'] = self.k['C'] * (signal_C - 1.0)
                # Reduce pulse if dopamine is high (confidence reduces stress)
                # Only use dopamine effect if dopamine is enabled
                if self.use_D:
                    pulses['C'] *= (1.0 - 0.3 * self.D)
            else:
                pulses['C'] = 0.0
        else:
            pulses['C'] = 0.0
            
        # Dopamine pulse: progress signal (reward/improvement)
        if self.use_D:
            # Main signal is progress (combines multiple metrics)
            signal_D = normalized['progress']
            # Additional boost from low TD error (good value predictions)
            td_bonus = max(0, 1.0 - normalized['td_hat']) * 0.3
            signal_D = signal_D + td_bonus
            
            
            if self.refrac['D'] == 0 and signal_D > self.th['D']:
                pulses['D'] = self.k['D'] * signal_D 
            else:
                pulses['D'] = 0.0
        else:
            pulses['D'] = 0.0
            
        return pulses
        
    def _apply_modulation(self):
        """Apply hormone-modulated hyperparameters."""
        
        if self.rollout_idx < self.warmup_rollouts:
            return
            
        # Get effective hormone levels (could be post-coupling)
        # Use baseline values for disabled hormones to ensure no modulation effect
        A_eff = self.A if self.use_A else self.base_A
        C_eff = self.C if self.use_C else self.base_C
        D_eff = self.D if self.use_D else self.base_D
        
        # Calculate hyperparameters with improved mapping
        # All formulas are centered around baseline (0.5) so that baseline values
        # produce multipliers of exactly 1.0 (no modulation)
        
        # Entropy: exploration (A) increases, exploitation (D) decreases
        # Map from centered hormones (deviation from 0.5) to multiplier
        ent_multiplier = 1.0 + 0.2 * (A_eff - 0.5) - 0.15 * (D_eff - 0.5)
        ent = self.ent0 * ent_multiplier
        
        # Learning rate: caution (C) decreases, confidence (D) increases
        # Centered around 0.5: when C=0.5 and D=0.5, multiplier = 1.0
        lr_multiplier = 1.0 - 0.1 * (C_eff - 0.5) + 0.08 * (D_eff - 0.5)
        lr = self.lr0 * lr_multiplier
        
        # Clip range: stability (C) tightens, exploration (A) loosens
        # Centered around 0.5: when C=0.5 and A=0.5, multiplier = 1.0
        clip_multiplier = 1.0 - 0.04 * (C_eff - 0.5) + 0.025 * (A_eff - 0.5)
        clip = self.clip0 * clip_multiplier
        
        # Apply KL-aware scaling if available
        kl_scale = self._get_kl_scale()
        if kl_scale != 1.0:
            # When KL is high, reduce adaptation strength
            ent = self.ent0 + (ent - self.ent0) * kl_scale
            lr = self.lr0 + (lr - self.lr0) * kl_scale
            clip = self.clip0 + (clip - self.clip0) * kl_scale
            
        # Apply safety clamping
        ent = np.clip(ent, self.clamp_ent[0], self.clamp_ent[1])
        lr = np.clip(lr, self.clamp_lr[0], self.clamp_lr[1])
        clip = np.clip(clip, self.clamp_clip[0], self.clamp_clip[1])
        
        # Ensure minimum entropy to prevent collapse
        ent_floor = max(1e-4, self.ent0 * 0.2)
        ent = max(ent, ent_floor)
        
        # Update model hyperparameters
        self.model.ent_coef = ent
        self.model.lr_schedule = get_schedule_fn(lr)
        
        # Update optimizer learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.model.clip_range = get_schedule_fn(clip)
        if hasattr(self.model, 'clip_range_vf'):
            self.model.clip_range_vf = get_schedule_fn(clip)
            
        # Log everything
        self._log_state()

    def _get_kl_scale(self) -> float:
        """Get KL-based scaling factor for safety."""
        if self.target_kl is None:
            return 1.0
            
        # Try to get latest KL from logger
        try:
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                kl = self.model.logger.name_to_value.get('train/approx_kl', None)
                if kl is not None:
                    self._last_approx_kl = float(kl)
        except:
            pass
            
        if self._last_approx_kl is None or self._last_approx_kl <= 0:
            return 1.0
            
        # Calculate scale based on how far we are from target
        ratio = self.target_kl / self._last_approx_kl
        # If KL is 2x target, scale = 0.5 (half strength)
        # If KL is at target, scale = 1.0 (full strength)
        # If KL is 0.5x target, scale = 2.0 (double strength, capped)
        return np.clip(ratio, 0.5, 2.0)
        
        
    def _log_state(self):
        """Log all relevant metrics."""
        # Basic hormone levels
        self.logger.record("hormones/A", self.A)
        self.logger.record("hormones/C", self.C)
        self.logger.record("hormones/D", self.D)
        
        # Pulses (spikes)
        self.logger.record("hormones/pulse_A", self._last_pulses.get('A', 0))
        self.logger.record("hormones/pulse_C", self._last_pulses.get('C', 0))
        self.logger.record("hormones/pulse_D", self._last_pulses.get('D', 0))
        
        # Normalized signals
        for key, value in self._last_signals.items():
            self.logger.record(f"signals/{key}", value)
            
        # Progress metrics
        if self.progress_tracker:
            signals = self.progress_tracker.get_progress_signal()
            for key, value in signals.items():
                self.logger.record(f"progress/{key}", value)
                
        # Novelty metrics
        if self.novelty_detector:
            self.logger.record("novelty/baseline", self.novelty_detector.novelty_baseline or 0.0)
            self.logger.record("novelty/count", self.novelty_detector.count)
            
        # Coupling effects
        if self.hormone_coupler:
            for hormone in ['A', 'C', 'D']:
                sensitivity = self.hormone_coupler.receptor_sensitivity.get(hormone, 1.0)
                self.logger.record(f"coupling/sensitivity_{hormone}", sensitivity)
                trend = self.hormone_coupler._get_trend(hormone)
                self.logger.record(f"coupling/trend_{hormone}", trend)
                
        # Hyperparameters
        self.logger.record("hparams/entropy_coef", self.model.ent_coef)
        self.logger.record("hparams/learning_rate", 
                          self.model.policy.optimizer.param_groups[0]['lr'])
        self.logger.record("hparams/clip_range", 
                          schedule_value(self.model.clip_range, 1.0))
        
        # KL monitoring
        if self._last_approx_kl is not None:
            self.logger.record("guards/approx_kl", self._last_approx_kl)
            self.logger.record("guards/kl_scale", self._get_kl_scale())

# Helper function for schedule values
def schedule_value(schedule, progress: float) -> float:
    """Get value from schedule or constant."""
    if callable(schedule):
        return float(schedule(progress))
    return float(schedule)


if __name__ == "__main__":

    # ---------- Config ----------
    PROJECT = "hormonal-rl"
    ENV_ID  = "LunarLander-v3"     # "CartPole-v1"  "LunarLander-v3"
    ALGO    = "PPO"
    VARIANT = "hormones:D_only"
    SEED    = 42
    TOTAL_TIMESTEPS = 250_000
    EVAL_FREQ = 10_000
    N_EVAL_EPISODES = 10

    hyperparameters = {
        "CartPole-v1": dict(
            learning_rate=3e-4,     # LR
            n_steps=2048,
            batch_size=2048,        # must divide (n_steps * n_envs); here it equals it
            n_epochs=10,            # epochs per update
            gamma=0.99,             # γ
            gae_lambda=0.95,        # λ (GAE)
            clip_range=0.2,         # ε
            ent_coef=0.01,          # entropy coef (note: >0 vs your previous 0.0)
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,         # e.g., set 0.03 if you want gentle early stopping
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),  # or swap to [128, 128]
                activation_fn=torch.nn.Tanh,              # tanh
            )
        ),
        "LunarLander-v3": dict (
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,           # Key for LunarLander
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            )
        )
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
            
            # Core hormone parameters
            "warmup_rollouts": 15,  # 5 for CartPole, 15 for LunarLander
            "base_A": 0.5, "base_C": 0.5, "base_D": 0.5,
            
            # Pulse-decay dynamics
            "T12_A": 2.0,  # Adrenaline half-life (rollouts)
            "T12_C": 6.0,  # Cortisol half-life
            "T12_D": 4.0,  # Dopamine half-life
            "k_A": 0.35,   # Adrenaline pulse gain
            "k_C": 0.18,   # Cortisol pulse gain  
            "k_D": 0.30,   # Dopamine pulse gain
            "th_A": 0.10,  # Adrenaline threshold
            "th_C": 0.20,  # Cortisol threshold
            "th_D": 0.00,  # Dopamine threshold
            
            # Hyperparameter settings
            "ent0": 0.01, "lr0": 3e-4, "clip0": 0.2,
            "clamp_ent": (1e-4, 0.2), 
            "clamp_lr": (2e-4, 3e-4), 
            "clamp_clip": (0.18, 0.3),
            
            # Feature flags for ablation studies
            "use_A": False, "use_C": False, "use_D": True,
        },
        save_code=True,
        sync_tensorboard=True,
    )

    # ---------- Envs ----------
    vec_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED)
    vec_env = VecMonitor(vec_env)

    # eval_env = gym.make(ENV_ID)
    # eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env = VecMonitor(make_vec_env(ENV_ID, n_envs=1, seed=SEED+1))
    # eval_env.reset(seed=SEED + 1)

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

    # horm_cb = HormonePPOCallback(
    #     warmup_rollouts=15,  # 5 for CartPole, 10 for LunarLander
    #     base_A=0.5,
    #     base_C=0.5,
    #     base_D=0.5,
    #     ent0=0.01,
    #     lr0=3e-4,
    #     clip0=0.2,
    #     verbose=1
    # )

    # horm_cb = HormonePPOCallback(
    #     warmup_rollouts=25,  # Let it learn baseline behavior first
        
    #     # Very long half-lives (slow changes)
    #     T12_A=15.0, T12_C=25.0, T12_D=20.0,
        
    #     # Minimal gains
    #     k_A=0.05, k_C=0.03, k_D=0.05,
        
    #     # High thresholds
    #     th_A=0.40, th_C=0.50, th_D=0.30,
        
    #     # Extremely tight clamps (almost fixed)
    #     clamp_ent=(0.0095, 0.0105),  # 0.01 ± 5%
    #     clamp_lr=(2.85e-4, 3.15e-4),  # 3e-4 ± 5%
    #     clamp_clip=(0.19, 0.21),
        
    #     base_A=0.5, base_C=0.5, base_D=0.5,
    #     ent0=0.01, lr0=3e-4, clip0=0.2,

    #     # Feature flags for ablation studies
    #     use_A=False, use_C=False, use_D=False,
    #     verbose=1
    # )

    horm_cb = HormonePPOCallback(
        # ========================================
        # DOPAMINE ONLY - Conservative Settings
        # ========================================
        
        use_A=False, 
        use_C=False, 
        use_D=True,  # Only test Dopamine
        
        # Base hormone levels
        base_A=0.5, 
        base_C=0.5, 
        base_D=0.5,
        
        # Initial hyperparameters
        ent0=0.01, 
        lr0=3e-4, 
        clip0=0.2,
        
        # ========================================
        # DOPAMINE PARAMETERS
        # ========================================
        
        # Warmup: Calibrate signals before modulating
        warmup_rollouts=20,  # ~80k steps to establish baseline
        
        # Half-life: How fast D decays back to baseline
        T12_D=15.0,  # 15 rollouts = ~60k steps
        # Longer half-life = D changes persist longer
        
        # Pulse gain: How much D increases when triggeresd
        k_D=0.10,  # Conservative (was 0.10-0.20 in your tests)
        # Smaller gain = gentler changes
        
        # Threshold: When to trigger D pulse
        th_D=0.20,  # Moderate (progress signal > 1.20 baseline)
        # Higher threshold = pulses less often (more selective)
        
        # Refractory period: Cooldown after pulse
        # (using default from __init__: refrac_len['D'] = 1 rollout)
        
        # ========================================
        # HYPERPARAMETER CLAMPING
        # ========================================
        
        # Dopamine modulates LEARNING RATE only
        clamp_lr=(2.7e-4, 3.3e-4),  # ±10% around baseline (3e-4)
        # When D high (progress) → LR increases (faster learning)
        # When D low (plateau) → LR decreases (more stable)
        
        # Keep these FIXED (D doesn't affect them)
        clamp_ent=(0.01, 0.01),     # Entropy fixed
        clamp_clip=(0.2, 0.2),      # Clip fixed
        
        verbose=1  # Print diagnostics
    )

    callbacks = CallbackList([eval_cb, wandb_cb, horm_cb])

    # ---------- Train ----------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=1,
        tb_log_name=f"{ENV_ID}-{VARIANT}-seed{SEED}",
    )

    # ---------- Save + log final model ----------
    final_path = "checkpoints/final_model.zip"
    model.save(final_path)
    wandb.save(final_path)
    artifact = wandb.Artifact("ppo_cartpole_model", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)

    # ---------- Final eval summary ----------
    final_env = gym.make(ENV_ID)
    final_env = gym.wrappers.RecordEpisodeStatistics(final_env)

    returns, lengths = [], []
    for i in range(N_EVAL_EPISODES):
        obs, info = final_env.reset(seed=SEED + 1 + i)
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Ensure action is a scalar (not 0-dimensional array)
            action = int(action) if isinstance(action, (np.ndarray, np.generic)) else action
            obs, reward, terminated, truncated, info = final_env.step(action)
            ep_ret += float(reward); ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret); lengths.append(ep_len)

    wandb.log({
        "final_eval/mean_return": float(np.mean(returns)),
        "final_eval/std_return": float(np.std(returns)),
        "final_eval/median_return": float(np.median(returns)),
        "final_eval/mean_ep_len": float(np.mean(lengths)),
        "final_eval/episodes": wandb.Table(
            columns=["episode", "return", "length"],
            data=[[i, float(r), int(l)] for i, (r, l) in enumerate(zip(returns, lengths))]
        )
    })

    final_env.close()
    wandb.finish()
