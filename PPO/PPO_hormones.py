import os
import gymnasium as gym
import numpy as np
import math
import wandb
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from wandb.integration.sb3 import WandbCallback

from wandb_utils import wandb_context

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
    def __init__(self, warmup_rollouts=5, verbose=0,
                 beta_A=0.6, beta_C=0.35, beta_D=0.35,
                 base_A=0.5, base_C=0.5, base_D=0.5,
                 homeo_A=0.05, homeo_C=0.04, homeo_D=0.03,
                 ent0=0.01, lr0=3e-4, clip0=0.2,
                 clamp_ent=(1e-4, 0.2), clamp_lr=(2e-4, 3e-4), clamp_clip=(0.18, 0.3)):

        super().__init__(verbose)
        self.rollout_idx = 0  # count rollouts

        self.warmup_rollouts = warmup_rollouts   

        # running EMAs of raw signals
        self.ema_adv = None
        self.ema_td = None
        self.ema_vol = None

        # frozen references (set after warmup)
        self.ref_adv = None
        self.ref_td = None
        self.ref_vol = None

        self.A_inst = 0.0
        self.C_inst = 0.0
        self.D_inst = 0.0

        self.A = base_A; self.C = base_C; self.D = base_D
        self.beta_A, self.beta_C, self.beta_D = beta_A, beta_C, beta_D
        self.base_A, self.base_C, self.base_D = base_A, base_C, base_D
        self.homeo_A, self.homeo_C, self.homeo_D = homeo_A, homeo_C, homeo_D
        
        self.ent0, self.lr0, self.clip0 = ent0, lr0, clip0
        self.clamp_ent = clamp_ent
        self.clamp_lr  = clamp_lr
        self.clamp_clip = clamp_clip

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
        # Called before the next rollout is collected
        if self.verbose > 1:
            print(f"[HormonePPO] Rollout {self.rollout_idx}: start.")
        # (Later: apply hormone-modulated hyperparams here)
        self._apply_modulation()

    def _on_rollout_end(self) -> None:
        # Called after rollout collection and advantage computation
        adv_mag, td_mag, ret_vol = self._collect_signals()
        norm = self._normalize_signals(adv_mag, td_mag, ret_vol)
        if norm is not None:
            adv_hat, td_hat, vol_hat = norm
            self._compute_instant_hormones(adv_hat, td_hat, vol_hat)
            self._decay_toward_targets()
        self.rollout_idx += 1
        # if norm is None:
        #     print(f"[Step4] Warmup rollout {self.rollout_idx}: refs not set yet.")
        # else:
        #     a_hat, t_hat, v_hat = norm
        #     print(f"[Step4] Norm hats: adv={a_hat:.3f}, td={t_hat:.3f}, vol={v_hat:.3f}")


    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("[HormonePPO] Training finished.")

    def _collect_signals(self):
        buf = self.model.rollout_buffer

        # 1) Advantages / TD error
        adv = buf.advantages.flatten()
        rets = buf.returns.flatten()
        vals = buf.values.flatten()
        td   = rets - vals

        adv_mag = float(np.mean(np.abs(adv)))
        td_mag  = float(np.mean(np.abs(td)))

        # 2) Return volatility from recent episodes (deque of dicts: {'r': return, 'l': length})
        epi = list(self.model.ep_info_buffer)  # may be empty early on
        if len(epi) >= 5:
            ep_returns = np.array([e['r'] for e in epi], dtype=np.float32)
            ret_vol = float(np.std(ep_returns))
        else:
            # Fallback: per-env return proxy from this rollout
            # (less accurate, but avoids 'NaN' early in training)
            n_envs = getattr(self.model, "n_envs", 1)
            # Approx: sum of rewards per env slice
            # Note: rollout_buffer.rewards shape is (n_steps, n_envs)
            rewards = buf.rewards  # shape [n_steps, n_envs]
            ep_returns = np.sum(rewards, axis=0)
            ret_vol = float(np.std(ep_returns))

        return adv_mag, td_mag, ret_vol

    
    def _normalize_signals(self, adv_mag, td_mag, ret_vol):
        # Update EMAs
        self.ema_adv = EMA(self.ema_adv, adv_mag, 0.2)
        self.ema_td  = EMA(self.ema_td,  td_mag,  0.2)
        self.ema_vol = EMA(self.ema_vol, ret_vol, 0.2)

        # During warmup, do not normalize (and don't modulate yet)
        if self.rollout_idx < self.warmup_rollouts:
            return None  # means "not ready"

        # Freeze references once when leaving warmup
        if self.ref_adv is None:
            self.ref_adv, self.ref_td, self.ref_vol = self.ema_adv, self.ema_td, self.ema_vol

        eps = 1e-8
        adv_hat = float(self.ema_adv / (self.ref_adv + eps))
        td_hat  = float(self.ema_td  / (self.ref_td  + eps))
        vol_hat = float(self.ema_vol / (self.ref_vol + eps))
        return adv_hat, td_hat, vol_hat
    
    def _compute_instant_hormones(self, adv_hat, td_hat, vol_hat):
        # 0) Ensure hats are finite
        adv_hat = float(np.nan_to_num(adv_hat, nan=1.0, posinf=2.0, neginf=0.5))
        td_hat  = float(np.nan_to_num(td_hat,  nan=1.0, posinf=2.0, neginf=0.5))
        vol_hat = float(np.nan_to_num(vol_hat, nan=1.0, posinf=2.0, neginf=0.5))

        # 1) Adrenaline: surprise -> explore
        A = squash(adv_hat)  # more than baseline -> closer to 1

        # 2) Cortisol: volatility -> caution
        C = squash(vol_hat)

        # 3) Dopamine: progress -> confidence (td_hat < 1 == improving)
        td_trend_down = clamp(1.0 - td_hat, 0.0, 1.0)
        D = squash(0.8 * td_trend_down + 0.2 * 1.0)  # add a small bias toward neutral

        self.A_inst, self.C_inst, self.D_inst = A, C, D

    def _decay_toward_targets(self):
        # Move toward instantaneous targets
        self.A = (1 - self.beta_A) * self.A + self.beta_A * self.A_inst
        self.C = (1 - self.beta_C) * self.C + self.beta_C * self.C_inst
        self.D = (1 - self.beta_D) * self.D + self.beta_D * self.D_inst
        # Homeostatic drift back to base
        self.A -= self.homeo_A * (self.A - self.base_A)
        self.C -= self.homeo_C * (self.C - self.base_C)
        self.D -= self.homeo_D * (self.D - self.base_D)
        # Final clamp (just in case)
        self.A = clamp(self.A, 0.0, 1.0)
        self.C = clamp(self.C, 0.0, 1.0)
        self.D = clamp(self.D, 0.0, 1.0)

        # print(f"[Step6] A={self.A:.3f} C={self.C:.3f} D={self.D:.3f} (decayed)")

    def _apply_modulation(self):
        # Skip during warmup (no refs)
        if self.rollout_idx <= self.warmup_rollouts or self.ref_adv is None:
            return

        # Multipliers (small, safe)
        ent = self.ent0 * (1.0 + 0.5 * self.A)               # up to +50% with high A
        lr  = self.lr0  / (1.0 + 0.3 * self.C)               # down with C
        lr  = lr * (1.0 + 0.35 * self.D)                     # up a bit with D
        clip = self.clip0 * (1.0 - 0.15 * self.C)            # shrink with C

        ent  = clamp(ent,  *self.clamp_ent)
        lr   = clamp(lr,   *self.clamp_lr)
        clip = clamp(clip, *self.clamp_clip)

        # --- Apply to PPO ---
        # 1) Entropy coef is fine as a float in PPO (no schedule expected)
        self.model.ent_coef = ent

        # 2) Learning rate: SB3 updates LR each train step via lr_schedule(progress)
        #    If we only change optimizer param groups, SB3 will overwrite it soon.
        #    So: also update the schedule and immediately apply it.
        self.model.lr_schedule = get_schedule_fn(lr)  # or get_schedule_fn(lr)   
        for g in self.model.policy.optimizer.param_groups:
            g["lr"] = lr
        # Optionally force SB3 to re-apply schedule logic now:
        try:
            self.model._update_learning_rate(self.model.policy.optimizer)
        except Exception:
            pass  # older/newer versions are slightly different; the line above is best-effort

        # 3) Clip range: MUST be a callable schedule
        self.model.clip_range = get_schedule_fn(clip)  # or get_schedule_fn(clip)

        # (Optional) If you're using value function clipping too:
        if getattr(self.model, "clip_range_vf", None) is not None:
            # keep vf clip proportional to policy clip, or set explicitly
            self.model.clip_range_vf = get_schedule_fn(clip)  # or another value

        # Logging unchanged
        self.logger.record("hormones/A", self.A)
        self.logger.record("hormones/C", self.C)
        self.logger.record("hormones/D", self.D)
        self.logger.record("hparams/entropy_coef", ent)
        self.logger.record("hparams/lr", lr)
        self.logger.record("hparams/clip_range", clip)

        # --- 4) Debug print (schedule-aware) -------------------------------------
        progress = getattr(self.model, "_current_progress_remaining", 1.0)
        ent_now  = float(self.model.ent_coef)
        lr_now   = float(self.model.policy.optimizer.param_groups[0]["lr"])
        clip_now = schedule_value(self.model.clip_range, progress)

        print(f"[Step7] ent={ent_now:.5f} lr={lr_now:.6f} clip={clip_now:.3f}")

        lr_sched_now = schedule_value(self.model.lr_schedule, progress)
        print(f"[Step7] lr_sched={lr_sched_now:.6f} lr_opt={lr_now:.6f}")
        


        