import os
import gymnasium as gym
import numpy as np
import wandb
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

from wandb_utils import wandb_context
from PPO_hormones import HormonePPOCallback

# ---------- Config ----------
PROJECT = "hormonal-rl"
ENV_ID  = "CartPole-v1"
ALGO    = "PPO"
VARIANT = "hormones"
SEED    = 42
TOTAL_TIMESTEPS = 250_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10

ppo_kwargs = dict(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    n_epochs=10,
    n_steps=2048,
    batch_size=2048,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh,
    ),
    max_grad_norm=0.5,
    vf_coef=0.5,
    target_kl=None,
)

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
        # hormone controller params to track
        "warmup_rollouts": 5,
        "beta_A": 0.6, "beta_C": 0.35, "beta_D": 0.35,
        "base_A": 0.5, "base_C": 0.5, "base_D": 0.5,
        "homeo_A": 0.05, "homeo_C": 0.04, "homeo_D": 0.03,
        "ent0": 0.01, "lr0": 3e-4, "clip0": 0.2,
        "clamp_ent": (1e-4, 0.2), "clamp_lr": (2e-4, 3e-4), "clamp_clip": (0.18, 0.3),
        "use_A": True, "use_C": True, "use_D": True,
    },
    save_code=True,
    sync_tensorboard=True,
)

# ---------- Envs ----------
vec_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED)
vec_env = VecMonitor(vec_env)

eval_env = gym.make(ENV_ID)
eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
eval_env.reset(seed=SEED + 1)

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
    warmup_rollouts=wandb.config.warmup_rollouts,
    beta_A=wandb.config.beta_A, beta_C=wandb.config.beta_C, beta_D=wandb.config.beta_D,
    base_A=wandb.config.base_A, base_C=wandb.config.base_C, base_D=wandb.config.base_D,
    homeo_A=wandb.config.homeo_A, homeo_C=wandb.config.homeo_C, homeo_D=wandb.config.homeo_D,
    ent0=wandb.config.ent0, lr0=wandb.config.lr0, clip0=wandb.config.clip0,
    clamp_ent=wandb.config.clamp_ent, clamp_lr=wandb.config.clamp_lr, clamp_clip=wandb.config.clamp_clip,
    verbose=0,
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
returns, lengths = [], []
for _ in range(N_EVAL_EPISODES):
    obs, info = eval_env.reset()
    done = False
    ep_ret, ep_len = 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
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

eval_env.close()
wandb.finish()
