import os
import sys
import gymnasium as gym
import numpy as np
import wandb
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import wandb_context

# ---------- Config ----------
PROJECT = "hormonal-rl"   
ENV_ID  = "LunarLander-v3" # "CartPole-v1"  "LunarLander-v3"
ALGO    = "PPO"           
VARIANT = "baseline"      
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
    wandb.login()  # no-op if already logged in
except Exception:
    pass

ctx = wandb_context(
    env_id=ENV_ID,
    algo=ALGO,
    variant=VARIANT,
    seed=SEED,
    hormone_enabled=False,
    project=PROJECT,
)

wandb_run = wandb.init(
    **ctx,
    config=ppo_kwargs | {"env_id": ENV_ID, "seed": SEED, "total_timesteps": TOTAL_TIMESTEPS},
    save_code=True,
    sync_tensorboard=True,
)

# ---------- Training env (vectorized + monitor) ----------
# make_vec_env adds Monitor; VecMonitor ensures ep stats aggregation for logging
vec_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED)
vec_env = VecMonitor(vec_env)

# ---------- Evaluation env (non-vector for clean eval) ----------
# eval_env = gym.make(ENV_ID)
# eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env) 
# eval_env.reset(seed=SEED + 1)
eval_env = VecMonitor(make_vec_env(ENV_ID, n_envs=1, seed=SEED+1))

# ---------- Model ----------
model = PPO(
    "MlpPolicy",
    vec_env,
    seed=SEED,
    verbose=1,
    tensorboard_log="tb", 
    **ppo_kwargs,
)

# ---------- Callbacks: periodic eval + W&B sync + model saving ----------
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
    gradient_save_freq=0,           # set >0 to log grads occasionally (costly)
    model_save_path="checkpoints",  # logs checkpoints as W&B artifacts
    model_save_freq=EVAL_FREQ,      # align with eval frequency
    log="all",                      # log hyperparams, metrics, etc.
    verbose=2,
)

callbacks = CallbackList([eval_cb, wandb_cb])

# ---------- Train ----------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=callbacks,
    log_interval=1,                    # log every rollout
    tb_log_name=f"{ENV_ID}-{VARIANT}-seed{SEED}",  # ensures TB subdir is created
    )

# ---------- Save + log final model ----------
final_path = "checkpoints/final_model.zip"
model.save(final_path)
wandb.save(final_path)  # uploads as a run file
artifact = wandb.Artifact("ppo_cartpole_model", type="model")
artifact.add_file(final_path)
wandb.log_artifact(artifact)

# ---------- Final evaluation summary (table + scalars) ----------
# Run a clean eval to log a summary to W&B
final_env = gym.make(ENV_ID)
final_env = gym.wrappers.RecordEpisodeStatistics(final_env)

returns = []
lengths = []
for i in range(N_EVAL_EPISODES):
    obs, info = final_env.reset(seed=SEED + 1 + i)
    done = False
    ep_ret, ep_len = 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Ensure action is a scalar (not 0-dimensional array)
        action = int(action) if isinstance(action, (np.ndarray, np.generic)) else action
        obs, reward, terminated, truncated, info = final_env.step(action)
        ep_ret += float(reward)
        ep_len += 1
        done = terminated or truncated
    returns.append(ep_ret)
    lengths.append(ep_len)

wandb.log({
    "final_eval/mean_return": float(np.mean(returns)),
    "final_eval/std_return": float(np.std(returns)),
    "final_eval/median_return": float(np.median(returns)),
    "final_eval/mean_ep_len": float(np.mean(lengths)),
})

# Optional: log per-episode table
wandb.log({
    "final_eval/episodes": wandb.Table(
        columns=["episode", "return", "length"],
        data=[[i, float(r), int(l)] for i, (r, l) in enumerate(zip(returns, lengths))]
    )
})

final_env.close()
wandb.finish()
