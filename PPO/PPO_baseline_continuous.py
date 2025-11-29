import os
import sys
import gymnasium as gym
import numpy as np
import wandb
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import wandb_context

# ---------- Config ----------
PROJECT = "hormonal-rl"   
ENV_ID  = "BipedalWalker-v3"  # "Ant-v5" or "BipedalWalker-v3"
ALGO    = "PPO"           
VARIANT = "baseline"      
SEED    = 42

# Environment-specific settings
ENV_CONFIGS = {
    "Ant-v5": {
        "total_timesteps": 1_000_000,  # Quick: 500k-1M, Full: 2M
        "n_envs": 8,
        "eval_freq": 25_000,
    },
    "BipedalWalker-v3": {
        "total_timesteps": 1_000_000,  # Quick: 1M, Full: 5M
        "n_envs": 16,
        "eval_freq": 25_000,
    },
}

# Get config for current environment
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
    hormone_enabled=False,
    project=PROJECT,
)

wandb_run = wandb.init(
    **ctx,
    config=ppo_kwargs | {
        "env_id": ENV_ID,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_envs": N_ENVS,
    },
    save_code=True,
    sync_tensorboard=True,
)

# ---------- Training env (vectorized + monitor + normalization) ----------
vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
vec_env = VecMonitor(vec_env)

# VecNormalize is CRITICAL for continuous control
print(f"[INFO] Using VecNormalize for {ENV_ID}")
vec_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=ppo_kwargs["gamma"],
)

# ---------- Evaluation env ----------
eval_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED+1)
eval_env = VecMonitor(eval_env)
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,  # Don't normalize rewards during eval
    clip_obs=10.0,
    gamma=ppo_kwargs["gamma"],
    training=False,  # Don't update normalization stats during eval
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

callbacks = CallbackList([eval_cb, wandb_cb])

# ---------- Train ----------
print(f"\n{'='*60}")
print(f"Training {ENV_ID}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Parallel environments: {N_ENVS}")
print(f"Expected time: {TOTAL_TIMESTEPS / (N_ENVS * 2048) * 2:.1f}-{TOTAL_TIMESTEPS / (N_ENVS * 2048) * 3:.1f} minutes")
print(f"{'='*60}\n")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=callbacks,
    log_interval=1,
    tb_log_name=f"{ENV_ID}-{VARIANT}-seed{SEED}",
)

# ---------- Save model and normalization stats ----------
final_path = "checkpoints/final_model.zip"
model.save(final_path)

# Save VecNormalize stats (CRITICAL for loading model later)
vec_env.save("checkpoints/vec_normalize.pkl")
print(f"[INFO] Saved VecNormalize stats to checkpoints/vec_normalize.pkl")

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

returns = []
lengths = []

for i in range(N_EVAL_EPISODES):
    obs = eval_env.reset()
    done = False
    ep_ret, ep_len = 0.0, 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        # Check if episode finished
        if done[0]:
            # Get unnormalized return from info
            if 'episode' in info[0]:
                ep_ret = info[0]['episode']['r']
                ep_len = info[0]['episode']['l']
            else:
                # Fallback: use accumulated normalized rewards
                ep_ret += float(reward[0])
                ep_len += 1
            break
        
        ep_ret += float(reward[0])
        ep_len += 1
    
    returns.append(ep_ret)
    lengths.append(ep_len)

# Log results
print(f"\nFinal Evaluation Results:")
print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
print(f"  Median Return: {np.median(returns):.2f}")
print(f"  Min/Max Return: {np.min(returns):.2f} / {np.max(returns):.2f}")
print(f"  Mean Episode Length: {np.mean(lengths):.1f}")

wandb.log({
    "final_eval/mean_return": float(np.mean(returns)),
    "final_eval/std_return": float(np.std(returns)),
    "final_eval/median_return": float(np.median(returns)),
    "final_eval/min_return": float(np.min(returns)),
    "final_eval/max_return": float(np.max(returns)),
    "final_eval/mean_ep_len": float(np.mean(lengths)),
})

wandb.log({
    "final_eval/episodes": wandb.Table(
        columns=["episode", "return", "length"],
        data=[[i, float(r), int(l)] for i, (r, l) in enumerate(zip(returns, lengths))]
    )
})

wandb.finish()

print(f"\n{'='*60}")
print("Training and evaluation completed!")
print(f"Model saved to: {final_path}")
print(f"VecNormalize stats saved to: checkpoints/vec_normalize.pkl")
print(f"{'='*60}\n")