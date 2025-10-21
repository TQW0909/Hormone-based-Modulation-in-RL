# # test_step1_skeleton.py
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# import gymnasium as gym
# from PPO_hormones import HormonePPOCallback  # your Step-1 class

# def make_env():
#     return gym.make("CartPole-v1")

# env = VecMonitor(DummyVecEnv([make_env for _ in range(4)]))

# model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, verbose=0)

# cb = HormonePPOCallback(verbose=2)  # verbose=2 prints start/end
# model.learn(total_timesteps=10_000, callback=cb)


# # test_step2_utils.py
# from PPO_hormones import clamp, EMA, squash

# # clamp
# assert clamp(5, 0, 3) == 3
# assert clamp(-2, 0, 3) == 0
# assert clamp(1.5, 0, 3) == 1.5

# # ema
# prev = None
# x = EMA(prev, 10, 0.2); assert x == 10    # first value
# x = EMA(x, 0, 0.2);      assert round(x,2) == 8.0   # 0.2*0 + 0.8*10

# # squash (centered at 1.0)
# mid = squash(1.0); assert 0.45 < mid < 0.55
# hi  = squash(1.5); assert hi > mid
# lo  = squash(0.5); assert lo < mid

# print("Step 2 utils OK")


# # test_step3_signals.py
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# import gymnasium as gym
# from PPO_hormones import HormonePPOCallback  # your Step-1 class

# def make_env():
#     return gym.make("CartPole-v1")

# env = VecMonitor(DummyVecEnv([make_env for _ in range(4)]))
# model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, verbose=0)

# cb = HormonePPOCallback(verbose=0)
# model.learn(total_timesteps=5_000, callback=cb)


# # test_step4_warmup_norm.py
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# import gymnasium as gym
# from PPO_hormones import HormonePPOCallback  # your Step-1 class

# def make_env():
#     return gym.make("CartPole-v1")

# env = VecMonitor(DummyVecEnv([make_env for _ in range(4)]))
# model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, verbose=0)

# cb = HormonePPOCallback(warmup_rollouts=3, verbose=0)
# model.learn(total_timesteps=8_000, callback=cb)


# # test_step6_decay.py
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# import gymnasium as gym
# from PPO_hormones import HormonePPOCallback

# def make_env():
#     return gym.make("CartPole-v1")

# env = VecMonitor(DummyVecEnv([make_env for _ in range(4)]))
# model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, verbose=0)

# cb = HormonePPOCallback(
#     warmup_rollouts=3,
#     beta_A=0.6, beta_C=0.25, beta_D=0.35,  # decay rates
#     verbose=0
# )
# model.learn(total_timesteps=10_000, callback=cb)


# # test_clip_lr_schedule_fix.py
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# import gymnasium as gym
# from PPO_hormones import HormonePPOCallback

# def make_env():
#     return gym.make("CartPole-v1")

# env = VecMonitor(DummyVecEnv([make_env for _ in range(4)]))

# model = PPO("MlpPolicy", env,
#             learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
#             n_steps=256, batch_size=256, verbose=0)

# cb = HormonePPOCallback(
#     warmup_rollouts=2,
#     ent0=0.01, lr0=3e-4, clip0=0.2,
#     verbose=2
# )

# model.learn(total_timesteps=8_000, callback=cb)

# # Verify types / callables
# print("clip_range callable?", callable(model.clip_range))
# print("lr schedule callable?", callable(model.lr_schedule))
# print("OK")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import gymnasium as gym

# Make sure to wrap with Monitor/VecMonitor so ep_info_buffer is populated
def make_env():
    return gym.make("CartPole-v1")

env = DummyVecEnv([make_env for _ in range(8)])
env = VecMonitor(env)  # <-- important

eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

from PPO_hormones import HormonePPOCallback

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

horm_cb = HormonePPOCallback(
    warmup_rollouts=5,
    verbose=2,          # show start/end logs at first
    ent0=0.01, lr0=3e-4, clip0=0.2
)

eval_cb = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=10_000)

model.learn(total_timesteps=250_000, callback=CallbackList([horm_cb, eval_cb]))
