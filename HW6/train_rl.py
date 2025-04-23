import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from panda_pushing_env import PandaPushingEnv, TARGET_POSE, OBSTACLE_CENTRE, OBSTACLE_HALFDIMS, BOX_SIZE

from pushing_rl import *

# 💾 用于保存 reward 的 callback
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
            ep_rew = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(ep_rew)
        return True

# ✅ 配置训练参数
total_timesteps = 25000
env = PandaPushingEnv(
    state_space='object_pose',
    reward_function=obstacle_free_pushing_reward_function_object_pose_space
)
env.reset()

# ✅ 实例化 PPO（强制使用 CPU，日志目录启用 TensorBoard）
model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./logs", learning_rate=1e-4)

# ✅ 启动训练
reward_callback = RewardTrackingCallback()
model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=reward_callback)

# ✅ 保存模型
model.save("free_pushing_object_pose")

# 📊 画 reward 曲线（如果有记录）
if reward_callback.rewards:
    plt.plot(reward_callback.rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve During Training")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curve.png")  # 保存图像
    plt.show()
else:
    print("⚠️ 没有记录到 reward，可能是 environment 的 info 没包含 'episode' 信息")
