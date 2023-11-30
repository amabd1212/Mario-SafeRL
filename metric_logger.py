import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from pathlib import Path

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []

        self.init_episode()
        self.record_time = time.time()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

    def log_episode(self, loss):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1
        ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5) if self.curr_ep_loss_length > 0 else 0
        self.ep_avg_losses.append(ep_avg_loss)
        self.init_episode()

    def record(self, episode, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)

        if episode % 50 == 0:
            self.plot_metrics()
            time_since_last_record = np.round(time.time() - self.record_time, 3)
            self.log_to_file(episode, step, mean_ep_reward, mean_ep_length, mean_ep_loss, time_since_last_record)

        self.record_time = time.time()

    def plot_metrics(self):
        for metric, plot_file in [("ep_rewards", self.ep_rewards_plot),
                                  ("ep_lengths", self.ep_lengths_plot),
                                  ("ep_avg_losses", self.ep_avg_losses_plot)]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"Moving Avg {metric}")
            plt.legend()
            plt.savefig(plot_file)

    def log_to_file(self, episode, step, mean_reward, mean_length, mean_loss, time_delta):
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:>8}{step:>8}{mean_reward:15.3f}{mean_length:15.3f}{mean_loss:15.3f}"
                f"{time_delta:15.3f}{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        print(
            f"Episode {episode} - Step {step} - Mean Reward {mean_reward} - "
            f"Mean Length {mean_length} - Mean Loss {mean_loss} - "
            f"Time Delta {time_delta} - Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

# Usage example
# save_dir = Path("your_directory_path")
# logger = MetricLogger(save_dir)
