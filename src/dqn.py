import os
import random
import warnings
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _activation_layer(name: str) -> nn.Module:
    normalized = str(name or "relu").lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported model activation: {name}")


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        use_dueling: bool = False,
        dueling_aggregation: str = "mean",
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = max(1, int(action_dim))
        self.use_dueling = bool(use_dueling)
        self.dueling_aggregation = str(dueling_aggregation or "mean").lower()
        if self.action_dim < 2:
            warnings.warn(
                "QNetwork was created with action_dim < 2; traffic-light RL normally expects two actions.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.dueling_aggregation != "mean":
            raise ValueError("Only mean dueling aggregation is currently implemented.")

        feature_layers = []
        in_dim = self.obs_dim
        for _ in range(max(1, int(num_hidden_layers))):
            feature_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            feature_layers.append(_activation_layer(activation))
            in_dim = int(hidden_dim)
        self.feature_extractor = nn.Sequential(*feature_layers)

        if self.use_dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(in_dim, int(hidden_dim)),
                _activation_layer(activation),
                nn.Linear(int(hidden_dim), 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(in_dim, int(hidden_dim)),
                _activation_layer(activation),
                nn.Linear(int(hidden_dim), self.action_dim),
            )
            self.net = None
        else:
            self.value_stream = None
            self.advantage_stream = None
            self.net = nn.Linear(in_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        if not self.use_dueling:
            return self.net(features)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
        )


class DQNShared:
    def __init__(self, obs_dim: int, action_dim: int, config):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.algorithm = str(getattr(config, "algorithm", "dqn"))
        self.use_double_dqn = bool(getattr(config, "use_double_dqn", False))
        self.use_dueling_dqn = bool(getattr(config, "use_dueling_dqn", False))
        self.dueling_aggregation = str(getattr(config, "dueling_aggregation", "mean"))
        self.checkpoint_version = int(getattr(config, "checkpoint_version", 1))
        self.gamma = float(config.gamma)
        self.batch_size = int(config.batch_size)
        self.eps_start = float(config.eps_start)
        self.eps_end = float(config.eps_end)
        self.eps_decay_steps = max(1, int(config.eps_decay_steps))
        self.grad_clip_norm = float(getattr(config, "grad_clip_norm", 10.0))
        self.global_step = 0
        self.target_updates_count = 0
        self.last_target_update_step = 0
        self.last_update_metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = int(getattr(config, "model_hidden_dim", getattr(config, "hidden_dim", 128)))
        num_hidden_layers = int(
            getattr(config, "model_num_hidden_layers", getattr(config, "num_hidden_layers", 2))
        )
        activation = str(getattr(config, "model_activation", "relu"))

        self.q_net = QNetwork(
            obs_dim,
            action_dim,
            hidden_dim,
            num_hidden_layers,
            use_dueling=self.use_dueling_dqn,
            dueling_aggregation=self.dueling_aggregation,
            activation=activation,
        ).to(self.device)
        self.target_net = QNetwork(
            obs_dim,
            action_dim,
            hidden_dim,
            num_hidden_layers,
            use_dueling=self.use_dueling_dqn,
            dueling_aggregation=self.dueling_aggregation,
            activation=activation,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(config.lr))
        self.replay = ReplayBuffer(int(config.replay_size))
        self.steps_done = 0

    def epsilon(self, step: Optional[int] = None) -> float:
        step_value = self.steps_done if step is None else int(step)
        frac = min(1.0, step_value / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, epsilon: Optional[float] = None) -> int:
        eps = self.epsilon() if epsilon is None else float(epsilon)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def q_values(self, obs) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.q_net(obs_t).squeeze(0).detach().cpu().numpy()
        return values.astype(np.float32, copy=False)

    def store(self, obs, action, reward, next_obs, done) -> None:
        self.replay.push(obs, action, reward, next_obs, done)

    def compute_td_target(
        self,
        rewards_t: torch.Tensor,
        next_obs_t: torch.Tensor,
        dones_t: torch.Tensor,
    ) -> torch.Tensor:
        reward_was_1d = rewards_t.dim() == 1
        rewards = rewards_t.unsqueeze(1) if reward_was_1d else rewards_t
        dones = dones_t.unsqueeze(1) if dones_t.dim() == 1 else dones_t
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_net(next_obs_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs_t).gather(1, next_actions)
            else:
                next_q = self.target_net(next_obs_t).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * next_q
        return target.squeeze(1) if reward_was_1d else target

    def update(self) -> Optional[dict]:
        if len(self.replay) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        all_q_values = self.q_net(obs_t)
        q_values = all_q_values.gather(1, actions_t)
        target = self.compute_td_target(rewards_t, next_obs_t, dones_t)
        td_error = target - q_values

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "td_error_avg": float(td_error.detach().abs().mean().item()),
            "q_value_avg": float(all_q_values.detach().mean().item()),
            "q_value_max": float(all_q_values.detach().max().item()),
            "q_value_min": float(all_q_values.detach().min().item()),
            "target_q_avg": float(target.detach().mean().item()),
        }
        self.last_update_metrics = metrics
        return metrics

    def update_target(self, global_step: Optional[int] = None) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_updates_count += 1
        if global_step is not None:
            self.last_target_update_step = int(global_step)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "checkpoint_version": self.checkpoint_version,
                "algorithm": self.algorithm,
                "use_double_dqn": self.use_double_dqn,
                "use_dueling_dqn": self.use_dueling_dqn,
                "dueling_aggregation": self.dueling_aggregation,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "online_model_state_dict": self.q_net.state_dict(),
                "target_model_state_dict": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "global_step": self.global_step,
                "target_updates_count": self.target_updates_count,
                "last_target_update_step": self.last_target_update_step,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        ckpt_obs_dim = int(checkpoint.get("obs_dim", self.obs_dim))
        ckpt_action_dim = int(checkpoint.get("action_dim", self.action_dim))
        ckpt_algorithm = str(checkpoint.get("algorithm", "dqn"))
        ckpt_version = int(checkpoint.get("checkpoint_version", 1))
        ckpt_double = bool(checkpoint.get("use_double_dqn", False))
        ckpt_dueling = bool(checkpoint.get("use_dueling_dqn", False))
        if (
            ckpt_obs_dim != self.obs_dim
            or ckpt_action_dim != self.action_dim
            or ckpt_algorithm != self.algorithm
            or ckpt_version != self.checkpoint_version
            or ckpt_double != self.use_double_dqn
            or ckpt_dueling != self.use_dueling_dqn
        ):
            raise ValueError(
                "Checkpoint incompatible with current algorithm or scenario: "
                f"checkpoint algorithm/version/double/dueling/obs/action="
                f"{ckpt_algorithm}/{ckpt_version}/{ckpt_double}/{ckpt_dueling}/"
                f"{ckpt_obs_dim}/{ckpt_action_dim}, current="
                f"{self.algorithm}/{self.checkpoint_version}/{self.use_double_dqn}/"
                f"{self.use_dueling_dqn}/{self.obs_dim}/{self.action_dim}"
            )
        self.q_net.load_state_dict(checkpoint.get("online_model_state_dict", checkpoint["q_net"]))
        self.target_net.load_state_dict(
            checkpoint.get("target_model_state_dict", checkpoint.get("target_net", checkpoint["q_net"]))
        )
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", checkpoint["optimizer"]))
        self.steps_done = int(checkpoint.get("steps_done", 0))
        self.global_step = int(checkpoint.get("global_step", self.steps_done))
        self.target_updates_count = int(checkpoint.get("target_updates_count", 0))
        self.last_target_update_step = int(checkpoint.get("last_target_update_step", 0))
        self.q_net.eval()
        self.target_net.eval()


def model_info_from_config(config, obs_dim: int, action_dim: int) -> dict:
    return {
        "algorithm": str(getattr(config, "algorithm", "dqn")),
        "use_double_dqn": bool(getattr(config, "use_double_dqn", False)),
        "use_dueling_dqn": bool(getattr(config, "use_dueling_dqn", False)),
        "checkpoint_version": int(getattr(config, "checkpoint_version", 1)),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
    }
