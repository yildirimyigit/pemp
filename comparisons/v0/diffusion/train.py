# %%
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import importlib

import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType


def import_diffusion_policy():
    """
    Workaround for occasional importlib KeyError: 'lerobot.policies' when using
    `from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy`.
    """
    import lerobot.policies  # ensure parent package is registered
    mod = importlib.import_module("lerobot.policies.diffusion.modeling_diffusion")
    return mod.DiffusionPolicy


DiffusionPolicy = import_diffusion_policy()

# %%
class FirstFrameOnly(Dataset):
    """Keep only the first index of each episode so each sample is a full 0..399 rollout."""
    def __init__(self, base_ds: Dataset):
        self.ds = base_ds
        self.first_indices: List[int] = []
        last_ep = None
        for i in range(len(base_ds)):
            item = base_ds[i]
            ep = int(item["episode_index"])
            if ep != last_ep:
                self.first_indices.append(i)
                last_ep = ep

    def __len__(self) -> int:
        return len(self.first_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.ds[self.first_indices[idx]]

# %%
dataset_dir = Path("/home/yigit/projects/pemp/comparisons/diffusion/lerobot_data/local/my_skill_time2q")
repo_id = "local/my_skill_time2q"
out_dir = Path("outputs/my_skill_full400")

horizon = 400
batch_size = 8
num_workers = 2
lr = 1e-4
weight_decay = 1e-6
steps = 20000
log_every = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- metadata + features
meta = LeRobotDatasetMetadata(repo_id, root=dataset_dir)
features = dataset_to_policy_features(meta.features)

# Ensure diffusion sees ACTION outputs and everything else as input
output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {k: ft for k, ft in features.items() if k not in output_features}

# ---- config
cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
cfg.n_obs_steps = 1
cfg.horizon = horizon
cfg.n_action_steps = horizon

policy = DiffusionPolicy(cfg, dataset_stats=meta.stats).to(device)
policy.train()

# ---- dataset windowing (use computed delta indices)
delta_timestamps = {
    "observation.state": [i / meta.fps for i in cfg.observation_delta_indices],
    "observation.environment_state": [i / meta.fps for i in cfg.observation_delta_indices],
    "action": [i / meta.fps for i in cfg.action_delta_indices],
}

base_ds = LeRobotDataset(repo_id, root=dataset_dir, delta_timestamps=delta_timestamps)
ds = FirstFrameOnly(base_ds)

loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)

optim = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

it = iter(loader)
for step in range(steps):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)

    # Keep only what diffusion expects (+ optional action_is_pad if present)
    keep = {
        "observation.state",
        "observation.environment_state",
        "action",
        "action_is_pad",
    }
    batch = {k: v.to(device) for k, v in batch.items() if k in keep}

    # If dataset didn't provide padding mask, create it (all False: no padding)
    if "action_is_pad" not in batch:
        B = batch["action"].shape[0]
        batch["action_is_pad"] = torch.zeros((B, horizon), dtype=torch.bool, device=device)

    # If env state missing for any reason, duplicate obs state (satisfies assertion)
    if "observation.environment_state" not in batch:
        batch["observation.environment_state"] = batch["observation.state"]

    # Your time was stored per frame as (1,1); flatten to (B, n_obs_steps, obs_dim)
    for k in ["observation.state", "observation.environment_state"]:
        x = batch[k]
        B = x.shape[0]
        batch[k] = x.view(B, cfg.n_obs_steps, -1)

    # LeRobot DiffusionPolicy forward returns (loss, output_dict_or_none)
    loss, _ = policy(batch)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optim.step()

    if step % log_every == 0:
        print(
            f"step {step} | loss {loss.item():.6f} "
            f"| obs {tuple(batch['observation.state'].shape)} "
            f"| env {tuple(batch['observation.environment_state'].shape)} "
            f"| act {tuple(batch['action'].shape)} "
            f"| pad {tuple(batch['action_is_pad'].shape)}"
        )

out_dir.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(out_dir)
print(f"Saved policy to: {out_dir.resolve()}")

# %%



