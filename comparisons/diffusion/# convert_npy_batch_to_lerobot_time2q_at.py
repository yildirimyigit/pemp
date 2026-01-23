# convert_time2q_dual_state_atomic.py
from __future__ import annotations

from pathlib import Path
import shutil
import time
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    npy_path = Path("/home/yigit/projects/pemp/data/xarm/processed/norm_400.npy")  # (24, 400, 7)
    datasets_root = Path("/home/yigit/projects/pemp/comparisons/diffusion/lerobot_data")

    repo_id = "local/my_skill_time2q"
    fps = 50
    task_str = "my_skill"

    final_dir = datasets_root / repo_id
    tmp_dir = datasets_root / (repo_id.replace("/", "_") + f"__tmp_{int(time.time())}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Key point: provide BOTH robot state and env state (no images).
    # Use (1,1) so HF doesn't try to float() a 1D array.
    features = {
        "observation.state": {"dtype": "float32", "shape": (1, 1), "names": ["t_robot"]},
        "observation.environment_state": {"dtype": "float32", "shape": (1, 1), "names": ["t_env"]},
        "action": {"dtype": "float32", "shape": (7,), "names": [f"q{i}" for i in range(1, 8)]},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=tmp_dir,   # dataset will be written into tmp_dir/{meta,data}
        fps=fps,
        features=features,
        use_videos=False,
    )

    trajs = np.load(npy_path)
    if trajs.ndim != 3 or trajs.shape[2] != 7:
        raise ValueError(f"Expected (N, T, 7), got {trajs.shape}")

    N, T, _ = trajs.shape

    for ep_idx in range(N):
        traj = trajs[ep_idx]
        for t in range(T):
            tt = np.array([[t]], dtype=np.float32)
            dataset.add_frame({
                "observation.state": tt,
                "observation.environment_state": tt,
                "action": traj[t].astype(np.float32),
                "task": task_str,
            })
        dataset.save_episode()
        print(f"Saved episode {ep_idx + 1}/{N}")

    del dataset  # close parquet writers

    # Move tmp_dir -> final_dir (tmp_dir contains meta/ and data/)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_dir), str(final_dir))

    print(f"Done. Dataset written to: {final_dir}")


if __name__ == "__main__":
    main()
