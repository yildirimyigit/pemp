"""Process periodic Adroit action trajectories into CNMP-ready arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_raw_trajectories(raw_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    trajectories: list[np.ndarray] = []
    frequencies: list[int] = []
    for npz_file in sorted(raw_dir.glob("*.npz")):
        with np.load(npz_file) as data:
            trajectories.append(data["actions"].astype(np.float32))
            frequencies.append(int(data["num_strikes"].item()))

    if not trajectories:
        raise FileNotFoundError(f"no .npz files found in {raw_dir}")
    return trajectories, frequencies


def resample_trajectory(trajectory: np.ndarray, target_steps: int) -> np.ndarray:
    source_steps = np.arange(trajectory.shape[0])
    target_positions = np.linspace(0, trajectory.shape[0] - 1, target_steps)
    resampled = np.zeros((target_steps, trajectory.shape[1]), dtype=np.float32)
    for dim in range(trajectory.shape[1]):
        resampled[:, dim] = np.interp(
            target_positions,
            source_steps,
            trajectory[:, dim],
        )
    return resampled


def split_by_frequency(
    trajectories: list[np.ndarray],
    frequencies: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_steps = min(trajectory.shape[0] for trajectory in trajectories)
    max_frequency = max(frequencies)
    resampled = [
        resample_trajectory(trajectory, target_steps) for trajectory in trajectories
    ]

    x_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    g_train: list[float] = []
    x_test: list[np.ndarray] = []
    y_test: list[np.ndarray] = []
    g_test: list[float] = []

    for frequency in sorted(set(frequencies)):
        indices = [
            index for index, value in enumerate(frequencies) if value == frequency
        ]
        if len(indices) < 2:
            raise ValueError(
                f"frequency {frequency} needs at least 2 demonstrations, got {len(indices)}"
            )
        normalized_frequency = frequency / max_frequency
        for index in indices[:-1]:
            x_train.append(np.linspace(0, 1, target_steps).reshape(-1, 1))
            y_train.append(resampled[index])
            g_train.append(normalized_frequency)

        held_out_index = indices[-1]
        x_test.append(np.linspace(0, 1, target_steps).reshape(-1, 1))
        y_test.append(resampled[held_out_index])
        g_test.append(normalized_frequency)

    return (
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(g_train, dtype=np.float32),
        np.asarray(x_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        np.asarray(g_test, dtype=np.float32),
    )


def save_processed_arrays(
    processed_dir: Path,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    names = ("x", "y", "g", "x_test", "y_test", "g_test")
    for name, values in zip(names, arrays, strict=True):
        np.save(processed_dir / f"{name}.npy", values)
        print(f"saved {processed_dir / f'{name}.npy'} shape={values.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("sim/data/raw"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("sim/data/processed"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_processed_arrays(
        processed_dir=args.processed_dir,
        arrays=split_by_frequency(*load_raw_trajectories(args.raw_dir)),
    )
