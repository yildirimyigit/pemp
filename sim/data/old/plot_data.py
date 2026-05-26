import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot a given dimension of data from npz files over timesteps.")
    parser.add_argument("--key", type=str, default="actions", help="Data key to plot (e.g., actions, qpos, ctrl). Default: actions")
    parser.add_argument("--dim", type=int, default=0, help="Dimension index to plot. Default: 0")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent/"raw"
    npz_files = sorted(data_dir.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return

    plt.figure(figsize=(10, 6))

    for npz_file in npz_files:
        try:
            with np.load(npz_file) as data:
                if args.key not in data:
                    print(f"Key '{args.key}' not found in {npz_file.name}. Available keys: {list(data.keys())}")
                    continue
                
                val = data[args.key]
                if val.ndim > 1:
                    if args.dim >= val.shape[1]:
                        print(f"Dimension {args.dim} is out of bounds for {args.key} in {npz_file.name} (shape: {val.shape})")
                        continue
                    plot_data = val[:, args.dim]
                else:
                    plot_data = val
                    
                plt.plot(plot_data, label=f"{npz_file.name} - dim {args.dim}")
        except Exception as e:
            print(f"Error loading {npz_file.name}: {e}")

    plt.title(f"Plot of {args.key} - dimension {args.dim}")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
