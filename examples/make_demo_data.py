"""Write synthetic Feather inputs to exercise the experiment runner."""
import argparse
from pathlib import Path
from examples.quickstart import synthetic_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--station", default="wind1")
    args = parser.parse_args()
    x, y = synthetic_series(2000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, start, stop in [("train", 0, 1200), ("valid", 1200, 1600), ("test", 1600, 2000)]:
        x.iloc[start:stop].reset_index(drop=True).to_feather(args.output_dir/f"{name}_raw_{args.station}.feather")
        y.iloc[start:stop].reset_index(drop=True).to_frame().to_feather(args.output_dir/f"{name}_y_{args.station}.feather")
    print(f"Synthetic inputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
