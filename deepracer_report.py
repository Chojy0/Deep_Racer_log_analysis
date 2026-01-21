#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ReportPaths:
    root_path: Path
    track_path: Path
    log_path: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepRacer log analysis report generator")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path.cwd(),
        help="Project root containing log_analysis and tracks directories",
    )
    parser.add_argument(
        "--log-folder",
        type=str,
        default="center",
        help="Folder name under log_analysis (e.g., center)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Override path to training-simtrace directory",
    )
    parser.add_argument(
        "--track-file",
        type=str,
        default=None,
        help="Track .npy filename (e.g., reInvent2019_track_ccw.npy)",
    )
    parser.add_argument(
        "--track-path",
        type=Path,
        default=None,
        help="Override tracks directory path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report_output"),
        help="Output directory for report artifacts",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ReportPaths:
    root_path = args.root_path.resolve()
    track_path = args.track_path or root_path / "tracks"
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = root_path / "log_analysis" / args.log_folder / "training-simtrace"
    output_dir = args.output_dir
    return ReportPaths(
        root_path=root_path,
        track_path=track_path,
        log_path=log_path,
        output_dir=output_dir,
    )


def list_log_files(log_path: Path) -> list[Path]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log path not found: {log_path}")
    files = [path for path in log_path.iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No log files found in: {log_path}")

    def sort_key(path: Path) -> int:
        name = path.name.split("-")[0]
        return int(name) if name.isdigit() else 0

    return sorted(files, key=sort_key)


def load_log_file(path: Path, iteration: int) -> pd.DataFrame:
    text = path.read_text()
    if "[" in text:
        text = text.replace("[", "").replace("]", "")
        text = text.replace("action", "action_1,action_2")
        text = text.replace(" ", ",")

    df = pd.read_csv(StringIO(text))
    df.dropna(inplace=True)
    if df.empty:
        return df

    float_cols = df.columns[~df.columns.isin(["episode_status", "done", "all_wheels_on_track"])]
    int_cols = [col for col in ["episode", "steps", "closest_waypoint"] if col in df.columns]
    bool_cols = [col for col in ["done", "all_wheels_on_track"] if col in df.columns]

    df[float_cols] = df[float_cols].astype("float")
    if int_cols:
        df[int_cols] = df[int_cols].astype("int")
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("bool")

    df.insert(0, "iteration", iteration)
    return df


def load_all_logs(files: list[Path]) -> pd.DataFrame:
    data_frames = []
    for iteration, file_path in enumerate(files):
        df = load_log_file(file_path, iteration)
        if not df.empty:
            data_frames.append(df)
    if not data_frames:
        raise ValueError("No valid log data loaded.")
    return pd.concat(data_frames, axis=0, ignore_index=True)


def summarize_episodes(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("episode", as_index=False)

    def maybe_last(series: pd.Series) -> str:
        return series.iloc[-1] if not series.empty else ""

    summary = grouped.agg(
        steps=("steps", "max"),
        reward_sum=("reward", "sum"),
        reward_mean=("reward", "mean"),
        speed_mean=("speed", "mean"),
        steering_abs_mean=("steering_angle", lambda s: s.abs().mean()),
        progress_max=("progress", "max"),
        offtrack_count=("all_wheels_on_track", lambda s: (~s).sum() if s.dtype == bool else 0),
        episode_status=("episode_status", maybe_last),
    )

    summary.sort_values("episode", inplace=True)
    return summary


def summarize_overall(episode_summary: pd.DataFrame) -> dict:
    completion = (episode_summary["progress_max"] >= 100).sum()
    total = len(episode_summary)
    return {
        "total_episodes": int(total),
        "completed_episodes": int(completion),
        "completion_rate": float(completion / total) if total else 0.0,
        "reward_mean": float(episode_summary["reward_mean"].mean()),
        "reward_sum_mean": float(episode_summary["reward_sum"].mean()),
        "speed_mean": float(episode_summary["speed_mean"].mean()),
        "steering_abs_mean": float(episode_summary["steering_abs_mean"].mean()),
    }


def save_line_plot(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_histogram(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(series, bins=30, color="#4c78a8", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.scatter(df["steering_angle"], df["speed"], alpha=0.4, s=8)
    plt.title("Speed vs. Steering Angle")
    plt.xlabel("Steering Angle")
    plt.ylabel("Speed")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_track_plot(
    df: pd.DataFrame,
    track_path: Path,
    track_file: str,
    output_path: Path,
) -> bool:
    if not track_file:
        return False
    track_file_path = track_path / track_file
    if not track_file_path.exists():
        return False
    track_arr = np.load(track_file_path)
    track_center = track_arr[:, [0, 1]]
    track_left = track_arr[:, [2, 3]]
    track_right = track_arr[:, [4, 5]]

    best_episode = df.groupby("episode")["progress"].max().idxmax()
    episode_df = df[df["episode"] == best_episode]

    plt.figure(figsize=(8, 8))
    plt.scatter(track_left[:, 0], track_left[:, 1], s=8, c="#c0c0c0", label="Left")
    plt.scatter(track_center[:, 0], track_center[:, 1], s=8, c="#808080", label="Center")
    plt.scatter(track_right[:, 0], track_right[:, 1], s=8, c="#c0c0c0", label="Right")
    plt.plot(episode_df["X"], episode_df["Y"], color="#f58518", linewidth=2, label="Best Episode")
    plt.title(f"Trajectory on Track ({track_file})")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def write_report_md(
    output_path: Path,
    overall: dict,
    episode_csv: Path,
    plots: dict,
    track_plot_included: bool,
) -> None:
    lines = [
        "# DeepRacer Log Analysis Report",
        "",
        "## Summary",
        f"- Total episodes: **{overall['total_episodes']}**",
        f"- Completed episodes: **{overall['completed_episodes']}**",
        f"- Completion rate: **{overall['completion_rate']:.2%}**",
        f"- Mean reward (per episode): **{overall['reward_sum_mean']:.2f}**",
        f"- Mean speed: **{overall['speed_mean']:.2f}**",
        f"- Mean absolute steering: **{overall['steering_abs_mean']:.2f}**",
        "",
        "## Outputs",
        f"- Episode summary CSV: `{episode_csv.name}`",
        "",
        "## Plots",
        f"![Completion](plots/{plots['completion'].name})",
        f"![Reward](plots/{plots['reward'].name})",
        f"![Speed Distribution](plots/{plots['speed_hist'].name})",
        f"![Steering vs Speed](plots/{plots['steer_speed'].name})",
    ]
    if track_plot_included:
        lines.append(f"![Track Trajectory](plots/{plots['track'].name})")
    output_path.write_text("\n".join(lines))


def generate_report(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    output_dir = paths.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_files = list_log_files(paths.log_path)
    df = load_all_logs(log_files)

    episode_summary = summarize_episodes(df)
    episode_csv = output_dir / "episode_summary.csv"
    episode_summary.to_csv(episode_csv, index=False)

    overall = summarize_overall(episode_summary)
    summary_json = output_dir / "overall_summary.json"
    summary_json.write_text(json.dumps(overall, indent=2, ensure_ascii=False))

    plots = {
        "completion": plots_dir / "completion_rate.png",
        "reward": plots_dir / "reward_per_episode.png",
        "speed_hist": plots_dir / "speed_distribution.png",
        "steer_speed": plots_dir / "steering_vs_speed.png",
        "track": plots_dir / "track_trajectory.png",
    }

    save_line_plot(
        episode_summary["episode"],
        episode_summary["progress_max"],
        "Episode Completion",
        "Episode",
        "Progress (%)",
        plots["completion"],
    )
    save_line_plot(
        episode_summary["episode"],
        episode_summary["reward_sum"],
        "Reward per Episode",
        "Episode",
        "Total Reward",
        plots["reward"],
    )
    save_histogram(
        df["speed"],
        "Speed Distribution",
        "Speed",
        plots["speed_hist"],
    )
    save_scatter(df, plots["steer_speed"])
    track_plot_included = save_track_plot(
        df,
        paths.track_path,
        args.track_file,
        plots["track"],
    )

    report_md = output_dir / "report.md"
    write_report_md(report_md, overall, episode_csv, plots, track_plot_included)


def main() -> None:
    args = parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
