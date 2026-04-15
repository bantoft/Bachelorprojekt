from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def _resolve_timeseries_path(archive_path: Path, archive: np.lib.npyio.NpzFile) -> Path | None:
    for key in ("timeseries_file", "snapshot_file"):
        if key in archive.files:
            value = archive[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                candidate = Path(str(value.item()))
            else:
                candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = archive_path.parent / candidate
            return candidate
    return None


def load_simulation(path: str) -> tuple[np.ndarray, np.ndarray]:
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"File not found: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        if "u_hist" in archive.files and "t_hist" in archive.files:
            return np.asarray(archive["t_hist"]), np.asarray(archive["u_hist"])

        timeseries_path = _resolve_timeseries_path(archive_path, archive)
        if timeseries_path is None:
            raise ValueError(
                f"Could not find 'u_hist'/'t_hist' in {archive_path} and no timeseries reference was stored."
            )

    with np.load(timeseries_path, allow_pickle=False) as archive:
        if "u_hist" not in archive.files or "t_hist" not in archive.files:
            raise ValueError(f"Timeseries file {timeseries_path} does not contain 'u_hist' and 't_hist'.")
        return np.asarray(archive["t_hist"]), np.asarray(archive["u_hist"])


def _field_count(u_hist: np.ndarray) -> int:
    if u_hist.ndim == 3:
        return 1
    if u_hist.ndim == 4:
        return int(u_hist.shape[1])
    raise ValueError(f"Unsupported u_hist shape: {u_hist.shape}")


def _frame_at(u_hist: np.ndarray, index: int) -> np.ndarray:
    if u_hist.ndim == 3:
        return u_hist[index][None, :, :]
    return u_hist[index]


def visualize(path: str) -> None:
    t_hist, u_hist = load_simulation(path)
    n_frames = int(u_hist.shape[0])
    n_fields = _field_count(u_hist)

    if n_frames == 0:
        raise ValueError("No snapshots found in the loaded simulation.")

    n_cols = min(2, n_fields)
    n_rows = int(np.ceil(n_fields / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.0 * n_rows), squeeze=False)
    fig.subplots_adjust(bottom=0.16, hspace=0.28, wspace=0.20)

    flat_axes = axes.ravel()
    images = []

    initial_frame = _frame_at(u_hist, 0)
    vmin = initial_frame.min(axis=(1, 2))
    vmax = initial_frame.max(axis=(1, 2))

    for field_idx in range(n_fields):
        ax = flat_axes[field_idx]
        field = initial_frame[field_idx]
        image = ax.imshow(field, origin="lower", cmap="viridis", vmin=vmin[field_idx], vmax=vmax[field_idx])
        ax.set_title(f"Field {field_idx}")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        images.append(image)

    for extra_ax in flat_axes[n_fields:]:
        extra_ax.axis("off")

    time_text = fig.text(0.02, 0.03, f"t = {t_hist[0]:.6g}   step = 0", fontsize=11)

    slider_ax = fig.add_axes((0.12, 0.06, 0.76, 0.03))
    slider = Slider(
        slider_ax,
        "t",
        float(t_hist[0]),
        float(t_hist[-1]),
        valinit=float(t_hist[0]),
        valstep=t_hist,
    )

    def update(time_value: float) -> None:
        index = int(np.searchsorted(t_hist, time_value, side="left"))
        if index >= n_frames:
            index = n_frames - 1
        elif index > 0:
            prev_idx = index - 1
            if abs(time_value - t_hist[prev_idx]) <= abs(time_value - t_hist[index]):
                index = prev_idx

        frame = _frame_at(u_hist, index)
        for field_idx, image in enumerate(images):
            image.set_data(frame[field_idx])
        time_text.set_text(f"t = {t_hist[index]:.6g}   step = {index}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved PDE simulation data with a time slider.")
    parser.add_argument("path", type=str, help="Path to a simulation .npz file or a timeseries .npz file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize(args.path)


if __name__ == "__main__":
    main()