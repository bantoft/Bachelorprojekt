# simuleringer/2D/main.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
from datetime import datetime
from typing import Optional


def plot_fields_with_slider(
    ts: np.ndarray,
    Us: np.ndarray,
    field_names=None,
    save_data: bool = True,
    save_path: Optional[str] = None,
):
    """
    ts: (Nt,)
    Us: (Nt, m, nx, ny)
    field_names: list[str] længde m
    save_data: gem ts og Us til .npz
    save_path: valgfri sti til outputfil (default: data/sim_data_YYYYmmdd_HHMMSS.npz)
    """
    Nt, m, nx, ny = Us.shape

    if save_data:
        if save_path is None:
            out_dir = Path("data")
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = out_dir / f"sim_data_{stamp}.npz"
        else:
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out_file, ts=ts, Us=Us)
        print(f"Gemte data til: {out_file.resolve()}")

    if field_names is None:
        field_names = [f"field {k}" for k in range(m)]

    field_mins = np.nanmin(Us, axis=(0, 2, 3))
    field_maxs = np.nanmax(Us, axis=(0, 2, 3))

    # vælg layout automatisk (pænt grid)
    ncols = int(np.ceil(np.sqrt(m)))
    nrows = int(np.ceil(m / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    plt.subplots_adjust(bottom=0.18)

    # gør axes flad liste
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    ims = []
    for k in range(m):
        ax = axes[k]
        vmin = field_mins[k]
        vmax = field_maxs[k]
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        elif np.isclose(vmin, vmax):
            pad = max(1e-12, 1e-6 * max(abs(vmin), 1.0))
            vmin -= pad
            vmax += pad

        im = ax.imshow(Us[0, k], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(field_names[k])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    # slå evt. overskydende subplot-axes fra
    for k in range(m, len(axes)):
        axes[k].axis("off")

    ax_slider = plt.axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(ax_slider, "frame", 0, Nt - 1, valinit=0, valstep=1)

    def update(val):
        i = int(s.val)
        for k in range(m):
            ims[k].set_data(Us[i, k])
        fig.suptitle(f"t = {ts[i]:.4f}  |  frame {i}/{Nt-1}")
        fig.canvas.draw_idle()

    s.on_changed(update)
    update(0)
    plt.show()