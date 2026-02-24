
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pathlib import Path
from datetime import datetime

def save_data(config: object, ts: np.ndarray, Us: np.ndarray):
    save_path = config.save_path
    if save_path is None:
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"sim_data_{stamp}.npz"
    else:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(ts, np.memmap):
        ts.flush()
    if isinstance(Us, np.memmap):
        Us.flush()

    np.savez_compressed(out_file, ts=ts, Us=Us)

    for arr in (ts, Us):
        if isinstance(arr, np.memmap):
            try:
                Path(arr.filename).unlink(missing_ok=True)
            except OSError:
                pass

    print(f"Gemte data til: {out_file.resolve()}")

def read_data(config: object):
    in_file = Path(config.save_path)
    data = np.load(in_file)
    ts = data["ts"]
    Us = data["Us"]
    print(f"Indlæste data fra: {in_file.resolve()}")
    return ts, Us


def plot_fields_with_slider(ts: np.ndarray, Us: np.ndarray, system: object):
    Nt, m, _, _ = Us.shape

    if getattr(system, "field_names", None) is None:
        field_names = [f"field {k}" for k in range(m)]
    else:
        field_names = list(system.field_names)
        if len(field_names) != m:
            field_names = [f"field {k}" for k in range(m)]

    sorted_fields = sorted(range(m), key=lambda k: field_names[k])
    Us = Us[:, sorted_fields, :, :]
    field_names = [field_names[k] for k in sorted_fields]

    finite = np.isfinite(Us)
    Us_f = np.where(finite, Us, np.nan)
    field_mins = np.nanmin(Us_f, axis=(0, 2, 3))
    field_maxs = np.nanmax(Us_f, axis=(0, 2, 3))

    ncols = int(np.ceil(np.sqrt(m)))
    nrows = int(np.ceil(m / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    plt.subplots_adjust(bottom=0.18)

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
        ax.set_title(f"{field_names[k]}\nmin={field_mins[k]:.3e}, max={field_maxs[k]:.3e}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

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
    