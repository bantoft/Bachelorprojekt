from __future__ import annotations

from dataclasses import replace
import importlib
import zipfile
import numpy as np

from pathlib import Path

from .grid import Grid2D
from .types import PDESystem
from .config import EngineConfig
from .dynamisk_dt import AdaptiveStepController


def _save_npz_archive(path: Path, arrays: dict[str, np.ndarray], compression_level: int) -> None:
    if compression_level <= 0:
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, array in arrays.items():
                member_name = name if name.endswith(".npy") else f"{name}.npy"
                with archive.open(member_name, mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
        return

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as archive:
        for name, array in arrays.items():
            member_name = name if name.endswith(".npy") else f"{name}.npy"
            with archive.open(member_name, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)

def simulate_system(pde_system_path: str, cfg: EngineConfig) -> tuple[np.ndarray, float]:
    """Simulate PDE system for n_steps using RK4."""

    # --- Group 2: Load PDE module and build system container ---
    # Dynamically import selected PDE and wrap callbacks in PDESystem.
    module = importlib.import_module(f"PDE_system.{pde_system_path}")

    # --- Group 1: Configure global grid settings from runtime config ---
    # PDE modules can optionally specify N_FIELDS for coupled systems.
    module_n_fields = int(getattr(module, "N_FIELDS", cfg.n_fields))
    if module_n_fields < 1:
        raise ValueError(f"Invalid N_FIELDS={module_n_fields} in PDE_system.{pde_system_path}")
    Grid2D.cfg = replace(cfg, n_fields=module_n_fields)

    system = PDESystem(module.rhs, Grid2D, module.ic, getattr(module, "create_rhs_kwargs", None))

    # --- Group 3: Create simulation state and adaptive stepper ---
    # Build grid + initial condition, then allocate integration buffers.
    grid = system.create_grid()
    u = system.initial_condition(grid)
    u_next = grid.alloc()
    stepper = AdaptiveStepController(grid, system, cfg)
    

    # --- Group 4: Build static RHS arguments used every step ---
    # These kwargs are passed to rhs/integrator at each time step.
    rhs_kwargs: dict = {"grid": grid}
    if system.create_rhs_kwargs is not None:
        rhs_kwargs.update(system.create_rhs_kwargs(grid, u))

    # --- Group 5: Pre-allocate streaming output buffers ---
    # Do not size from min_dt: very small min_dt can make allocation explode.
    # We estimate from the starting dt instead.
    dt_for_estimate = max(float(cfg.dt), np.finfo(np.float64).eps)
    max_steps = int(np.ceil((cfg.tn - cfg.t0) / dt_for_estimate)) + 1
    if cfg.save_dt <= 0.0:
        raise ValueError("save_dt must be > 0. Pass --save-dt or use positive --save-every.")
    n_snapshots = int(np.ceil((cfg.tn - cfg.t0) / cfg.save_dt)) + 2
    snapshot_file = Path(cfg.output).parent / f"{Path(cfg.output).stem}_snapshots.npy"
    timeseries_file = Path(cfg.output).parent / f"{Path(cfg.output).stem}_timeseries.npz"
    u_hist = np.lib.format.open_memmap(
        str(snapshot_file),
        mode='w+',
        dtype=u.dtype,
        shape=(n_snapshots, *u.shape),
    )
    t_hist = []
    
    # --- Group 6: Initialize history with initial condition ---
    u_hist[0] = u.copy()
    t_hist.append(cfg.t0)
    snapshot_idx = 1
    initial_peak = float(np.max(np.abs(u)))
    u_cap = 3.0 * initial_peak

    # --- Group 7: Initialize loop counters/state ---
    t = cfg.t0
    step = 0
    dt_step = cfg.dt  # Start with base dt
    next_snapshot_t = cfg.t0 + cfg.save_dt

    while t < cfg.tn:
        u_next, dt_step = stepper.step(t, u, dt_step, **rhs_kwargs)

        if t + dt_step > cfg.tn:
            dt_step = cfg.tn - t
            u_next = stepper.integrator.step( t, dt_step, u, u_next, **rhs_kwargs)
        
        u, u_next = u_next, u
        t += dt_step
        step += 1

        if np.isinf(u).any():
            raise FloatingPointError(
                f"Simulation aborted: state contains inf/-inf at step={step}, t={t:.6f}."
            )

        # Simple hard cap: keep values within +/- 5x initial peak magnitude.
        if u_cap > 0.0:
            np.clip(u, -u_cap, u_cap, out=u)

        while t >= next_snapshot_t:
            if snapshot_idx >= u_hist.shape[0]:
                raise RuntimeError(
                    "Snapshot buffer is full. Increase --save-dt or --dt, "
                    "or shorten --tn for this run."
                )
            u_hist[snapshot_idx] = u.copy()
            t_hist.append(t)
            snapshot_idx += 1
            next_snapshot_t += cfg.save_dt

        if step % 50 == 0:
            print(f"Step {step}, t={t:.8f}, dt={dt_step:.6f}")

    # Flush memmap to disk
    u_hist.flush()
    
    # Save actual snapshot count for loading
    if t_hist[-1] < t:
        if snapshot_idx >= u_hist.shape[0]:
            raise RuntimeError(
                "Snapshot buffer is full before final save. Increase --save-dt or --dt, "
                "or shorten --tn for this run."
            )
        u_hist[snapshot_idx] = u.copy()
        t_hist.append(t)
        snapshot_idx += 1

    n_snapshots_actual = snapshot_idx
    
    # Save time-series payload in a separate file with configurable ZIP compression.
    _save_npz_archive(
        timeseries_file,
        {
            "u_hist": u_hist[:n_snapshots_actual],
            "t_hist": np.array(t_hist, dtype=np.float64),
            # Config values
            "cfg_nx": np.array(cfg.nx, dtype=np.int32),
            "cfg_ny": np.array(cfg.ny, dtype=np.int32),
            "cfg_tn": np.array(cfg.tn, dtype=np.float64),
            "cfg_dt": np.array(cfg.dt, dtype=np.float64),
            "cfg_t0": np.array(cfg.t0, dtype=np.float64),
            "cfg_save_every": np.array(cfg.save_every, dtype=np.int32),
            "cfg_save_dt": np.array(cfg.save_dt, dtype=np.float64),
            "cfg_gradient_threshold": np.array(cfg.gradient_threshold, dtype=np.float64),
            "cfg_sensitivity": np.array(cfg.sensitivity, dtype=np.float64),
            "cfg_min_dt": np.array(cfg.min_dt, dtype=np.float64),
            "cfg_x_min": np.array(cfg.x_min, dtype=np.float64),
            "cfg_x_max": np.array(cfg.x_max, dtype=np.float64),
            "cfg_y_min": np.array(cfg.y_min, dtype=np.float64),
            "cfg_y_max": np.array(cfg.y_max, dtype=np.float64),
            "cfg_n_fields": np.array(module_n_fields, dtype=np.int32),
            "system": np.array(pde_system_path),
        },
        cfg.compression_level,
    )

    # Remove temporary uncompressed memmap after compression.
    if snapshot_file.exists():
        snapshot_file.unlink()

    print(f"Saved to: {timeseries_file}")
    print(f"Snapshots saved: {snapshot_idx} ({'stored' if cfg.compression_level <= 0 else f'compressed level {cfg.compression_level}'} in {timeseries_file})")
    return u, t
    