from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from engine.config import EngineConfig
from engine.grid import Grid2D
from engine.simulator import simulate_system

def parse_args() -> argparse.Namespace:
    cfg = EngineConfig()
    parser = argparse.ArgumentParser(description="Simulate PDE module from Min_version/PDE_system")
    parser.add_argument("--system", type=str,   default="heat_dif", help="Module in PDE_system folder")
    parser.add_argument("--tn",     type=float, default=cfg.tn,     help="End time for simulation")
    parser.add_argument("--dt",     type=float, default=cfg.dt,     help="Time step size")
    parser.add_argument("--nx",     type=int,   default=cfg.nx,     help="Grid cells in x")
    parser.add_argument("--ny",     type=int,   default=cfg.ny,     help="Grid cells in y")
    parser.add_argument("--output", type=str,   default=cfg.output, help="Output npz path")
    parser.add_argument("--save-dt", type=float, default=None, help="Save snapshot every fixed time interval")
    parser.add_argument("--save-every", type=int, default=cfg.save_every, help="Deprecated: converted to save_dt as dt * save_every")
    parser.add_argument("--min-dt", type=float, default=cfg.min_dt, help="Minimum adaptive time step")
    parser.add_argument("--compression-level", type=int, default=cfg.compression_level, help="ZIP compression level for saved npz files (0 = fastest, no compression; 1-9 = compressed)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dt = args.save_dt if args.save_dt is not None else (args.dt * max(1, args.save_every))
    cfg = EngineConfig(
        tn=args.tn,
        dt=args.dt,
        nx=args.nx,
        ny=args.ny,
        output=args.output,
        save_every=args.save_every,
        save_dt=save_dt,
        min_dt=args.min_dt,
        compression_level=args.compression_level,
    )
    simulate_system(args.system, cfg)
    

if __name__ == "__main__":
    main()


"""
uv run main.py --system=heat_dif --tn=0.1 --dt=0.1 --nx=8 --ny=8 --output=data/start.npz --save-every=1 --compression-level=0

uv run main.py --system=heat_dif --tn=3.37 --dt=0.01 --nx=512 --ny=512 --output=data/heat_dif_sim0.npz --save-every=2 --compression-level=0
"""

"""
uv run main.py --system=heat_dif --tn=0.1 --dt=0.1 --nx=8 --ny=8 --output=data/start.npz --save-every=1 --compression-level=0

uv run main.py   
    --system=complex_coupled_plasma\
    --tn=5\
    --dt=1e-4\
    --save-dt=0.01\
    --nx=256\
    --ny=256\
    --output=data/smoke.npz\
    --compression-level=0
"""

