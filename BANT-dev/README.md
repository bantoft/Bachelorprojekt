# BANT-dev

Et letvægts framework til 2D-PDE simulationer med fokus på hurtig iteration:
- modulære PDE-systemer
- RK4-integration med adaptiv tidsstyring
- snapshot/tidsserie-output i `.npz`
- interaktiv visualisering med slider

## Indhold

- `main.py`: CLI entrypoint til simulationer.
- `engine/`: kernekomponenter (grid, integrator, adaptive tidsstep, simulator).
- `PDE_system/`: PDE-moduler, fx `heat_dif` og `complex_coupled_plasma`.
- `visulizer/visualize.py`: interaktiv afspilning af gemte resultater.
- `data/`: anbefalet mappe til outputfiler.

## Krav

Python 3.12+ anbefales.

Pakker, som bruges i koden:
- `numpy`
- `numba`
- `matplotlib`

## Hurtig opsaetning med uv

Koer fra denne mappe (`simulatorer/BANT-dev`):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy numba matplotlib
```

Alternativt uden aktivering:

```bash
uv venv .venv
uv pip install --python .venv/bin/python numpy numba matplotlib
```

## Koer simulation

Generel form:

```bash
uv run main.py --system=<modulnavn> [andre argumenter]
```

`--system` er modulnavnet i `PDE_system/` uden `.py`.

### Eksempel 1: Heat diffusion

```bash
uv run main.py \
  --system=heat_dif \
  --tn=3.37 \
  --dt=0.01 \
  --nx=512 \
  --ny=512 \
  --output=data/heat_dif_sim0.npz \
  --save-dt=0.02 \
  --compression-level=0
```

### Eksempel 2: Koblet plasma-model

```bash
uv run main.py \
  --system=complex_coupled_plasma \
  --tn=5 \
  --dt=1e-4 \
  --save-dt=0.01 \
  --nx=256 \
  --ny=256 \
  --output=data/plasma_run.npz \
  --compression-level=0
```

## Vigtige CLI-parametre

- `--system`: PDE-modul i `PDE_system`.
- `--tn`: sluttid for simulation.
- `--dt`: start-tidsstep (adaptiv styring kan justere undervejs).
- `--nx`, `--ny`: grid-oploesning.
- `--output`: basis-outputsti.
- `--save-dt`: gem snapshot hver `save_dt` tid.
- `--save-every`: deprecieret fallback (`save_dt = dt * save_every`, hvis `--save-dt` ikke gives).
- `--min-dt`: minimum adaptivt tidsstep.
- `--compression-level`: 0-9 for `.npz` komprimering (0 hurtigst, mindst CPU).

## Outputformat

Simulationen skriver en tidsserie-fil:
- `<output_stem>_timeseries.npz`

Den indeholder bl.a.:
- `u_hist`: snapshots af felt(er) over tid
- `t_hist`: tidspunkter for snapshots
- flere `cfg_*` metadatafelter

For enkeltfelt er formen typisk `u_hist.shape = (n_snapshots, nx+2, ny+2)`.
For flerfeltsmodeller (fx plasma) typisk `u_hist.shape = (n_snapshots, n_fields, nx+2, ny+2)`.

## Visualisering

Koer visualizeren paa en gemt `.npz`:

```bash
uv run visulizer/visualize.py data/heat_dif_sim0_timeseries.npz
```

UI viser felter som heatmaps og en tids-slider.

## Tilfoej dit eget PDE-system

1. Opret ny fil i `PDE_system/`, fx `my_system.py`.
2. Implementer mindst:
   - `ic(grid) -> np.ndarray`
   - `rhs(t, u, out, grid, ...) -> None`
3. Tilfoej evt. `create_rhs_kwargs(grid, u)` for preallokerede buffere/parametre.
4. Saet evt. `N_FIELDS` hvis modellen har flere koblede felter.
5. Koer med `--system=my_system`.

## Fejlfinding

- `ModuleNotFoundError` for pakker: installer pakker i den samme interpreter som `uv run` bruger.
- Tom/for kort tidsserie: kontroller `--tn`, `--save-dt` og `--dt`.
- Langsom koersel: saenk `nx/ny`, oeg `save-dt`, og brug lav eller ingen komprimering (`--compression-level=0`).
