# Bachelorprojekt

Arbejdsrepository til bachelorprojektet ved DTU om Scientific Machine Learning til plasmasimuleringer.

## Projektet i dag

Projektet er ikke laengere kun en ide eller en samling noter. Repoet er blevet et samlet arbejdsrum med tre hovedspor:

1. `SciML/` indeholder kode til at traene en Physics-Informed Neural Network (PINN) paa data fra `BOUT-HESEL`.
2. `simulatorer/` indeholder baade en let, egenudviklet PDE-simulator til hurtige eksperimenter og den eksterne BOUT/HESEL-kodebase, som data og fysik er bygget op omkring.
3. `skriftlig_arbajde/` samler den skriftlige del af bachelorprojektet, herunder projektplan, ligningsnoter og logbog.

Kort sagt er repoet blevet en kombination af forskningskode, simuleringsmiljoe og projektmateriale.

## Projektfokus

Det faglige fokus er stadig at undersoege, om Scientific Machine Learning kan bruges til at modellere eller accelerere 2D nHESEL-lignende plasmaforloeb. Den nuvaerende kodebase peger isaer i retning af:

- indlaesning af output og konfiguration fra `BOUT-HESEL`
- afledning af fysiske parametre og randbetingelser direkte fra simulatoropsaetningen
- traening af et PINN, som predikterer felterne `lnn`, `lnpe`, `lnpi` og `phi`
- kombination af dataloss og fysikloss, saa modellen baade passer simulerede data og respekterer PDE-strukturen
- hurtig prototyping af enklere PDE-systemer i et separat, lettere simuleringsmiljoe

## Projektdetaljer

**Titel:** Optimizing nHESEL Plasma Simulations using Scientific Machine Learning
**Institution:** Danmarks Tekniske Universitet (DTU)  
**Studerende:** Andreas Bjarnastein Antoft  
**Vejledere:** Jesper Loeve Hinrich, Morten Moerup  
**Medvejleder:** Alexander Simon Thrysoe  
**Periode:** Foraar 2026

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml              # Python-projekt og afhaengigheder
├── uv.lock                     # Laast dependency-resolve til uv
├── SciML/
│   ├── traening.py             # Entrypoint til PINN-traening
│   ├── data/                   # Gemte model/data-artefakter
│   ├── loss_funktion/
│   │   ├── bout_info.py        # Laeser BOUT-settings og afleder fysikparametre
│   │   ├── bout_data.py        # Dataset/dataloader oven paa BOUT-output
│   │   ├── bout_phys.py        # PDE-, rand- og initial-condition-losses
│   │   └── api/                # Hjaelpefunktioner til laesning og operatorer
│   └── utils/
│       ├── model.py            # PINN-modelen
│       └── custom_types.py     # Dataklasser og konfigurationsstrukturer
├── simulatorer/
│   ├── BANT-dev/               # Egen letvaegts 2D-PDE-simulator til prototyper
│   │   ├── main.py
│   │   ├── engine/
│   │   ├── PDE_system/
│   │   └── visulizer/
│   └── BOUT/
│       ├── BOUT-HESEL/         # HESEL-case og output, som SciML-koden laeser fra
│       └── BOUT-dev/           # BOUT++-kodebase/reference
└── skriftlig_arbajde/
    ├── Hesel_ligningerne/      # LaTeX-noter om ligningerne
    ├── Projektplan/            # Projektplan med figurer og kilder
    ├── logbog/                 # Loebende arbejdslog
    ├── opsaetning/             # Miljoe- og setupnoter
    └── projekt_indberettelse/  # Tidlig projektbeskrivelse
```

## `SciML/` i praksis

Den nuvaerende SciML-del er bygget op omkring output fra `simulatorer/BOUT/BOUT-HESEL/data/`. Traeningsscriptet:

- laeser BOUT-konfiguration fra `BOUT.settings`
- laeser felter fra `BOUT.dmp.0.nc`
- afleder normaliserede fysiske parametre fra simulatoropsaetningen
- opretter et dataset over rum-tidspunkter og felter
- traener et PINN med en samlet loss bestaaende af:
  - dataloss
  - PDE-residualer
  - randbetingelser
  - initialbetingelser

Det betyder, at projektet i sin nuvaerende form er taet koblet til BOUT-HESEL som reference- og datakilde.

## `simulatorer/` i praksis

`simulatorer/BANT-dev/` er et separat eksperimentmiljoe til hurtige numeriske tests. Her kan man afproeve enklere PDE-systemer uden hele BOUT-stakken. Det goer mappen nyttig til:

- hurtig iteration paa diskretisering og tidsintegration
- toy-modeller som varmeledning eller koblede plasmafelter
- visualisering af tidsserier gemt som `.npz`

`simulatorer/BOUT/` er derimod den tunge reference-del, som indeholder den eksterne kode og data, der bruges som fysisk fundament for SciML-arbejdet.

## Koer projektet

Projektet bruger `uv` som Python-workflow.

Installer afhaengigheder:

```bash
uv sync
```

Koer PINN-traening fra repo-roden:

```bash
uv run python SciML/træning.py
```

Hvis du vil logge traeningsloss til Weights & Biases, saet dine miljoevariabler foer du starter:

```bash
export WANDB_API_KEY="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"
export WANDB_ENTITY="bantoft-"
export WANDB_PROJECT="Bachelor_projekt"
uv run python træning.py
```

Traeningen logger foelgende metrics for hver batch:

- `train_step/total_loss`
- `train_step/data_loss`
- `train_step/eq_loss`
- `train_step/bc_loss`
- `train_step/ic_loss`

Derudover bliver epoch-gennemsnit ogsaa gemt som:

- `epoch_summary/total_loss`
- `epoch_summary/data_loss`
- `epoch_summary/eq_loss`
- `epoch_summary/bc_loss`
- `epoch_summary/ic_loss`

Traeningen bruger ogsaa early stopping efter hver epoch baseret paa `total_loss`. Som standard stopper den, hvis der ikke er forbedring i `10` epochs.

Den bedste model bliver automatisk gemt lokalt i:

```text
SciML/data/trained_pinn_YYYY-MM-DD_HH-MM-SS.pt
```

Checkpointet indeholder model-vaegte, netvaerksstruktur, traeningskonfiguration og information om bedste epoch/loss.

Runs kan derefter ses i W&B-dashboardet paa:

```text
https://wandb.ai/<entity>/<project>
```

Hvis `WANDB_API_KEY` ikke er sat, koerer scriptet i `offline` mode og logger lokalt. De lokale runs kan uploades senere med:

```bash
wandb sync wandb/
```

Koer den lette PDE-simulator:

```bash
cd simulatorer/BANT-dev
uv run main.py --system=heat_dif
```

## Afhaengigheder

`pyproject.toml` peger i oejeblikket paa disse centrale Python-pakker:

- `torch`
- `numpy`
- `matplotlib`
- `numba`
- `pyarrow`
- `xbout`

Python-versionen er sat til `>=3.12`.

## Bemærkninger

- Repoet er et arbejdsrepo og ikke et faerdigt bibliotek eller reproducibelt release.
- Flere mapper indeholder genererede filer, modeller og LaTeX-build artefakter.
- SciML-koden forventer, at relevante BOUT-HESEL-data findes lokalt i den nuvaerende mappe-struktur.

## Licens

Projektet er udgivet under MIT-licensen. Se [LICENSE](LICENSE).
