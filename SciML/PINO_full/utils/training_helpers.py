from __future__ import annotations

import json
import torch
import torch.nn as nn

from pathlib import Path
from dataclasses import dataclass, field, asdict




@dataclass
class LossGroup:
    total: list[float] = field(default_factory=list)
    da: list[float] = field(default_factory=list)
    eq: list[float] = field(default_factory=list)


@dataclass
class SplitHistory:
    raw: LossGroup = field(default_factory=LossGroup)
    weighted: LossGroup = field(default_factory=LossGroup)

@dataclass
class HistoryData:
    training: SplitHistory = field(default_factory=SplitHistory)
    validation: SplitHistory = field(default_factory=SplitHistory)
    test: SplitHistory = field(default_factory=SplitHistory)


class HistoryBuffer:
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.buffer = HistoryData()

        if not self.save_path.exists(): self._save(HistoryData())

    def append(self, split: str, loss_type: str, values: dict[str, float]):
        split = split.lower()
        group: LossGroup = getattr(getattr(self.buffer, split), loss_type)
        for key, value in values.items():
            getattr(group, key).append(float(value))



    def flush(self):
        history = self._load()
        self._extend(history.training, self.buffer.training)
        self._extend(history.validation, self.buffer.validation)
        self._extend(history.test, self.buffer.test)

        self._save(history)
        self.buffer = HistoryData()


    def _extend(self, history_split: SplitHistory, buffer_split: SplitHistory):
        self._extend_group(history_split.raw, buffer_split.raw)
        self._extend_group(history_split.weighted, buffer_split.weighted)


    def _extend_group(self, history_group: LossGroup, buffer_group: LossGroup):
        for key in vars(history_group):
            getattr(history_group, key).extend(getattr(buffer_group, key))


    def _save(self, history: HistoryData):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(asdict(history), f, indent=2)

    def _load(self) -> HistoryData:
        with open(self.save_path, "r") as f:
            data = json.load(f)

        return HistoryData(
            training=SplitHistory(
                raw=LossGroup(**data["training"]["raw"]),
                weighted=LossGroup(**data["training"]["weighted"]),
            ),
            validation=SplitHistory(
                raw=LossGroup(**data["validation"]["raw"]),
                weighted=LossGroup(**data["validation"]["weighted"]),
            ),
            test=SplitHistory(
                raw=LossGroup(**data["test"]["raw"]),
                weighted=LossGroup(**data["test"]["weighted"]),
            ),
        )


def init_trainer(training_config: dict):
    import neuralop as nop
    from SciML.PINO.utils.model import WrappedFNO
    from SciML.PINO.utils.data_loader import make_dataloaders


    from hesel_scraper.bout_dump import BOUTHESELInfo
    from hesel_scraper.bout_phys import BOUTHESELPhys

    root = Path(__file__).parents[2] / training_config["root"]
    info = BOUTHESELInfo(root)
    phys = BOUTHESELPhys(info)

    condition = nn.MSELoss().to(info.device)

    fno = nop.models.FNO(
        n_modes=(128, 128),
        in_channels=8,      # 4 fields + t + x + z_1 + z_2 #fourier embedder z
        out_channels=4,
        hidden_channels=64,
        positional_embedding= None, # vigtigt: du embedder selv coords
        domain_padding=[0.1, 0.0], # padding kun i x, ikke z
    )

    model = WrappedFNO(fno, info, phys)
    model.to(info.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    train_loader, val_loader, test_loader = make_dataloaders(info,
                                                             batch_size=training_config["batch_size"],
                                                             train_split=training_config["train_split"],
                                                             val_split=training_config["val_split"],
                                                             num_workers=training_config.get("num_workers", 0),
                                                             pin_memory=training_config.get("pin_memory"))

    return (info,
            phys,
            condition,
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader)
