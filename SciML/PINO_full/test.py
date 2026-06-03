import gc
import sys
import torch
import neuralop as nop

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from SciML.PINO_full.utils.model import WrappedFNO
from SciML.PINO_full.utils.data_loader import make_dataloaders
from SciML.PINO_full.utils.training_helpers import HistoryBuffer

from hesel_scraper.bout_dump import BOUTHESELInfo
from hesel_scraper.bout_phys import BOUTHESELPhys
def cleanup_phase(model):
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(path, model, optimizer, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def compute_loss(info, phys, model, avg_z, u, t, y, criterion, mode, eq_context, use_checkpoint=True):
    if mode == "data":
        prediction, _ = model(u, t, use_checkpoint=use_checkpoint)
        return criterion(prediction, y)

    row_indices = eq_context["row_indices"]
    prediction_eq, coord_eq = model(u, t, row_indices=row_indices, use_checkpoint=False)
    batch_size, _, num_rows, num_z = prediction_eq.shape

    avg_z_eq = avg_z.index_select(1, row_indices)
    avg_z_eq = avg_z_eq.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, num_z)

    x_idx = eq_context["x_idx"].to(dtype=prediction_eq.dtype)
    x_idx = x_idx.expand(batch_size, 1, num_rows, num_z)
    z_idx = eq_context["z_idx"].to(dtype=prediction_eq.dtype)
    z_idx = z_idx.expand(batch_size, 1, num_rows, num_z)
    t_idx = torch.round(t / info.parameters.dt).to(dtype=prediction_eq.dtype)
    t_idx = t_idx.view(batch_size, 1, 1, 1).expand(batch_size, 1, num_rows, num_z)
    cord_num_eq = torch.cat([x_idx, z_idx, t_idx], dim=1)

    state_eq = {
        "lnn": prediction_eq[:, 0:1],
        "lnpe": prediction_eq[:, 1:2],
        "lnpi": prediction_eq[:, 2:3],
        "phi": prediction_eq[:, 3:4],
    }
    eq_residuals = phys.eq_res(avg_z_eq, state_eq, coord_eq, cord_num_eq)
    eq_residuals_stack = torch.stack(list(eq_residuals.values()))
    return criterion(eq_residuals_stack, torch.zeros_like(eq_residuals_stack))


def evaluate(split, model, loader, info, phys, history=None, eq_weight=1.0, eq_rows=3, max_batches=None):
    criterion = torch.nn.MSELoss()
    row_indices = torch.linspace(
        0,
        info.parameters.num_x - 1,
        steps=eq_rows,
        device=info.device,
    ).round().long().unique()
    eq_context = {
        "row_indices": row_indices,
        "x_idx": row_indices.view(1, 1, -1, 1),
        "z_idx": torch.arange(info.parameters.num_z, device=info.device).view(1, 1, 1, -1),
    }

    was_training = model.training
    model.eval()
    param_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    running_total = 0.0
    running_data = 0.0
    running_eq = 0.0
    num_batches = 0

    try:
        for batch_idx, (avg_z, u, t, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            u = u.to(info.device)
            t = t.to(info.device)
            y = y.to(info.device)
            avg_z = avg_z.to(info.device, dtype=info.dtype)

            losses = {}
            losses["data"] = compute_loss(
                info=info,
                phys=phys,
                model=model,
                avg_z=avg_z,
                u=u,
                t=t,
                y=y,
                criterion=criterion,
                mode="data",
                eq_context=eq_context,
                use_checkpoint=True,
            )
            losses["eq"] = compute_loss(
                info=info,
                phys=phys,
                model=model,
                avg_z=avg_z,
                u=u,
                t=t,
                y=y,
                criterion=criterion,
                mode="eq",
                eq_context=eq_context,
                use_checkpoint=False,
            )

            total_loss = losses["data"] + eq_weight * losses["eq"]
            weighted_losses = {
                "total": total_loss,
                "da": losses["data"],
                "eq": eq_weight * losses["eq"],
            }
            if history is not None:
                history.append(
                    split,
                    "raw",
                    {
                        "total": total_loss.detach().cpu().item(),
                        "da": losses["data"].detach().cpu().item(),
                        "eq": losses["eq"].detach().cpu().item(),
                    },
                )
                history.append(
                    split,
                    "weighted",
                    {
                        "total": weighted_losses["total"].detach().cpu().item(),
                        "da": weighted_losses["da"].detach().cpu().item(),
                        "eq": weighted_losses["eq"].detach().cpu().item(),
                    },
                )

            running_total += total_loss.item()
            running_data += losses["data"].item()
            running_eq += losses["eq"].item()
            num_batches += 1
            del losses, weighted_losses, total_loss, u, t, y, avg_z

        if num_batches == 0:
            return None

        metrics = {
            "total": running_total / num_batches,
            "da": running_data / num_batches,
            "eq": running_eq / num_batches,
        }
        print(
            f"{split} total={metrics['total']:.6e} "
            f"data={metrics['da']:.6e} "
            f"eq={metrics['eq']:.6e}"
        )
        return metrics
    finally:
        for param, requires_grad in zip(model.parameters(), param_requires_grad):
            param.requires_grad_(requires_grad)
        model.train(was_training)


def validate(model, val_loader, info, phys, history=None, eq_weight=1.0, eq_rows=3, max_batches=None):
    return evaluate("validation", model, val_loader, info, phys, history, eq_weight, eq_rows, max_batches)


def test(model, test_loader, info, phys, history=None, eq_weight=1.0, eq_rows=3, max_batches=None):
    return evaluate("test", model, test_loader, info, phys, history, eq_weight, eq_rows, max_batches)


def train(model,
          train_loader,
          val_loader,
          info,
          phys,
          history,
          checkpoint_path,
          epochs=2,
          lr=1e-3,
          eq_weight=1.0,
          eq_rows=3,
          train_max_batches=5,
          val_max_batches=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    row_indices = torch.linspace(
        0,
        info.parameters.num_x - 1,
        steps=eq_rows,
        device=info.device,
    ).round().long().unique()
    eq_context = {
        "row_indices": row_indices,
        "x_idx": row_indices.view(1, 1, -1, 1),
        "z_idx": torch.arange(info.parameters.num_z, device=info.device).view(1, 1, 1, -1),
    }

    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_data = 0.0
        running_eq = 0.0
        num_batches = 0

        for batch_idx, (avg_z, u, t, y) in enumerate(train_loader):
            if train_max_batches is not None and batch_idx >= train_max_batches:
                break

            u = u.to(info.device)
            t = t.to(info.device)
            y = y.to(info.device)
            avg_z = avg_z.to(info.device, dtype=info.dtype)

            optimizer.zero_grad(set_to_none=True)

            losses = {}
            losses["data"] = compute_loss(
                info=info,
                phys=phys,
                model=model,
                avg_z=avg_z,
                u=u,
                t=t,
                y=y,
                criterion=criterion,
                mode="data",
                eq_context=eq_context,
            )

            losses["eq"] = compute_loss(
                info=info,
                phys=phys,
                model=model,
                avg_z=avg_z,
                u=u,
                t=t,
                y=y,
                criterion=criterion,
                mode="eq",
                eq_context=eq_context,
            )
            weighted_losses = {
                "data": losses["data"],
                "eq": eq_weight * losses["eq"],
            }
            loss_total = weighted_losses["data"] + weighted_losses["eq"]
            loss_total.backward()
            optimizer.step()

            history.append(
                "training",
                "raw",
                {
                    "total": loss_total.detach().cpu().item(),
                    "da": losses["data"].detach().cpu().item(),
                    "eq": losses["eq"].detach().cpu().item(),
                },
            )
            history.append(
                "training",
                "weighted",
                {
                    "total": loss_total.detach().cpu().item(),
                    "da": weighted_losses["data"].detach().cpu().item(),
                    "eq": weighted_losses["eq"].detach().cpu().item(),
                },
            )

            total_loss = loss_total.detach()

            running_total += total_loss.item()
            running_data += losses["data"].item()
            running_eq += losses["eq"].item()
            num_batches += 1

            print(
                f"epoch={epoch + 1} batch={batch_idx + 1} "
                f"total={total_loss.item():.6e} "
                f"data={losses['data'].item():.6e} "
                f"eq={losses['eq'].item():.6e}"
            )
            del losses, weighted_losses, loss_total, total_loss, u, t, y, avg_z

        if num_batches == 0:
            continue

        print(
            f"epoch={epoch + 1} mean_total={running_total / num_batches:.6e} "
            f"mean_data={running_data / num_batches:.6e} "
            f"mean_eq={running_eq / num_batches:.6e}"
        )
        cleanup_phase(model)
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            info=info,
            phys=phys,
            history=history,
            eq_weight=eq_weight,
            eq_rows=eq_rows,
            max_batches=val_max_batches,
        )

        history.flush()
        save_checkpoint(checkpoint_path, model, optimizer, epoch + 1)
        cleanup_phase(model)

if __name__ == "__main__":

    root = ROOT_DIR / r"sim_data/data_25_512_Alexander"

    info = BOUTHESELInfo(root)
    phys = BOUTHESELPhys(info)
    train_loader, val_loader, test_loader = make_dataloaders(info,
                                                             batch_size=2,
                                                             num_workers=2,
                                                             prefetch_factor=1,
                                                             pin_memory=True,
                                                             shuffle=False)


    fno = nop.models.FNO(n_modes=(256, 256),
                        in_channels=8,
                        out_channels=4,
                        hidden_channels=32,
                        positional_embedding=None).to(info.device)


    wrapped_fno = WrappedFNO(fno, info, phys).to(info.device)
    out_dir = ROOT_DIR / "SciML/PINO/out"
    history = HistoryBuffer(out_dir / "history.json")
    checkpoint_path = out_dir / "checkpoint.pt"
    eq_weight = 1.0
    eq_rows = 2

    train(
        model=wrapped_fno,
        train_loader=train_loader,
        val_loader=val_loader,
        info=info,
        phys=phys,
        history=history,
        checkpoint_path=checkpoint_path,
        epochs=10,
        lr=1e-4,
        eq_weight=eq_weight,
        eq_rows=eq_rows,
        train_max_batches=1000,
        val_max_batches=1000,
    )
    del train_loader
    cleanup_phase(wrapped_fno)
    del val_loader
    test_metrics = test(
        model=wrapped_fno,
        test_loader=test_loader,
        info=info,
        phys=phys,
        history=history,
        eq_weight=eq_weight,
        eq_rows=eq_rows,
        max_batches=1000,
    )
    if test_metrics is not None:
        history.flush()
    del test_loader
    cleanup_phase(wrapped_fno)
