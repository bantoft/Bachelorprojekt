import sys
import torch
import neuralop as nop

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from SciML.PINO.utils.model import WrappedFNO
from SciML.PINO.utils.data_loader import make_dataloaders

from hesel_scraper.bout_dump import BOUTHESELInfo
from hesel_scraper.bout_phys import BOUTHESELPhys


root = ROOT_DIR / r"sim_data/data_25_512_Alexander"

info = BOUTHESELInfo(root)
phys = BOUTHESELPhys(info)
train_loader, val_loader, test_loader = make_dataloaders(info, batch_size=1)



def build_eq_inputs(info, avg_z, t, coord, prediction):
    batch_size, _, num_x, num_z = coord.shape

    avg_z = avg_z.to(device=coord.device, dtype=coord.dtype)
    avg_z = avg_z.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, num_z)

    x_idx = torch.arange(num_x, device=coord.device, dtype=coord.dtype)
    x_idx = x_idx.view(1, 1, num_x, 1).expand(batch_size, 1, num_x, num_z)

    z_idx = torch.arange(num_z, device=coord.device, dtype=coord.dtype)
    z_idx = z_idx.view(1, 1, 1, num_z).expand(batch_size, 1, num_x, num_z)

    t_idx = torch.round(t / info.parameters.dt).to(dtype=coord.dtype)
    t_idx = t_idx.view(batch_size, 1, 1, 1).expand(batch_size, 1, num_x, num_z)

    cord_num = torch.cat([x_idx, z_idx, t_idx], dim=1)

    state = {
        "lnn": prediction[:, 0:1],
        "lnpe": prediction[:, 1:2],
        "lnpi": prediction[:, 2:3],
        "phi": prediction[:, 3:4],
    }

    return avg_z, state, cord_num


def compute_losses(info, phys, avg_z, prediction, target, t, coord, criterion):
    avg_z, state, cord_num = build_eq_inputs(info, avg_z, t, coord, prediction)
    eq_residuals = phys.eq_res(avg_z, state, coord, cord_num)
    eq_residuals_stack = torch.stack(list(eq_residuals.values()))

    target = target.to(device=prediction.device, dtype=prediction.dtype)
    data_loss = criterion(prediction, target)
    eq_loss = criterion(eq_residuals_stack, torch.zeros_like(eq_residuals_stack))

    return data_loss, eq_loss


def train(model, train_loader, info, phys, epochs=2, lr=1e-3, eq_weight=1.0, max_batches=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_data = 0.0
        running_eq = 0.0
        num_batches = 0

        for batch_idx, (avg_z, u, t, y) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            u = u.to(info.device)
            t = t.to(info.device)

            optimizer.zero_grad(set_to_none=True)

            prediction, coord = model(u, t)
            data_loss, eq_loss = compute_losses(
                info=info,
                phys=phys,
                avg_z=avg_z,
                prediction=prediction,
                target=y,
                t=t,
                coord=coord,
                criterion=criterion,
            )
            total_loss = data_loss + eq_weight * eq_loss
            total_loss.backward()
            optimizer.step()

            running_total += total_loss.item()
            running_data += data_loss.item()
            running_eq += eq_loss.item()
            num_batches += 1

            print(
                f"epoch={epoch + 1} batch={batch_idx + 1} "
                f"total={total_loss.item():.6e} "
                f"data={data_loss.item():.6e} "
                f"eq={eq_loss.item():.6e}"
            )

        if num_batches == 0:
            continue

        print(
            f"epoch={epoch + 1} mean_total={running_total / num_batches:.6e} "
            f"mean_data={running_data / num_batches:.6e} "
            f"mean_eq={running_eq / num_batches:.6e}"
        )

fno = nop.models.FNO(n_modes=(16, 16),
                     in_channels=8,
                     out_channels=4,
                     hidden_channels=64,
                     positional_embedding=None).to(info.device)


wrapped_fno = WrappedFNO(fno, info, phys).to(info.device)

train(
    model=wrapped_fno,
    train_loader=train_loader,
    info=info,
    phys=phys,
    epochs=2,
    lr=1e-3,
    eq_weight=1.0,
    max_batches=5,
)
