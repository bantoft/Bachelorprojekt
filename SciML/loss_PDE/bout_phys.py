from __future__ import annotations

import re
import math
import torch


from torch import Tensor

from util.model import PINN
from loss_PDE.bout_dump import BOUTHESELInfo
from loss_PDE.api.operators import grad_x, grad_z, grad_t, laplacian_perp, d2dx2, d2dz2, d2dxdz


class BOUTHESELPhys:
    def __init__(self, info: BOUTHESELInfo):
        self.info = info
        self.x_line = self.info.x_line
        self.z_line = self.info.z_line

        self.ic = self._ic()
        self.bc = self._bc()
        self.scalar_tensors = self._scalar_tensors()
        self.reference_tensors = self._reference_tensors()
        self.device = self.x_line.device

        self.ic_res = self._ic_residuals
        self.bc_res = self._bc_residuals
        self.eq_res = self._eq_residuals

    def to(self, device: torch.device | str):
        device = torch.device(device)
        self.device = device
        self.x_line = self.x_line.to(device)
        self.z_line = self.z_line.to(device)
        self.ic = {name: value.to(device) for name, value in self.ic.items()}
        self.scalar_tensors = {
            name: value.to(device)
            for name, value in self.scalar_tensors.items()
        }
        self.reference_tensors = {
            name: value.to(device)
            for name, value in self.reference_tensors.items()
        }
        return self


    def _x_indices(self, x: Tensor) -> Tensor:
        x_flat = x.reshape(-1)
        dx = float(self.info.parameters.dx)
        tol = max(dx, 1.0) * 1e-4 + 1e-6

        raw_idx = torch.round(x_flat).to(torch.long)
        phys_idx = torch.round(x_flat / dx).to(torch.long)

        raw_like = torch.all((x_flat - raw_idx.to(x_flat.dtype)).abs() <= tol)
        phys_like = torch.all((x_flat - phys_idx.to(x_flat.dtype) * dx).abs() <= tol)

        if raw_like and not phys_like:
            if torch.any(raw_idx >= self.info.parameters.num_x):
                raw_idx = raw_idx - 1
            idx = raw_idx
        else:
            idx = phys_idx

        return idx.clamp(0, self.info.parameters.num_x - 1)


    def _t_indices(self, t: Tensor) -> Tensor:
        t_flat = t.reshape(-1)
        dt = float(self.info.parameters.dt)
        idx = torch.round(t_flat / dt).to(torch.long)
        return idx.clamp(0, self.info.parameters.num_t - 1)


    def _get_vort_field(self, lnpi: Tensor, phi: Tensor, x: Tensor, z: Tensor):
        vort = laplacian_perp(phi + torch.exp(lnpi), x, z)
        return vort.reshape(phi.shape).to(phi.dtype)


    def _scalar_tensors(self) -> dict[str, Tensor]:
        dtype = self.x_line.dtype
        params = self.info.parameters

        def scalar(value: float) -> Tensor:
            return torch.tensor(float(value), dtype=dtype)

        return {
            "x_min": self.x_line[0].detach().clone(),
            "x_max": self.x_line[-1].detach().clone(),
            "z_min": self.z_line[0].detach().clone(),
            "z_max": self.z_line[-1].detach().clone(),
            "curvature_scale": scalar(params.rhos / (params.rmajor + params.rminor)),
            "ti0_over_te0": scalar(params.ti0 / params.te0),
            "norm_de": scalar(params.norm_de),
            "norm_di": scalar(params.norm_di),
            "norm_eta": scalar(params.norm_eta),
            "bohm_potential": scalar(params.bohm_potential),
            "norm_taun": scalar(params.norm_taun),
            "norm_lc": scalar(params.norm_lc),
            "norm_taudw": scalar(params.norm_taudw),
            "norm_taushe": scalar(params.norm_taushe),
            "norm_taushi": scalar(params.norm_taushi),
            "force_time": scalar(params.force_time),
            "floor_time": scalar(params.floor_time),
            "floor_n": scalar(params.floor_n),
            "floor_pe": scalar(params.floor_pe),
            "floor_pi": scalar(params.floor_pi),
            "te_bck": scalar(params.te_bck),
            "ti_bck": scalar(params.ti_bck),
            "z_eff": scalar(params.z_eff),
            "me_over_mi": scalar(params.me / params.mi),
            "nuei_over_oci": scalar(params.nuei / params.oci),
        }


    def _reference_tensors(self) -> dict[str, Tensor]:
        dtype = self.x_line.dtype
        ref_lnn = self.info.data.lnn.to(dtype=dtype)
        ref_lnpe = self.info.data.lnpe.to(dtype=dtype)
        ref_lnpi = self.info.data.lnpi.to(dtype=dtype)
        ref_phi = self.info.data.phi.to(dtype=dtype)

        b_profile = self.info.data.B.to(dtype=dtype)
        if b_profile.ndim > 1:
            b_profile = b_profile.mean(dim=tuple(range(1, b_profile.ndim)))
        b_profile = b_profile.reshape(-1)

        full_length = torch.clamp(
            self.x_line[-1], min=torch.finfo(self.x_line.dtype).eps
        )
        x_func = self.x_line / full_length

        return {
            "avg_n": torch.exp(ref_lnn).mean(dim=-1),
            "avg_te": torch.exp(ref_lnpe - ref_lnn).mean(dim=-1),
            "avg_ti": torch.exp(ref_lnpi - ref_lnn).mean(dim=-1),
            "avg_phi": ref_phi.mean(dim=-1),
            "avg_tau": torch.exp(ref_lnpi - ref_lnpe).mean(dim=-1),
            "avg_cs_hot": torch.sqrt(
                torch.clamp(
                    torch.exp(ref_lnpe - ref_lnn) + torch.exp(ref_lnpi - ref_lnn),
                    min=1e-12,
                )
            ).mean(dim=-1),
            "b_profile": b_profile,
            "sigma_open": self.info.functions.sigma_open_function(x_func).reshape(-1),
            "sigma_closed": self.info.functions.sigma_closed_function(x_func).reshape(-1),
            "sigma_force": self.info.functions.sigma_force_function(x_func).reshape(-1),
            "init_n": torch.exp(self.ic["lnn"]).reshape(-1),
            "init_pe": torch.exp(self.ic["lnpe"]).reshape(-1),
            "init_pi": torch.exp(self.ic["lnpi"]).reshape(-1),
        }



    def _ic(self) -> dict[str, Tensor]:
        full_length = torch.clamp(
            self.x_line[-1], min=torch.finfo(self.x_line.dtype).eps
        )
        x = self.x_line / full_length

        return {
            "lnn": torch.log(torch.clamp(self.info.functions.init_n_function(x), min=1e-12)),
            "lnpe": torch.log(torch.clamp(self.info.functions.init_pe_function(x), min=1e-12)),
            "lnpi": torch.log(torch.clamp(self.info.functions.init_pi_function(x), min=1e-12)),
            "vort": torch.full((self.info.parameters.num_x,),float(self.info.settings[("vort", "function")]),device=x.device,dtype=x.dtype,),
        }


    def _ic_residuals(self, ic_pred: dict[str, Tensor], x: Tensor, z: Tensor):
        x_idx = self._x_indices(x)
        vort = self._get_vort_field(ic_pred["lnpi"], ic_pred["phi"], x, z).view_as(ic_pred["lnpe"])

        ic = {
            name: value[x_idx].view(-1, 1)
            for name, value in self.ic.items()
        }
        return {
            "ic_n": ic_pred["lnn"] - ic["lnn"],
            "ic_pe": ic_pred["lnpe"] - ic["lnpe"],
            "ic_pi": ic_pred["lnpi"] - ic["lnpi"],
            "ic_vort": vort - ic["vort"],
        }


    def _bc(self):
        """
        Returns boundary conditions: dict{field: dict{side: (bc_type, value)}}.
        """
        boundary_conditions = {}
        allowed_fields = {"lnn", "lnpe", "lnpi", "vort"}
        for (field, key), value in self.info.settings.items():
            if field not in allowed_fields or "bndry" not in key:
                continue

            side_match = re.search(r"bndry_(\w+)", key)
            if side_match is None:
                continue
            side = side_match.group(1)

            raw_value = str(value).strip()
            type_match = re.match(r"(\w+)", raw_value)
            if type_match is None:
                continue
            bc_type = type_match.group(1)

            expr_match = re.match(r"^\w+_o\d+\((.*)\)$", raw_value)
            if expr_match is not None:
                expr = expr_match.group(1).strip()
                parsed_value = eval(expr, {"__builtins__": {}, **vars(math)})
            else:
                parsed_value = value

            boundary_conditions.setdefault(field, {})[side] = (bc_type, parsed_value)
        return boundary_conditions
    

    def _bc_residuals(self, standardization,  model: PINN, x: Tensor, z: Tensor, t: Tensor, state_inputs: Tensor):
        x_bounds = {
            "xin": (torch.ones_like(x) * self.scalar_tensors["x_min"]).detach().requires_grad_(True),
            "xout": (torch.ones_like(x) * self.scalar_tensors["x_max"]).detach().requires_grad_(True),
        }
        z_bounds = {
            "zlower": (torch.ones_like(z) * self.scalar_tensors["z_min"]).detach().requires_grad_(True),
            "zupper": (torch.ones_like(z) * self.scalar_tensors["z_max"]).detach().requires_grad_(True),
        }

        boundary_predictions = {
            "xin": model.forward(standardization, x_bounds["xin"], z, t, state_inputs),
            "xout": model.forward(standardization, x_bounds["xout"], z, t, state_inputs),
            "zlower": model.forward(standardization, x, z_bounds["zlower"], t, state_inputs),
            "zupper": model.forward(standardization, x, z_bounds["zupper"], t, state_inputs),
        }

        for side, pred in boundary_predictions.items():
            x_side = x_bounds[side] if side in x_bounds else x
            z_side = z_bounds[side] if side in z_bounds else z
            pred["vort"] = self._get_vort_field(pred["lnpi"], pred["phi"], x_side, z_side).view_as(pred["lnpe"])

        x_side_preds = {"xin": boundary_predictions["xin"], "xout": boundary_predictions["xout"]}
        results = {}
        for field, specification in self.bc.items():
            for side, (bc_type, value) in specification.items():
                side_name = side.lower()
                bc_name = f"{field}_bc_{side}_{bc_type}"
                for x_side in ("xin", "xout"):
                    if x_side not in side_name:
                        continue

                    pred = x_side_preds[x_side][field]
                    target = torch.full_like(pred, float(value))
                    if "dirichlet" in bc_type.lower():
                        results[bc_name] = pred - target
                    elif "neumann" in bc_type.lower():
                        results[bc_name] = grad_z(pred, z).view_as(pred) - target
                    break

            results[f"{field}_bc_periodic"] = boundary_predictions["zlower"][field] - boundary_predictions["zupper"][field]
        return results
    
    
    def _eq_residuals(
        self,
        state: dict[str, Tensor], # Predicted state at current time step
        x: Tensor,
        z: Tensor,
        t: Tensor,
    ):
        settings = self.info.settings

        lnn, lnpe, lnpi, phi = state["lnn"], state["lnpe"], state["lnpi"], state["phi"]
        n, pe, pi = torch.exp(lnn), torch.exp(lnpe), torch.exp(lnpi)
        lnte, lnti = lnpe - lnn, lnpi - lnn
        te, ti = torch.exp(lnte), torch.exp(lnti)
        tau = torch.exp(lnti - lnte)
        cs_hot = torch.sqrt(torch.clamp(ti + te, min=1e-12))
        x_idx = self._x_indices(x)
        t_idx = self._t_indices(t)
        refs = self.reference_tensors
        scalars = self.scalar_tensors

        avg_n = refs["avg_n"][t_idx, x_idx].to(dtype=n.dtype).view_as(n)
        avg_te = refs["avg_te"][t_idx, x_idx].to(dtype=te.dtype).view_as(te)
        avg_ti = refs["avg_ti"][t_idx, x_idx].to(dtype=ti.dtype).view_as(ti)
        avg_phi = refs["avg_phi"][t_idx, x_idx].to(dtype=phi.dtype).view_as(phi)
        avg_tau = refs["avg_tau"][t_idx, x_idx].to(dtype=tau.dtype).view_as(tau)
        avg_cs_hot = refs["avg_cs_hot"][t_idx, x_idx].to(dtype=cs_hot.dtype).view_as(cs_hot)
        b_field = refs["b_profile"][x_idx].to(dtype=n.dtype).view_as(n)
        inv_b = 1.0 / torch.clamp(b_field, min=1e-12)

        sigma_open = refs["sigma_open"][x_idx].to(dtype=n.dtype).view_as(n)
        sigma_closed = refs["sigma_closed"][x_idx].to(dtype=n.dtype).view_as(n)
        sigma_force = refs["sigma_force"][x_idx].to(dtype=n.dtype).view_as(n)

        init_n = refs["init_n"][x_idx].to(dtype=n.dtype).view_as(n)
        init_pe = refs["init_pe"][x_idx].to(dtype=pe.dtype).view_as(pe)
        init_pi = refs["init_pi"][x_idx].to(dtype=pi.dtype).view_as(pi)

        def ddx(field: Tensor) -> Tensor:
            return grad_x(field, x).reshape(field.shape).to(field.dtype)

        def ddz(field: Tensor) -> Tensor:
            return grad_z(field, z).reshape(field.shape).to(field.dtype)

        def ddt(field: Tensor) -> Tensor:
            return grad_t(field, t).reshape(field.shape).to(field.dtype)

        def lap(field: Tensor) -> Tensor:
            return laplacian_perp(field, x, z).reshape(field.shape).to(field.dtype)

        def ddxx(field: Tensor) -> Tensor:
            return d2dx2(field, x).reshape(field.shape).to(field.dtype)

        def ddzz(field: Tensor) -> Tensor:
            return d2dz2(field, z).reshape(field.shape).to(field.dtype)

        def ddxz(field: Tensor) -> Tensor:
            return d2dxdz(field, x, z).reshape(field.shape).to(field.dtype)

        dphi_dx, dphi_dz = ddx(phi), ddz(phi)
        dpi_dx, dpi_dz = ddx(pi), ddz(pi)
        dlnn_dx, dlnn_dz = ddx(lnn), ddz(lnn)
        dlnte_dx, dlnte_dz = ddx(lnte), ddz(lnte)
        dlnti_dx, dlnti_dz = ddx(lnti), ddz(lnti)
        ddt_lnn, ddt_lnpe, ddt_lnpi = ddt(lnn), ddt(lnpe), ddt(lnpi)
        vort = self._get_vort_field(lnpi, phi, x, z)
        ddt_vort = ddt(vort)


        def brackets(f: Tensor, g: Tensor) -> Tensor:
            bracket = ddx(f) * ddz(g) - ddz(f) * ddx(g)
            return -bracket if settings["hesel", "right_handed_coord"] else bracket

        def curvature(f: Tensor) -> Tensor:
            curv = (
                (2.0 if settings["hesel", "double_curvature_coeff"] else 1.0)
                * scalars["curvature_scale"]
                * ddz(f)
            )
            return -curv if settings["hesel", "right_handed_coord"] else curv

        interchange = {name: torch.zeros_like(vort if name == "vort" else lnn) for name in ("lnn", "lnpe", "lnpi", "vort")}
        if settings["hesel", "interchange_dynamics"]:
            interchange["lnn"] = -inv_b * brackets(phi, lnn) - curvature(phi) + curvature(te) + curvature(lnn) * te
            interchange["lnpe"] = -inv_b * brackets(phi, lnpe) - 5.0 / 3.0 * curvature(phi) + 5.0 / 3.0 * curvature(te) + 5.0 / 3.0 * curvature(lnpe) * te
            interchange["lnpi"] = -inv_b * brackets(phi, lnpi) - 5.0 / 3.0 * curvature(phi) - 5.0 / 3.0 * curvature(ti) - 5.0 / 3.0 * curvature(lnpi) * ti + 2.0 / 3.0 * curvature(pe + pi)
            interchange["vort"] = -brackets(phi, vort) + curvature(pe + pi)
            if not settings["hesel", "test_vort_cross_term"]:
                interchange["vort"] = interchange["vort"] - brackets(dphi_dx, dpi_dx) - brackets(dphi_dz, dpi_dz)

        ti_rcpte = (
            torch.ones_like(tau) * scalars["ti0_over_te0"]
            if settings["hesel", "ti_over_te"] == 1
            else avg_tau if settings["hesel", "ti_over_te"] == 2 else tau
        )
        if settings["hesel", "diffusion_coeff"] == 1:
            de = torch.ones_like(n) * scalars["norm_de"]
            di = torch.ones_like(n) * scalars["norm_di"]
        elif settings["hesel", "diffusion_coeff"] == 2:
            de = scalars["norm_de"] * avg_n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = scalars["norm_di"] * avg_n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 3:
            de = scalars["norm_de"] * n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2)
            di = scalars["norm_di"] * n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 4:
            de = scalars["norm_de"] * avg_n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2)
            di = scalars["norm_di"] * avg_n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 5:
            de = scalars["norm_de"] * n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = scalars["norm_di"] * n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 6:
            de = torch.ones_like(n) * scalars["norm_de"] / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = torch.ones_like(n) * scalars["norm_di"] / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        else:
            raise ValueError("Unsupported diffusion_coeff option from BOUT-HESEL.")

        de, di = de * scalars["z_eff"], di * scalars["z_eff"]
        dn = de * (1.0 + ti_rcpte)
        if settings["hesel", "collisional_model"] == 0:
            collisional = {name: torch.zeros_like(term) for name, term in interchange.items()}
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif settings["hesel", "collisional_model"] == 1:
            collisional = {
                "lnn": scalars["norm_de"] * (1.0 + scalars["ti0_over_te0"]) * lap(lnn),
                "lnpe": 2.0 / 3.0 * scalars["norm_de"] * (1.0 + scalars["ti0_over_te0"]) * lap(lnpe),
                "lnpi": 2.0 / 3.0 * 2.0 * scalars["norm_di"] * lap(lnti),
                "vort": scalars["norm_eta"] * lap(vort),
            }
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif settings["hesel", "collisional_model"] == 2:
            deln_rcpn = lap(lnn) + dlnn_dx * dlnn_dx + dlnn_dz * dlnn_dz
            delte_rcpte = lap(lnte) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = lap(lnti) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            gradn_gradte_rcppe = dlnn_dx * dlnte_dx + dlnn_dz * dlnte_dz
            gradn_gradti_rcppi = dlnn_dx * dlnti_dx + dlnn_dz * dlnti_dz
            collisional = {
                "lnn": dn * deln_rcpn,
                "lnpe": 2.0 / 3.0 * dn * (deln_rcpn + gradn_gradte_rcppe) + 2.0 / 3.0 * 29.0 / 12.0 * de * (delte_rcpte + gradn_gradte_rcppe),
                "lnpi": 2.0 / 3.0 * 5.0 / 2.0 * dn * (deln_rcpn + gradn_gradti_rcppi) + 2.0 / 3.0 * 2.0 * di * (delti_rcpti + gradn_gradti_rcppi),
                "vort": scalars["norm_eta"] * lap(vort),
            }
            u_r_x, u_r_z = -dn * dlnn_dx, -dn * dlnn_dz
        elif settings["hesel", "collisional_model"] == 3:
            delte_rcpte = lap(lnte) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = lap(lnti) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            u_r_x = -de * ((1.0 + ti_rcpte) * dlnn_dx + dlnti_dx * ti_rcpte - 0.5 * dlnte_dx)
            u_r_z = -de * ((1.0 + ti_rcpte) * dlnn_dz + dlnti_dz * ti_rcpte - 0.5 * dlnte_dz)
            div_gamma_r_rcpn = dlnn_dx * u_r_x + dlnn_dz * u_r_z + ddx(u_r_x) + ddz(u_r_z)
            collisional = {
                "lnn": -div_gamma_r_rcpn,
                "lnpe": -2.0 / 3.0 * (div_gamma_r_rcpn + dlnte_dx * u_r_x + dlnte_dz * u_r_z) + 2.0 / 3.0 * 29.0 / 12.0 * ((ddx(de) + de * dlnn_dx) * dlnte_dx + (ddz(de) + de * dlnn_dz) * dlnte_dz + de * delte_rcpte),
                "lnpi": -2.0 / 3.0 * 5.0 / 2.0 * (div_gamma_r_rcpn + dlnti_dx * u_r_x + dlnti_dz * u_r_z) + 2.0 / 3.0 * 2.0 * ((ddx(di) + di * dlnn_dx) * dlnti_dx + (ddz(di) + di * dlnn_dz) * dlnti_dz + di * delti_rcpti),
                "vort": scalars["norm_eta"] * lap(vort),
            }
        else:
            raise ValueError("Unsupported collisional_model option from BOUT-HESEL.")

        heat_exchange = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel", "perpend_heat_exchange"]:
            q_resist_rcppi = u_r_x * (dlnn_dx + dlnti_dx) + u_r_z * (dlnn_dz + dlnti_dz)
            if settings["hesel", "qdelta_approx"] == 0:
                qdelta_rcppe = torch.zeros_like(n)
            elif settings["hesel", "qdelta_approx"] == 1:
                qdelta_rcppe = 3.0 * scalars["me_over_mi"] * scalars["nuei_over_oci"] * (1.0 - ti_rcpte)
            elif settings["hesel", "qdelta_approx"] == 2:
                qdelta_rcppe = 3.0 * scalars["me_over_mi"] * scalars["nuei_over_oci"] * (1.0 - ti_rcpte) * avg_n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif settings["hesel", "qdelta_approx"] == 3:
                qdelta_rcppe = 3.0 * scalars["me_over_mi"] * scalars["nuei_over_oci"] * (1.0 - ti_rcpte) * n / te / torch.sqrt(torch.clamp(te, min=1e-12))
            elif settings["hesel", "qdelta_approx"] == 4:
                qdelta_rcppe = 3.0 * scalars["me_over_mi"] * scalars["nuei_over_oci"] * (1.0 - ti_rcpte) * n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            else:
                raise ValueError("Unsupported qdelta_approx option from BOUT-HESEL.")
            heat_exchange["lnpe"] = -2.0 / 3.0 * q_resist_rcppi * ti_rcpte - 2.0 / 3.0 * qdelta_rcppe
            heat_exchange["lnpi"] = 2.0 / 3.0 * q_resist_rcppi + 2.0 / 3.0 * qdelta_rcppe / torch.clamp(ti_rcpte, min=1e-12)

        viscous = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel", "perpend_viscous_heating"]:
            field_sum = phi + pi
            qviscous = 3.0 / 10.0 * di * ((ddxx(field_sum) - ddzz(field_sum)) ** 2 + 4.0 * ddxz(field_sum) ** 2) / torch.clamp(ti, min=1e-12)
            viscous["lnpi"] = 2.0 / 3.0 * qviscous

        perpendicular = {name: collisional[name] + heat_exchange[name] + viscous[name] for name in interchange}
        if not settings["hesel","perpendicular_dynamics"]:
            perpendicular = {name: torch.zeros_like(term) for name, term in interchange.items()}

        parallel = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel","parallel_dynamics"]:
            if settings["hesel","parallel_advection_damping"] == 0:
                damp_advection = torch.zeros_like(n)
            elif settings["hesel","parallel_advection_damping"] == 1:
                damp_advection = torch.ones_like(n) / scalars["norm_taun"]
            elif settings["hesel","parallel_advection_damping"] == 2:
                damp_advection = avg_cs_hot / scalars["norm_taun"]
            elif settings["hesel","parallel_advection_damping"] == 3:
                damp_advection = cs_hot / scalars["norm_taun"]
            else:
                raise ValueError("Unsupported parallel_advection_damping option from BOUT-HESEL.")
            parallel["lnn"] = parallel["lnn"] - sigma_open * damp_advection
            parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["vort"] = parallel["vort"] - sigma_open * damp_advection * vort

            if settings["hesel","parallel_sheath_damping"] == 0:
                damp_sheath = torch.zeros_like(n)
            elif settings["hesel","parallel_sheath_damping"] == 1:
                damp_sheath = (1.0 / scalars["norm_lc"]) * (1.0 - torch.exp(scalars["bohm_potential"] - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif settings["hesel","parallel_sheath_damping"] == 2:
                damp_sheath = avg_cs_hot / scalars["norm_lc"] * (1.0 - torch.exp(scalars["bohm_potential"] - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif settings["hesel","parallel_sheath_damping"] == 3:
                damp_sheath = cs_hot / scalars["norm_lc"] * (1.0 - torch.exp(scalars["bohm_potential"] - phi / torch.clamp(te, min=1e-12)))
            else:
                raise ValueError("Unsupported parallel_sheath_damping option from BOUT-HESEL.")
            parallel["lnpi"] = parallel["lnpi"] + 2.0 / 3.0 * sigma_open * damp_sheath
            parallel["vort"] = parallel["vort"] + sigma_open * damp_sheath

            if settings["hesel","parallel_conduction"] == 0:
                damp_she = torch.zeros_like(n)
                damp_shi = torch.zeros_like(n)
            elif settings["hesel","parallel_conduction"] == 1:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / scalars["norm_taushe"]
                damp_shi = torch.zeros_like(n)
            elif settings["hesel","parallel_conduction"] == 2:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / scalars["norm_taushe"]
                damp_shi = ti * ti * torch.sqrt(torch.clamp(ti, min=1e-12)) / scalars["norm_taushi"]
            else:
                raise ValueError("Unsupported parallel_conduction option from BOUT-HESEL.")

            pert_n, pert_te, pert_phi = n - avg_n, te - avg_te, phi - avg_phi
            if settings["hesel","parallel_drift_wave"] == 0:
                driftwave = torch.zeros_like(n)
            elif settings["hesel","parallel_drift_wave"] == 1:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / scalars["norm_taudw"]
            elif settings["hesel","parallel_drift_wave"] == 2:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / scalars["norm_taudw"] * avg_te * torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif settings["hesel","parallel_drift_wave"] == 3:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / scalars["norm_taudw"] * te * torch.sqrt(torch.clamp(te, min=1e-12))
            else:
                raise ValueError("Unsupported parallel_drift_wave option from BOUT-HESEL.")
            parallel["vort"] = parallel["vort"] - sigma_closed * driftwave

            if settings["hesel","reciprocal_approx"] == 1:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti
            elif settings["hesel","reciprocal_approx"] == 2:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi / torch.clamp(avg_n, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave
            elif settings["hesel","reciprocal_approx"] == 3:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she * (te - scalars["te_bck"]) / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi * (te - scalars["ti_bck"]) / torch.clamp(pi, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti / torch.clamp(pi, min=1e-12)
            else:
                raise ValueError("Unsupported reciprocal_approx option from BOUT-HESEL.")

        forcing = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel","force_profiles"]:
            if not settings["hesel","not_n_force"]:
                forcing["lnn"] = forcing["lnn"] + sigma_force * (init_n / torch.clamp(n, min=1e-12) - 1.0) / scalars["force_time"]
            if not settings["hesel","not_p_force"]:
                forcing_multiplier = torch.ones_like(pi)
                if settings["hesel","h_mode"]:
                    t_phys = t
                    forcing_multiplier = 1.0 + (settings["hesel","ramp_a"] - 1.0) / 2.0 * (torch.tanh((t_phys - settings["hesel","ramp_t0"]) / settings["hesel","ramp_trans"]) - torch.tanh((t_phys - settings["hesel","ramp_t0"] - settings["hesel","ramp_peak"]) / settings["hesel","ramp_trans"]))
                force_pe = sigma_force * (init_pe - pe) / scalars["force_time"]
                force_pi = sigma_force * (init_pi * forcing_multiplier - pi) / scalars["force_time"]
                forcing["lnpe"] = forcing["lnpe"] + force_pe / torch.clamp(pe, min=1e-12)
                forcing["lnpi"] = forcing["lnpi"] + force_pi / torch.clamp(pi, min=1e-12)

        floor_terms = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel","floor_profiles"]:
            floor_terms["lnn"] = floor_terms["lnn"] + torch.where(n < scalars["floor_n"], (scalars["floor_n"] / torch.clamp(n, min=1e-12) - 1.0) / scalars["floor_time"], torch.zeros_like(n))
            floor_terms["lnpe"] = floor_terms["lnpe"] + torch.where(pe < scalars["floor_pe"], (scalars["floor_pe"] / torch.clamp(pe, min=1e-12) - 1.0) / scalars["floor_time"], torch.zeros_like(pe))
            floor_terms["lnpi"] = floor_terms["lnpi"] + torch.where(pi < scalars["floor_pi"], (scalars["floor_pi"] / torch.clamp(pi, min=1e-12) - 1.0) / scalars["floor_time"], torch.zeros_like(pi))


        rhs_terms = interchange
        lambda_terms = {
            name: perpendicular[name] + parallel[name] + forcing[name] + floor_terms[name]
            for name in interchange
        }

        return {
            "eq_lnn": ddt_lnn - rhs_terms["lnn"] - lambda_terms["lnn"],
            "eq_lnpe": ddt_lnpe - rhs_terms["lnpe"] - lambda_terms["lnpe"],
            "eq_lnpi": ddt_lnpi - rhs_terms["lnpi"] - lambda_terms["lnpi"],
            "eq_vort": ddt_vort - rhs_terms["vort"] - lambda_terms["vort"],
        }
