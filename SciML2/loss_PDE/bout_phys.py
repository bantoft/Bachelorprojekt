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


        self.ic = self._ic()
        self.bc = self._bc()
        self.ic_res = self._ic_residuals
        self.bd_res = self._bc_residuals
        self.eq_res = self._eq_residuals

    def _broadcast_x_profile(self, values: Tensor, reference: Tensor) -> Tensor:
        values = values.to(reference.dtype).to(reference.device)
        view_shape = (1,) * max(reference.ndim - 3, 0) + (values.shape[0], 1, 1)
        return values.view(view_shape)

    def _mean_z(self, values: Tensor) -> Tensor:
        return values.mean(dim=-2, keepdim=True)

    def _boundary_slice(self, field: Tensor, side: str) -> Tensor:
        x_idx = 0 if side == "xin" else -1
        return field.select(dim=-3, index=x_idx)

    def _function_x_line(self) -> Tensor:
        full_length = torch.clamp(self.x_line[-1], min=torch.finfo(self.x_line.dtype).eps)
        return self.x_line / full_length
    

    def _ic(self) -> Tensor:
        """
        Returns initial condition: (3, nx)
        """
        x = self._function_x_line()
        ic_n =  torch.log(torch.clamp(self.info.functions.init_n_function(x), min=1e-12))
        ic_pe = torch.log(torch.clamp(self.info.functions.init_pe_function(x), min=1e-12))
        ic_pi = torch.log(torch.clamp(self.info.functions.init_pi_function(x), min=1e-12))
        ic_vort = torch.full((self.info.parameters.num_x,), float(self.info.settings[("vort", "function")]),)
        return torch.stack((ic_n, ic_pe, ic_pi, ic_vort), dim=0)


    def _ic_residuals(
        self,
        state_t: dict[str, Tensor], # Predicted state at initial time step (batch, 4, nx, nz)
        x: Tensor,
        z: Tensor,
    ):
        pi = state_t["lnpi"]
        phi = state_t["phi"]
        vort = laplacian_perp(phi + pi, x, z)

        init_n, init_pe, init_pi, init_vort = self.ic
        init_n = self._broadcast_x_profile(init_n, state_t["lnn"])
        init_pe = self._broadcast_x_profile(init_pe, state_t["lnpe"])
        init_pi = self._broadcast_x_profile(init_pi, state_t["lnpi"])
        init_vort = self._broadcast_x_profile(init_vort, state_t["phi"])

        return {
            "ic_n": state_t["lnn"] - init_n,
            "ic_pe": state_t["lnpe"] - init_pe,
            "ic_pi": state_t["lnpi"] - init_pi,
            "ic_vort": vort - init_vort,
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
    

    def _bc_residuals(
        self,
        state: dict[str, Tensor], # Predicted state at current time step
        x: Tensor,
        z: Tensor,
    ):
        pi = torch.exp(state["lnpi"])
        phi = state["phi"]
        vort = laplacian_perp(phi + pi, x, z)

        fields = {
            "lnn": state["lnn"],
            "lnpe": state["lnpe"],
            "lnpi": state["lnpi"],
            "vort": vort,
        }
        x_derivatives = {
            name: grad_x(field, x)
            for name, field in fields.items()
        }

        residuals = {}
        for field_name, side_map in self.bc.items():
            field = fields.get(field_name)
            ddx_field = x_derivatives.get(field_name)
            if field is None or ddx_field is None:
                continue

            for side, (bc_type, bc_value) in side_map.items():
                key = f"bc_{field_name}_{side}"
                if "dirichlet" in bc_type.lower():
                    residuals[key] = self._boundary_slice(field, side) - bc_value
                elif "neumann" in bc_type.lower():
                    residuals[key] = self._boundary_slice(ddx_field, side) - bc_value

        return residuals

    def _eq_residuals(
        self,
        state: dict[str, Tensor], # Predicted state at current time step
        x: Tensor,
        z: Tensor,
        t: Tensor,
    ):
        
        settings = self.info.settings
        params = self.info.parameters

        lnn, lnpe, lnpi, phi = state["lnn"], state["lnpe"], state["lnpi"], state["phi"]
        n, pe, pi = torch.exp(lnn), torch.exp(lnpe), torch.exp(lnpi)
        lnte, lnti = lnpe - lnn, lnpi - lnn
        te, ti = torch.exp(lnte), torch.exp(lnti)
        tau = torch.exp(lnti - lnte)
        cs_hot = torch.sqrt(torch.clamp(ti + te, min=1e-12))
        avg_n, avg_te, avg_ti, avg_phi, avg_tau, avg_cs_hot = (
            self._mean_z(value) for value in (n, te, ti, phi, tau, cs_hot)
        )
        

        # Laver field lines
        # Kunne ikke finde funktion for B så tager den der er i data:
        b_field = self._broadcast_x_profile(self.info.data.B, n)
        inv_b = 1.0 / torch.clamp(b_field, min=1e-12)
        

        x_func = self._function_x_line()
        sigma_open = self._broadcast_x_profile(self.info.functions.sigma_open_function(x_func), n)
        sigma_closed = self._broadcast_x_profile(self.info.functions.sigma_closed_function(x_func), n)
        sigma_force = self._broadcast_x_profile(self.info.functions.sigma_force_function(x_func), n)

        init_n, init_pe, init_pi, _ = self.ic
        init_n = self._broadcast_x_profile(init_n, n)
        init_pe = self._broadcast_x_profile(init_pe, n)
        init_pi = self._broadcast_x_profile(init_pi, n)
        

        dphi_dx, dphi_dz = grad_x(phi, x), grad_z(phi, z)
        dpi_dx, dpi_dz = grad_x(pi, x), grad_z(pi, z)
        dlnn_dx, dlnn_dz = grad_x(lnn, x), grad_z(lnn, z)
        dlnte_dx, dlnte_dz = grad_x(lnte, x), grad_z(lnte, z)
        dlnti_dx, dlnti_dz = grad_x(lnti, x), grad_z(lnti, z)
        ddt_lnn, ddt_lnpe, ddt_lnpi = grad_t(lnn, t), grad_t(lnpe, t), grad_t(lnpi, t)
        vort = laplacian_perp(phi + pi, x, z)
        ddt_vort = grad_t(vort, t)

        def brackets(f: Tensor, g: Tensor) -> Tensor:
            bracket = grad_x(f, x) * grad_z(g, z) - grad_z(f, z) * grad_x(g, x)
            return -bracket if settings["hesel", "right_handed_coord"] else bracket

        def curvature(f: Tensor) -> Tensor:
            curv = (2.0 if settings["hesel", "double_curvature_coeff"] else 1.0) * params.rhos / (params.rmajor + params.rminor) * grad_z(f, z)
            return -curv if settings["hesel", "right_handed_coord"] else curv

        interchange = {name: torch.zeros_like(vort if name == "vort" else lnn) for name in ("lnn", "lnpe", "lnpi", "vort")}
        if settings["hesel", "interchange_dynamics"]:
            interchange["lnn"] = -inv_b * brackets(phi, lnn) - curvature(phi) + curvature(te) + curvature(lnn) * te
            interchange["lnpe"] = -inv_b * brackets(phi, lnpe) - 5.0 / 3.0 * curvature(phi) + 5.0 / 3.0 * curvature(te) + 5.0 / 3.0 * curvature(lnpe) * te
            interchange["lnpi"] = -inv_b * brackets(phi, lnpi) - 5.0 / 3.0 * curvature(phi) - 5.0 / 3.0 * curvature(ti) - 5.0 / 3.0 * curvature(lnpi) * ti + 2.0 / 3.0 * curvature(pe + pi)
            interchange["vort"] = -brackets(phi, vort) + curvature(pe + pi)
            if not settings["hesel", "test_vort_cross_term"]:
                interchange["vort"] = interchange["vort"] - brackets(dphi_dx, dpi_dx) - brackets(dphi_dz, dpi_dz)

        ti_rcpte = torch.full_like(tau, params.ti0 / params.te0) if settings["hesel", "ti_over_te"] == 1 else avg_tau if settings["hesel", "ti_over_te"] == 2 else tau
        if settings["hesel", "diffusion_coeff"] == 1:
            de, di = torch.full_like(n, params.norm_de), torch.full_like(n, params.norm_di)
        elif settings["hesel", "diffusion_coeff"] == 2:
            de, di = params.norm_de * avg_n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2), params.norm_di * avg_n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 3:
            de, di = params.norm_de * n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2), params.norm_di * n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 4:
            de, di = params.norm_de * avg_n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2), params.norm_di * avg_n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 5:
            de, di = params.norm_de * n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2), params.norm_di * n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif settings["hesel", "diffusion_coeff"] == 6:
            de, di = torch.full_like(n, params.norm_de) / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2), torch.full_like(n, params.norm_di) / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        else:
            raise ValueError("Unsupported diffusion_coeff option from BOUT-HESEL.")

        de, di = de * params.z_eff, di * params.z_eff
        dn = de * (1.0 + ti_rcpte)
        if settings["hesel", "collisional_model"] == 0:
            collisional = {name: torch.zeros_like(term) for name, term in interchange.items()}
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif settings["hesel", "collisional_model"] == 1:
            collisional = {
                "lnn": params.norm_de * (1.0 + params.ti0 / params.te0) * laplacian_perp(lnn, x, z),
                "lnpe": 2.0 / 3.0 * params.norm_de * (1.0 + params.ti0 / params.te0) * laplacian_perp(lnpe, x, z),
                "lnpi": 2.0 / 3.0 * 2.0 * params.norm_di * laplacian_perp(lnti, x, z),
                "vort": params.norm_eta * laplacian_perp(vort, x, z),
            }
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif settings["hesel", "collisional_model"] == 2:
            deln_rcpn = laplacian_perp(lnn, x, z) + dlnn_dx * dlnn_dx + dlnn_dz * dlnn_dz
            delte_rcpte = laplacian_perp(lnte, x, z) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = laplacian_perp(lnti, x, z) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            gradn_gradte_rcppe = dlnn_dx * dlnte_dx + dlnn_dz * dlnte_dz
            gradn_gradti_rcppi = dlnn_dx * dlnti_dx + dlnn_dz * dlnti_dz
            collisional = {
                "lnn": dn * deln_rcpn,
                "lnpe": 2.0 / 3.0 * dn * (deln_rcpn + gradn_gradte_rcppe) + 2.0 / 3.0 * 29.0 / 12.0 * de * (delte_rcpte + gradn_gradte_rcppe),
                "lnpi": 2.0 / 3.0 * 5.0 / 2.0 * dn * (deln_rcpn + gradn_gradti_rcppi) + 2.0 / 3.0 * 2.0 * di * (delti_rcpti + gradn_gradti_rcppi),
                "vort": params.norm_eta * laplacian_perp(vort, x, z),
            }
            u_r_x, u_r_z = -dn * dlnn_dx, -dn * dlnn_dz
        elif settings["hesel", "collisional_model"] == 3:
            delte_rcpte = laplacian_perp(lnte, x, z) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = laplacian_perp(lnti, x, z) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            u_r_x = -de * ((1.0 + ti_rcpte) * dlnn_dx + dlnti_dx * ti_rcpte - 0.5 * dlnte_dx)
            u_r_z = -de * ((1.0 + ti_rcpte) * dlnn_dz + dlnti_dz * ti_rcpte - 0.5 * dlnte_dz)
            div_gamma_r_rcpn = dlnn_dx * u_r_x + dlnn_dz * u_r_z + grad_x(u_r_x, x) + grad_z(u_r_z, z)
            collisional = {
                "lnn": -div_gamma_r_rcpn,
                "lnpe": -2.0 / 3.0 * (div_gamma_r_rcpn + dlnte_dx * u_r_x + dlnte_dz * u_r_z) + 2.0 / 3.0 * 29.0 / 12.0 * ((grad_x(de, x) + de * dlnn_dx) * dlnte_dx + (grad_z(de, z) + de * dlnn_dz) * dlnte_dz + de * delte_rcpte),
                "lnpi": -2.0 / 3.0 * 5.0 / 2.0 * (div_gamma_r_rcpn + dlnti_dx * u_r_x + dlnti_dz * u_r_z) + 2.0 / 3.0 * 2.0 * ((grad_x(di, x) + di * dlnn_dx) * dlnti_dx + (grad_z(di, z) + di * dlnn_dz) * dlnti_dz + di * delti_rcpti),
                "vort": params.norm_eta * laplacian_perp(vort, x, z),
            }
        else:
            raise ValueError("Unsupported collisional_model option from BOUT-HESEL.")

        heat_exchange = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel", "perpend_heat_exchange"]:
            q_resist_rcppi = u_r_x * (dlnn_dx + dlnti_dx) + u_r_z * (dlnn_dz + dlnti_dz)
            if settings["hesel", "qdelta_approx"] == 0:
                qdelta_rcppe = torch.zeros_like(n)
            elif settings["hesel", "qdelta_approx"] == 1:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte)
            elif settings["hesel", "qdelta_approx"] == 2:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte) * avg_n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif settings["hesel", "qdelta_approx"] == 3:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte) * n / te / torch.sqrt(torch.clamp(te, min=1e-12))
            elif settings["hesel", "qdelta_approx"] == 4:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte) * n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            else:
                raise ValueError("Unsupported qdelta_approx option from BOUT-HESEL.")
            heat_exchange["lnpe"] = -2.0 / 3.0 * q_resist_rcppi * ti_rcpte - 2.0 / 3.0 * qdelta_rcppe
            heat_exchange["lnpi"] = 2.0 / 3.0 * q_resist_rcppi + 2.0 / 3.0 * qdelta_rcppe / torch.clamp(ti_rcpte, min=1e-12)

        viscous = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel", "perpend_viscous_heating"]:
            field_sum = phi + pi
            qviscous = 3.0 / 10.0 * di * ((d2dx2(field_sum, x) - d2dz2(field_sum, z)) ** 2 + 4.0 * d2dxdz(field_sum, x, z) ** 2) / torch.clamp(ti, min=1e-12)
            viscous["lnpi"] = 2.0 / 3.0 * qviscous

        perpendicular = {name: collisional[name] + heat_exchange[name] + viscous[name] for name in interchange}
        if not settings["hesel","perpendicular_dynamics"]:
            perpendicular = {name: torch.zeros_like(term) for name, term in interchange.items()}

        parallel = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel","parallel_dynamics"]:
            if settings["hesel","parallel_advection_damping"] == 0:
                damp_advection = torch.zeros_like(n)
            elif settings["hesel","parallel_advection_damping"] == 1:
                damp_advection = torch.full_like(n, 1.0 / params.norm_taun)
            elif settings["hesel","parallel_advection_damping"] == 2:
                damp_advection = avg_cs_hot / params.norm_taun
            elif settings["hesel","parallel_advection_damping"] == 3:
                damp_advection = cs_hot / params.norm_taun
            else:
                raise ValueError("Unsupported parallel_advection_damping option from BOUT-HESEL.")
            parallel["lnn"] = parallel["lnn"] - sigma_open * damp_advection
            parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["vort"] = parallel["vort"] - sigma_open * damp_advection * vort

            if settings["hesel","parallel_sheath_damping"] == 0:
                damp_sheath = torch.zeros_like(n)
            elif settings["hesel","parallel_sheath_damping"] == 1:
                damp_sheath = 1.0 / params.norm_lc * (1.0 - torch.exp(params.bohm_potential - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif settings["hesel","parallel_sheath_damping"] == 2:
                damp_sheath = avg_cs_hot / params.norm_lc * (1.0 - torch.exp(params.bohm_potential - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif settings["hesel","parallel_sheath_damping"] == 3:
                damp_sheath = cs_hot / params.norm_lc * (1.0 - torch.exp(params.bohm_potential - phi / torch.clamp(te, min=1e-12)))
            else:
                raise ValueError("Unsupported parallel_sheath_damping option from BOUT-HESEL.")
            parallel["lnpi"] = parallel["lnpi"] + 2.0 / 3.0 * sigma_open * damp_sheath
            parallel["vort"] = parallel["vort"] + sigma_open * damp_sheath

            if settings["hesel","parallel_conduction"] == 0:
                damp_she = torch.zeros_like(n)
                damp_shi = torch.zeros_like(n)
            elif settings["hesel","parallel_conduction"] == 1:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / params.norm_taushe
                damp_shi = torch.zeros_like(n)
            elif settings["hesel","parallel_conduction"] == 2:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / params.norm_taushe
                damp_shi = ti * ti * torch.sqrt(torch.clamp(ti, min=1e-12)) / params.norm_taushi
            else:
                raise ValueError("Unsupported parallel_conduction option from BOUT-HESEL.")

            pert_n, pert_te, pert_phi = n - avg_n, te - avg_te, phi - avg_phi
            if settings["hesel","parallel_drift_wave"] == 0:
                driftwave = torch.zeros_like(n)
            elif settings["hesel","parallel_drift_wave"] == 1:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / params.norm_taudw
            elif settings["hesel","parallel_drift_wave"] == 2:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / params.norm_taudw * avg_te * torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif settings["hesel","parallel_drift_wave"] == 3:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / params.norm_taudw * te * torch.sqrt(torch.clamp(te, min=1e-12))
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
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she * (te - params.te_bck) / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi * (te - params.ti_bck) / torch.clamp(pi, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti / torch.clamp(pi, min=1e-12)
            else:
                raise ValueError("Unsupported reciprocal_approx option from BOUT-HESEL.")

        forcing = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel","force_profiles"]:
            if not settings["hesel","not_n_force"]:
                forcing["lnn"] = forcing["lnn"] + sigma_force * (init_n / torch.clamp(n, min=1e-12) - 1.0) / params.force_time
            if not settings["hesel","not_p_force"]:
                forcing_multiplier = torch.ones_like(pi)
                if settings["hesel","h_mode"]:
                    t_phys = t
                    forcing_multiplier = 1.0 + (settings["hesel","ramp_a"] - 1.0) / 2.0 * (torch.tanh((t_phys - settings["hesel","ramp_t0"]) / settings["hesel","ramp_trans"]) - torch.tanh((t_phys - settings["hesel","ramp_t0"] - settings["hesel","ramp_peak"]) / settings["hesel","ramp_trans"]))
                force_pe, force_pi = sigma_force * (init_pe - pe) / params.force_time, sigma_force * (init_pi * forcing_multiplier - pi) / params.force_time
                forcing["lnpe"] = forcing["lnpe"] + force_pe / torch.clamp(pe, min=1e-12)
                forcing["lnpi"] = forcing["lnpi"] + force_pi / torch.clamp(pi, min=1e-12)

        floor_terms = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if settings["hesel","floor_profiles"]:
            floor_terms["lnn"] = floor_terms["lnn"] + torch.where(n < params.floor_n, (params.floor_n / torch.clamp(n, min=1e-12) - 1.0) / params.floor_time, torch.zeros_like(n))
            floor_terms["lnpe"] = floor_terms["lnpe"] + torch.where(pe < params.floor_pe, (params.floor_pe / torch.clamp(pe, min=1e-12) - 1.0) / params.floor_time, torch.zeros_like(pe))
            floor_terms["lnpi"] = floor_terms["lnpi"] + torch.where(pi < params.floor_pi, (params.floor_pi / torch.clamp(pi, min=1e-12) - 1.0) / params.floor_time, torch.zeros_like(pi))


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
