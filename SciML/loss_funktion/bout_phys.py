from __future__ import annotations

import torch

from torch import Tensor
from typing import Any, Callable, Mapping

from .api.operators import *
from .bout_info import BOUTHESELInfo
from .api.read_bout import mse_dict


class BOUTHESELPhysics:
    def __init__(self, info: BOUTHESELInfo):
        self.info = info
        self.parameters = self.info.parameters
        self.boundary_conditions = self.info.boundary_conditions
        self.active_settings = self.info.active_settings

    def equation_terms(
        self,
        model: Callable[..., Mapping[str, Tensor]],
        x: Tensor,
        z: Tensor,
        t: Tensor,
    ) -> dict[str, Any]:
        state = dict(model(x, z, t))

        params, settings = self.parameters, self.active_settings
        lnn, lnpe, lnpi, phi = state["lnn"], state["lnpe"], state["lnpi"], state["phi"]
        n, pe, pi = torch.exp(lnn), torch.exp(lnpe), torch.exp(lnpi)
        lnte, lnti = lnpe - lnn, lnpi - lnn
        te, ti = torch.exp(lnte), torch.exp(lnti)
        tau = torch.exp(lnti - lnte)
        cs_hot = torch.sqrt(torch.clamp(ti + te, min=1e-12))
        avg_n, avg_te, avg_ti, avg_phi, avg_tau, avg_cs_hot = (value.mean(dim=2, keepdim=True) for value in (n, te, ti, phi, tau, cs_hot) )
        b_field = self.info.magnetic_field(x)
        inv_b = 1.0 / torch.clamp(b_field, min=1e-12)

        dphi_dx, dphi_dz = grad_x(phi, x, params.total_x), grad_z(phi, z, params.total_z)
        dpi_dx, dpi_dz = grad_x(pi, x, params.total_x), grad_z(pi, z, params.total_z)
        dlnn_dx, dlnn_dz = grad_x(lnn, x, params.total_x), grad_z(lnn, z, params.total_z)
        dlnte_dx, dlnte_dz = grad_x(lnte, x, params.total_x), grad_z(lnte, z, params.total_z)
        dlnti_dx, dlnti_dz = grad_x(lnti, x, params.total_x), grad_z(lnti, z, params.total_z)
        ddt_lnn, ddt_lnpe, ddt_lnpi = grad_t(lnn, t, params.total_t), grad_t(lnpe, t, params.total_t), grad_t(lnpi, t, params.total_t)
        vort_from_phi = laplacian_perp(phi + pi, x, z, params.total_x, params.total_z)
        vort = state["vort"] if "vort" in state else vort_from_phi
        ddt_vort = grad_t(vort, t, params.total_t)

        def brackets(f: Tensor, g: Tensor) -> Tensor:
            bracket = grad_x(f, x, params.total_x) * grad_z(g, z, params.total_z) - grad_z(f, z, params.total_z) * grad_x(g, x, params.total_x)
            return -bracket if bool(settings.get("right_handed_coord", False)) else bracket

        def curvature(f: Tensor) -> Tensor:
            curv = (2.0 if bool(settings.get("double_curvature_coeff", False)) else 1.0) * params.rhos / (params.rmajor + params.rminor) * grad_z(f, z, params.total_z)
            return -curv if bool(settings.get("right_handed_coord", False)) else curv

        sigma = self.info.initial_profiles(x=x, z=z)
        sigma_open, sigma_closed, sigma_force = sigma["sigma_open"], sigma["sigma_closed"], sigma["sigma_force"]
        init_n, init_pe, init_pi = sigma["init_n"], sigma["init_pe"], sigma["init_pi"]
        interchange = {name: torch.zeros_like(vort if name == "vort" else lnn) for name in ("lnn", "lnpe", "lnpi", "vort")}
        if bool(settings.get("interchange_dynamics", True)):
            interchange["lnn"] = -inv_b * brackets(phi, lnn) - curvature(phi) + curvature(te) + curvature(lnn) * te
            interchange["lnpe"] = -inv_b * brackets(phi, lnpe) - 5.0 / 3.0 * curvature(phi) + 5.0 / 3.0 * curvature(te) + 5.0 / 3.0 * curvature(lnpe) * te
            interchange["lnpi"] = -inv_b * brackets(phi, lnpi) - 5.0 / 3.0 * curvature(phi) - 5.0 / 3.0 * curvature(ti) - 5.0 / 3.0 * curvature(lnpi) * ti + 2.0 / 3.0 * curvature(pe + pi)
            interchange["vort"] = -brackets(phi, vort) + curvature(pe + pi)
            if not bool(settings.get("test_vort_cross_term", False)):
                interchange["vort"] = interchange["vort"] - brackets(dphi_dx, dpi_dx) - brackets(dphi_dz, dpi_dz)

        ti_rcpte = torch.full_like(tau, params.ti0 / params.te0) if int(settings.get("ti_over_te", 3)) == 1 else avg_tau if int(settings.get("ti_over_te", 3)) == 2 else tau
        diffusion_coeff = int(settings.get("diffusion_coeff", 1))
        if diffusion_coeff == 1:
            de, di = torch.full_like(n, params.norm_de), torch.full_like(n, params.norm_di)
        elif diffusion_coeff == 2:
            de, di = params.norm_de * avg_n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2), params.norm_di * avg_n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif diffusion_coeff == 3:
            de, di = params.norm_de * n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2), params.norm_di * n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif diffusion_coeff == 4:
            de, di = params.norm_de * avg_n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2), params.norm_di * avg_n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif diffusion_coeff == 5:
            de, di = params.norm_de * n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2), params.norm_di * n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif diffusion_coeff == 6:
            de, di = torch.full_like(n, params.norm_de) / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2), torch.full_like(n, params.norm_di) / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        else:
            raise ValueError("Unsupported diffusion_coeff option from BOUT-HESEL.")

        de, di = de * params.z_eff, di * params.z_eff
        dn = de * (1.0 + ti_rcpte)
        collisional_model = int(settings.get("collisional_model", 2))
        if collisional_model == 0:
            collisional = {name: torch.zeros_like(term) for name, term in interchange.items()}
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif collisional_model == 1:
            collisional = {
                "lnn": params.norm_de * (1.0 + params.ti0 / params.te0) * laplacian_perp(lnn, x, z, params.total_x, params.total_z),
                "lnpe": 2.0 / 3.0 * params.norm_de * (1.0 + params.ti0 / params.te0) * laplacian_perp(lnpe, x, z, params.total_x, params.total_z),
                "lnpi": 2.0 / 3.0 * 2.0 * params.norm_di * laplacian_perp(lnti, x, z, params.total_x, params.total_z),
                "vort": params.norm_eta * laplacian_perp(vort, x, z, params.total_x, params.total_z),
            }
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif collisional_model == 2:
            deln_rcpn = laplacian_perp(lnn, x, z, params.total_x, params.total_z) + dlnn_dx * dlnn_dx + dlnn_dz * dlnn_dz
            delte_rcpte = laplacian_perp(lnte, x, z, params.total_x, params.total_z) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = laplacian_perp(lnti, x, z, params.total_x, params.total_z) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            gradn_gradte_rcppe = dlnn_dx * dlnte_dx + dlnn_dz * dlnte_dz
            gradn_gradti_rcppi = dlnn_dx * dlnti_dx + dlnn_dz * dlnti_dz
            collisional = {
                "lnn": dn * deln_rcpn,
                "lnpe": 2.0 / 3.0 * dn * (deln_rcpn + gradn_gradte_rcppe) + 2.0 / 3.0 * 29.0 / 12.0 * de * (delte_rcpte + gradn_gradte_rcppe),
                "lnpi": 2.0 / 3.0 * 5.0 / 2.0 * dn * (deln_rcpn + gradn_gradti_rcppi) + 2.0 / 3.0 * 2.0 * di * (delti_rcpti + gradn_gradti_rcppi),
                "vort": params.norm_eta * laplacian_perp(vort, x, z, params.total_x, params.total_z),
            }
            u_r_x, u_r_z = -dn * dlnn_dx, -dn * dlnn_dz
        elif collisional_model == 3:
            delte_rcpte = laplacian_perp(lnte, x, z, params.total_x, params.total_z) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = laplacian_perp(lnti, x, z, params.total_x, params.total_z) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            u_r_x = -de * ((1.0 + ti_rcpte) * dlnn_dx + dlnti_dx * ti_rcpte - 0.5 * dlnte_dx)
            u_r_z = -de * ((1.0 + ti_rcpte) * dlnn_dz + dlnti_dz * ti_rcpte - 0.5 * dlnte_dz)
            div_gamma_r_rcpn = dlnn_dx * u_r_x + dlnn_dz * u_r_z + grad_x(u_r_x, x, params.total_x) + grad_z(u_r_z, z, params.total_z)
            collisional = {
                "lnn": -div_gamma_r_rcpn,
                "lnpe": -2.0 / 3.0 * (div_gamma_r_rcpn + dlnte_dx * u_r_x + dlnte_dz * u_r_z) + 2.0 / 3.0 * 29.0 / 12.0 * ((grad_x(de, x, params.total_x) + de * dlnn_dx) * dlnte_dx + (grad_z(de, z, params.total_z) + de * dlnn_dz) * dlnte_dz + de * delte_rcpte),
                "lnpi": -2.0 / 3.0 * 5.0 / 2.0 * (div_gamma_r_rcpn + dlnti_dx * u_r_x + dlnti_dz * u_r_z) + 2.0 / 3.0 * 2.0 * ((grad_x(di, x, params.total_x) + di * dlnn_dx) * dlnti_dx + (grad_z(di, z, params.total_z) + di * dlnn_dz) * dlnti_dz + di * delti_rcpti),
                "vort": params.norm_eta * laplacian_perp(vort, x, z, params.total_x, params.total_z),
            }
        else:
            raise ValueError("Unsupported collisional_model option from BOUT-HESEL.")

        heat_exchange = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if bool(settings.get("perpend_heat_exchange", True)):
            q_resist_rcppi = u_r_x * (dlnn_dx + dlnti_dx) + u_r_z * (dlnn_dz + dlnti_dz)
            qdelta_approx = int(settings.get("qdelta_approx", 3))
            if qdelta_approx == 0:
                qdelta_rcppe = torch.zeros_like(n)
            elif qdelta_approx == 1:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte)
            elif qdelta_approx == 2:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte) * avg_n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif qdelta_approx == 3:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte) * n / te / torch.sqrt(torch.clamp(te, min=1e-12))
            elif qdelta_approx == 4:
                qdelta_rcppe = 3.0 * params.me / params.mi * (params.nuei / params.oci) * (1.0 - ti_rcpte) * n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            else:
                raise ValueError("Unsupported qdelta_approx option from BOUT-HESEL.")
            heat_exchange["lnpe"] = -2.0 / 3.0 * q_resist_rcppi * ti_rcpte - 2.0 / 3.0 * qdelta_rcppe
            heat_exchange["lnpi"] = 2.0 / 3.0 * q_resist_rcppi + 2.0 / 3.0 * qdelta_rcppe / torch.clamp(ti_rcpte, min=1e-12)

        viscous = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if bool(settings.get("perpend_viscous_heating", True)):
            field_sum = phi + pi
            qviscous = 3.0 / 10.0 * di * ((d2dx2(field_sum, x, params.total_x) - d2dz2(field_sum, z, params.total_z)) ** 2 + 4.0 * d2dxdz(field_sum, x, z, params.total_x, params.total_z) ** 2) / torch.clamp(ti, min=1e-12)
            viscous["lnpi"] = 2.0 / 3.0 * qviscous

        perpendicular = {name: collisional[name] + heat_exchange[name] + viscous[name] for name in interchange}
        if not bool(settings.get("perpendicular_dynamics", True)):
            perpendicular = {name: torch.zeros_like(term) for name, term in interchange.items()}

        parallel = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if bool(settings.get("parallel_dynamics", True)):
            advection_mode = int(settings.get("parallel_advection_damping", 3))
            if advection_mode == 0:
                damp_advection = torch.zeros_like(n)
            elif advection_mode == 1:
                damp_advection = torch.full_like(n, 1.0 / params.norm_taun)
            elif advection_mode == 2:
                damp_advection = avg_cs_hot / params.norm_taun
            elif advection_mode == 3:
                damp_advection = cs_hot / params.norm_taun
            else:
                raise ValueError("Unsupported parallel_advection_damping option from BOUT-HESEL.")
            parallel["lnn"] = parallel["lnn"] - sigma_open * damp_advection
            parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["vort"] = parallel["vort"] - sigma_open * damp_advection * vort

            sheath_mode = int(settings.get("parallel_sheath_damping", 3))
            if sheath_mode == 0:
                damp_sheath = torch.zeros_like(n)
            elif sheath_mode == 1:
                damp_sheath = 1.0 / params.norm_lc * (1.0 - torch.exp(params.bohm_potential - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif sheath_mode == 2:
                damp_sheath = avg_cs_hot / params.norm_lc * (1.0 - torch.exp(params.bohm_potential - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif sheath_mode == 3:
                damp_sheath = cs_hot / params.norm_lc * (1.0 - torch.exp(params.bohm_potential - phi / torch.clamp(te, min=1e-12)))
            else:
                raise ValueError("Unsupported parallel_sheath_damping option from BOUT-HESEL.")
            parallel["lnpi"] = parallel["lnpi"] + 2.0 / 3.0 * sigma_open * damp_sheath
            parallel["vort"] = parallel["vort"] + sigma_open * damp_sheath

            conduction_mode = int(settings.get("parallel_conduction", 1))
            if conduction_mode == 0:
                damp_she = torch.zeros_like(n)
                damp_shi = torch.zeros_like(n)
            elif conduction_mode == 1:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / params.norm_taushe
                damp_shi = torch.zeros_like(n)
            elif conduction_mode == 2:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / params.norm_taushe
                damp_shi = ti * ti * torch.sqrt(torch.clamp(ti, min=1e-12)) / params.norm_taushi
            else:
                raise ValueError("Unsupported parallel_conduction option from BOUT-HESEL.")

            pert_n, pert_te, pert_phi = n - avg_n, te - avg_te, phi - avg_phi
            driftwave_mode = int(settings.get("parallel_drift_wave", 1))
            if driftwave_mode == 0:
                driftwave = torch.zeros_like(n)
            elif driftwave_mode == 1:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / params.norm_taudw
            elif driftwave_mode == 2:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / params.norm_taudw * avg_te * torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif driftwave_mode == 3:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / params.norm_taudw * te * torch.sqrt(torch.clamp(te, min=1e-12))
            else:
                raise ValueError("Unsupported parallel_drift_wave option from BOUT-HESEL.")
            parallel["vort"] = parallel["vort"] - sigma_closed * driftwave

            reciprocal_approx = int(settings.get("reciprocal_approx", 3))
            if reciprocal_approx == 1:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti
            elif reciprocal_approx == 2:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi / torch.clamp(avg_n, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave
            elif reciprocal_approx == 3:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she * (te - params.te_bck) / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi * (te - params.ti_bck) / torch.clamp(pi, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti / torch.clamp(pi, min=1e-12)
            else:
                raise ValueError("Unsupported reciprocal_approx option from BOUT-HESEL.")

        forcing = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if bool(settings.get("force_profiles", True)):
            if not bool(settings.get("not_n_force", False)):
                forcing["lnn"] = forcing["lnn"] + sigma_force * (init_n / torch.clamp(n, min=1e-12) - 1.0) / params.force_time
            if not bool(settings.get("not_p_force", False)):
                forcing_multiplier = torch.ones_like(pi)
                if bool(settings.get("h_mode", False)):
                    ramp_a, ramp_t0, ramp_trans, ramp_peak = (float(settings.get(key, default)) for key, default in (("ramp_a", 2.0), ("ramp_t0", 0.0), ("ramp_trans", 5000.0), ("ramp_peak", 50000.0)))
                    t_phys = t * params.total_t
                    forcing_multiplier = 1.0 + (ramp_a - 1.0) / 2.0 * (torch.tanh((t_phys - ramp_t0) / ramp_trans) - torch.tanh((t_phys - ramp_t0 - ramp_peak) / ramp_trans))
                force_pe, force_pi = sigma_force * (init_pe - pe) / params.force_time, sigma_force * (init_pi * forcing_multiplier - pi) / params.force_time
                forcing["lnpe"] = forcing["lnpe"] + force_pe / torch.clamp(pe, min=1e-12)
                forcing["lnpi"] = forcing["lnpi"] + force_pi / torch.clamp(pi, min=1e-12)

        floor_terms = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if bool(settings.get("floor_profiles", False)):
            floor_terms["lnn"] = floor_terms["lnn"] + torch.where(n < params.floor_n, (params.floor_n / torch.clamp(n, min=1e-12) - 1.0) / params.floor_time, torch.zeros_like(n))
            floor_terms["lnpe"] = floor_terms["lnpe"] + torch.where(pe < params.floor_pe, (params.floor_pe / torch.clamp(pe, min=1e-12) - 1.0) / params.floor_time, torch.zeros_like(pe))
            floor_terms["lnpi"] = floor_terms["lnpi"] + torch.where(pi < params.floor_pi, (params.floor_pi / torch.clamp(pi, min=1e-12) - 1.0) / params.floor_time, torch.zeros_like(pi))

        rhs_terms = interchange
        lambda_terms = {name: perpendicular[name] + parallel[name] + forcing[name] + floor_terms[name] for name in interchange}
        result = {
            "state": {
                **state, "n": n, "pe": pe, "pi": pi, "te": te, "ti": ti, "tau": tau, "vort_from_phi": vort_from_phi, "vort": vort, "B": b_field,
                "sigma_open": sigma_open, "sigma_closed": sigma_closed, "sigma_force": sigma_force, "init_n": init_n, "init_pe": init_pe, "init_pi": init_pi,
            },
            "rhs_terms": rhs_terms,
            "lambda_terms": lambda_terms,
            "components": {"interchange": interchange, "perpendicular": perpendicular, "parallel": parallel, "forcing": forcing, "floor": floor_terms},
            "residuals": {
                "eq_lnn": ddt_lnn - rhs_terms["lnn"] - lambda_terms["lnn"],
                "eq_lnpe": ddt_lnpe - rhs_terms["lnpe"] - lambda_terms["lnpe"],
                "eq_lnpi": ddt_lnpi - rhs_terms["lnpi"] - lambda_terms["lnpi"],
                "eq_vort": ddt_vort - rhs_terms["vort"] - lambda_terms["vort"],
            },
        }
        if "vort" in state:
            result["poisson_residual"] = state["vort"] - vort_from_phi
        return result


    def eq_loss(
        self,
        model: Callable[..., Mapping[str, Tensor]],
        x: Tensor,
        z: Tensor,
        t: Tensor,
        weights: Mapping[str, float] | None = None,
    ) -> Tensor:
        return mse_dict(self.equation_terms(model, x=x, z=z, t=t)["residuals"], weights=weights)


    def bc_loss(
        self,
        model: Callable[..., Mapping[str, Tensor]],
        x: Tensor,
        z: Tensor,
        t: Tensor,
        weights: Mapping[str, float] | None = None,
    ) -> Tensor:
        
        state = dict(model(x, z, t))

        residuals: dict[str, Tensor] = {}
        for field in ("lnn", "lnpe", "lnpi", "phi", "vort"):
            if field not in state or field not in self.boundary_conditions:
                continue
            for side, idx in (("inner", 0), ("outer", -1)):
                bc = self.boundary_conditions[field][side]
                key = f"bc_{field}_{side}"
                if bc.kind.startswith("dirichlet"):
                    residuals[key] = state[field][:, idx, :, :] - (0.0 if bc.value is None else bc.value)
                elif bc.kind.startswith("neumann"):
                    residuals[key] = grad_x(state[field], x, self.parameters.total_x)[:, idx, :, :]
        return mse_dict(residuals, weights=weights)


    def ic_loss(
        self,
        model: Callable[..., Mapping[str, Tensor]],
        x: Tensor,
        z: Tensor,
        t: Tensor,
        weights: Mapping[str, float] | None = None,
    ) -> Tensor:
        t0 = torch.zeros_like(t, requires_grad=True)
        state = dict(model(x, z, t0))
        targets = self.info.initial_state_targets(x.squeeze(0), z.squeeze(0))
        residuals = {
            "ic_lnn": state["lnn"].squeeze(0) - targets["lnn"],
            "ic_lnpe": state["lnpe"].squeeze(0) - targets["lnpe"],
            "ic_lnpi": state["lnpi"].squeeze(0) - targets["lnpi"],
            "ic_phi": state["phi"].squeeze(0) - targets["phi"],
        }
        if "vort" in state:
            residuals["ic_vort"] = state["vort"].squeeze(0) - targets["vort"]
        return mse_dict(residuals, weights=weights)

