from __future__ import annotations

# Fiks path for imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import re
import math
import torch

from torch import Tensor

from hesel_scraper.bout_dump import BOUTHESELInfo
from hesel_scraper.api.operators import grad_components, grad_x, grad_z, laplacian_perp, d2dx2, d2dz2, d2dxdz

from SciML.PINN.util.model import PINN


class BOUTHESELPhys:
    def __init__(self, info: BOUTHESELInfo):
        self.info = info

        self.hes_settings = {key: value for (section, key), value in self.info.settings.items() if section == "hesel"}

        self.ic = self._ic()
        self.bc = self._bc()
        self.profiles = self._profiles_on_device()
        self.x_boundary_state = self._x_boundary_state()

        self.ic_res = self._ic_residuals
        self.bc_res = self._bc_residuals
        self.eq_res = self._eq_residuals


    def _get_vort_field(self, state: dict[str, Tensor], coord: Tensor):
        return laplacian_perp(state["phi"] + torch.exp(state["lnpi"]), coord)


    def _ic(self) -> dict[str, Tensor]:
        """
        Returns initial conditions for predicted fields as dict{field: Tensor}, and muves it to device.
        numerical domain: [0, parameters.num_(x,z)-1]
        n, pe, pi are log transformed to match prediction fields
        """
        return {
            "init_lnn":  torch.log(self.info.data.init_n).to(torch.float32).to(self.info.device), #(num_x, 1)
            "init_lnpe":  torch.log(self.info.data.init_pe).to(torch.float32).to(self.info.device), # (num_x, 1)
            "init_lnpi":  torch.log(self.info.data.init_pi).to(torch.float32).to(self.info.device), # (num_x, 1)
            "init_vort":  self.info.data.vort[0, :, 0].to(torch.float32).to(self.info.device), # (num_x, 1)
            "init_phi":  self.info.data.phi[0, :, 0].to(torch.float32).to(self.info.device) # (num_x, 1)
        }

    def _ic_residuals(self, model: PINN, cord_fys: Tensor, cord_num: Tensor, input: Tensor, standardization):
        # convert to initial cord: (x,z,t) -> (x, z, t=0)
        cord_fys_ic = cord_fys.clone()
        cord_fys_ic[:, 2] = 0

        ic_pred = model.forward(standardization, cord_fys_ic, torch.zeros_like(input)) # (model, cord_fys/num, input, standardization) allerede på device
        ic_vort = self._get_vort_field(ic_pred, cord_fys)# udregnet op device
        return {
            "ic_lnn" : ic_pred["lnn"]  - self.ic["init_lnn"][cord_num[:, 0].int()].view_as(ic_pred["lnn"]),
            "ic_lnpe": ic_pred["lnpe"] - self.ic["init_lnpe"][cord_num[:, 0].int()].view_as(ic_pred["lnpe"]),
            "ic_lnpi": ic_pred["lnpi"] - self.ic["init_lnpi"][cord_num[:, 0].int()].view_as(ic_pred["lnpi"]),
            "ic_vort": ic_vort - self.ic["init_vort"][cord_num[:, 0].int()].view_as(ic_vort),
        }


    def _bc(self):
        """
        Returns boundary conditions: dict{field: dict{side: (bc_type, value)}}.
        Values are ln transformed and moved to device.
        """
        boundary_conditions = {}
        allowed_fields = {"lnn", "lnpe", "lnpi", "vort", "phi"}
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

            # Konverter til tensor og flyt til device
            bc_tensor = torch.tensor(float(parsed_value), dtype=self.info.dtype, device=self.info.device)
            boundary_conditions.setdefault(field, {})[side] = (bc_type, bc_tensor)

        # Undskyld jeg har snydt lidt her men går ud fra at det er ok ;)
        boundary_conditions.setdefault("phi", {}).update({
            "xin": ("dirichlet_o2", self.ic["init_phi"][0]),
            "xout": ("neumann_o2", torch.tensor(0.0, dtype=self.info.dtype, device=self.info.device)),
        })
        return boundary_conditions

    def _x_boundary_state(self):
        return {
            "xin": torch.stack(
                (
                    self.ic["init_lnn"][0],
                    self.ic["init_lnpe"][0],
                    self.ic["init_lnpi"][0],
                    self.ic["init_vort"][0],
                )
            ).view(1, 1, 4, 1, 1),
            "xout": torch.stack(
                (
                    self.ic["init_lnn"][-1],
                    self.ic["init_lnpe"][-1],
                    self.ic["init_lnpi"][-1],
                    self.ic["init_vort"][-1],
                )
            ).view(1, 1, 4, 1, 1),
        }

    def _bc_residuals(self, model: PINN, cord_fys: Tensor, input: Tensor, standardization):        
        cord_fys_zmax = cord_fys.clone()
        cord_fys_zmax[:, 1] = self.info.parameters.Lz

        cord_fys_zmin = cord_fys.clone()
        cord_fys_zmin[:, 1] = 0

        cord_fys_xmin = cord_fys.clone()
        cord_fys_xmin[:, 0] = 0

        cord_fys_xmax = cord_fys.clone()
        cord_fys_xmax[:, 0] = self.info.parameters.Lx
        x_min_inp = self.x_boundary_state["xin"].expand_as(input)
        x_max_inp = self.x_boundary_state["xout"].expand_as(input)

        z_min_pred = model.forward(standardization, cord_fys_zmin, input)
        z_min_pred["vort"] = self._get_vort_field(z_min_pred, cord_fys_zmin)

        z_max_pred = model.forward(standardization, cord_fys_zmax, input)
        z_max_pred["vort"] = self._get_vort_field(z_max_pred, cord_fys_zmax)

        x_min_pred = model.forward(standardization, cord_fys_xmin, x_min_inp)
        x_min_pred["vort"] = self._get_vort_field(x_min_pred, cord_fys_xmin)

        x_max_pred = model.forward(standardization, cord_fys_xmax, x_max_inp)
        x_max_pred["vort"] = self._get_vort_field(x_max_pred, cord_fys_xmax)

        x_side_preds = {"xin": x_min_pred, "xout": x_max_pred}
        z_side_preds = {"zlower": z_min_pred, "zupper": z_max_pred}

        results = {}
        for field, specification in self.bc.items():
            for side, (bc_type, value) in specification.items():
                side_name = side.lower()
                bc_name = f"{field}_bc_{side}_{bc_type}"

                if "xin" in side_name:
                    pred = x_side_preds["xin"][field]
                    target = torch.full_like(pred, float(value))
                    if "dirichlet" in bc_type.lower():
                        results[bc_name] = pred - target
                    elif "neumann" in bc_type.lower():
                        results[bc_name] = grad_x(pred, cord_fys_xmin).view_as(pred) - target
                elif "xout" in side_name:
                    pred = x_side_preds["xout"][field]
                    target = torch.full_like(pred, float(value))
                    if "dirichlet" in bc_type.lower():
                        results[bc_name] = pred - target
                    elif "neumann" in bc_type.lower():
                        results[bc_name] = grad_x(pred, cord_fys_xmax).view_as(pred) - target

            results[f"{field}_bc_periodic"] = z_side_preds["zlower"][field] - z_side_preds["zupper"][field]

        return results
    
    def _profiles_on_device(self):
        """
        Flytter felt linjer til device: (4, num_x)
        """
        return {
            "b_field": self.info.data.B.to(device=self.info.device, dtype=self.info.dtype),
            "sigma_open": self.info.data.sigma_open.to(device=self.info.device, dtype=self.info.dtype),
            "sigma_closed": self.info.data.sigma_closed.to(device=self.info.device, dtype=self.info.dtype),
            "sigma_force": self.info.data.sigma_force.to(device=self.info.device, dtype=self.info.dtype)
        }


    def _eq_residuals(
        self,
        avg_z: Tensor,
        state: dict[str, Tensor],
        cord_fys: Tensor,
        cord_num: Tensor,
    ):
        # Params ikke på device men int/floats flyttes automatisk til device når de bruges i beregninger
        params = self.info.parameters
        x_idx = cord_num[:, 0].int()

        lnn, lnpe, lnpi, phi = state["lnn"], state["lnpe"], state["lnpi"], state["phi"]
        n, pe, pi = torch.exp(lnn), torch.exp(lnpe), torch.exp(lnpi)
        lnte, lnti = lnpe - lnn, lnpi - lnn
        te, ti = torch.exp(lnte), torch.exp(lnti)
        tau = torch.exp(lnti - lnte)
        cs_hot = torch.sqrt(torch.clamp(ti + te, min=1e-12))
        

        init_n = torch.exp(self.ic["init_lnn"][x_idx]).view_as(n)
        init_pe = torch.exp(self.ic["init_lnpe"][x_idx]).view_as(pe)
        init_pi = torch.exp(self.ic["init_lnpi"][x_idx]).view_as(pi)

        b_field = self.profiles["b_field"][x_idx].view_as(n)
        sigma_open = self.profiles["sigma_open"][x_idx].view_as(n)
        sigma_closed = self.profiles["sigma_closed"][x_idx].view_as(n)
        sigma_force = self.profiles["sigma_force"][x_idx].view_as(n)

        inv_b = 1.0 / torch.clamp(b_field, min=1e-12)


        avg_n = avg_z[:, 0:1]
        avg_te = avg_z[:, 1:2]
        avg_ti = avg_z[:, 2:3]
        avg_phi = avg_z[:, 3:4]

        avg_tau = avg_ti / torch.clamp(avg_te, min=1e-12)
        avg_cs_hot = torch.sqrt(torch.clamp(avg_ti + avg_te, min=1e-12))


        curvature_scale = 1.0 / max(float(params.norm_lb), 1e-12)
        ti0_over_te0 = float(params.ti0) / max(float(params.te0), 1e-12)
        me_over_mi = float(params.me) / max(float(params.mi), 1e-12)
        nuei_over_oci = float(params.nuei) / max(float(params.oci), 1e-12)
        force_time = max(float(params.force_time), 1e-12)
        floor_time = max(float(params.floor_time), 1e-12)
        floor_n = float(params.floor_n)
        floor_pe = float(params.floor_pe)
        floor_pi = float(params.floor_pi)
        te_bck = float(getattr(params, "te_bck", 0.0))
        ti_bck = float(getattr(params, "ti_bck", 0.0))
        t_phys = cord_fys[:, 2:3]
        derivative_cache: dict[tuple[str, int], Tensor] = {}
        component_cache: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

        def cache_key(field: Tensor) -> int:
            return id(field)

        def first_derivatives(field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            key = cache_key(field)
            if key not in component_cache:
                dx, dz, dt = grad_components(field, cord_fys)
                component_cache[key] = (
                    dx.reshape(field.shape).to(field.dtype),
                    dz.reshape(field.shape).to(field.dtype),
                    dt.reshape(field.shape).to(field.dtype),
                )
            return component_cache[key]

        dphi_dx, dphi_dz, _ = first_derivatives(phi)
        dpi_dx, dpi_dz, _ = first_derivatives(pi)
        dlnn_dx, dlnn_dz, ddt_lnn = first_derivatives(lnn)
        dlnte_dx, dlnte_dz, _ = first_derivatives(lnte)
        dlnti_dx, dlnti_dz, _ = first_derivatives(lnti)
        _, _, ddt_lnpe = first_derivatives(lnpe)
        _, _, ddt_lnpi = first_derivatives(lnpi)
        vort = self._get_vort_field(state, cord_fys)
        _, _, ddt_vort = first_derivatives(vort)


        def ddx(field: Tensor) -> Tensor:
            key = ("dx", cache_key(field))
            if key not in derivative_cache:
                derivative_cache[key] = first_derivatives(field)[0]
            return derivative_cache[key]

        def ddz(field: Tensor) -> Tensor:
            key = ("dz", cache_key(field))
            if key not in derivative_cache:
                derivative_cache[key] = first_derivatives(field)[1]
            return derivative_cache[key]

        def lap(field: Tensor) -> Tensor:
            key = ("lap", cache_key(field))
            if key not in derivative_cache:
                derivative_cache[key] = ddxx(field) + ddzz(field)
            return derivative_cache[key]

        def ddxx(field: Tensor) -> Tensor:
            key = ("ddxx", cache_key(field))
            if key not in derivative_cache:
                derivative_cache[key] = grad_x(ddx(field), cord_fys).reshape(field.shape).to(field.dtype)
            return derivative_cache[key]

        def ddzz(field: Tensor) -> Tensor:
            key = ("ddzz", cache_key(field))
            if key not in derivative_cache:
                derivative_cache[key] = grad_z(ddz(field), cord_fys).reshape(field.shape).to(field.dtype)
            return derivative_cache[key]

        def ddxz(field: Tensor) -> Tensor:
            key = ("ddxz", cache_key(field))
            if key not in derivative_cache:
                derivative_cache[key] = grad_x(ddz(field), cord_fys).reshape(field.shape).to(field.dtype)
            return derivative_cache[key]


        def brackets(f: Tensor, g: Tensor) -> Tensor:
            bracket = ddx(f) * ddz(g) - ddz(f) * ddx(g)
            return -bracket if self.hes_settings.get("right_handed_coord", False) else bracket

        def curvature(f: Tensor) -> Tensor:
            curv = (
                (2.0 if self.hes_settings.get("double_curvature_coeff", False) else 1.0)
                * curvature_scale
                * ddz(f)
            )
            return -curv if self.hes_settings.get("right_handed_coord", False) else curv

        interchange = {name: torch.zeros_like(vort if name == "vort" else lnn) for name in ("lnn", "lnpe", "lnpi", "vort")}
        if self.hes_settings.get("interchange_dynamics", False):
            interchange["lnn"] = -inv_b * brackets(phi, lnn) - curvature(phi) + curvature(te) + curvature(lnn) * te
            interchange["lnpe"] = -inv_b * brackets(phi, lnpe) - 5.0 / 3.0 * curvature(phi) + 5.0 / 3.0 * curvature(te) + 5.0 / 3.0 * curvature(lnpe) * te
            interchange["lnpi"] = -inv_b * brackets(phi, lnpi) - 5.0 / 3.0 * curvature(phi) - 5.0 / 3.0 * curvature(ti) - 5.0 / 3.0 * curvature(lnpi) * ti + 2.0 / 3.0 * curvature(pe + pi)
            interchange["vort"] = -brackets(phi, vort) + curvature(pe + pi)
            if not self.hes_settings.get("test_vort_cross_term", False):
                interchange["vort"] = interchange["vort"] - brackets(dphi_dx, dpi_dx) - brackets(dphi_dz, dpi_dz)

        ti_over_te_mode = self.hes_settings.get("ti_over_te", 3)
        ti_rcpte = (
            torch.ones_like(tau) * ti0_over_te0
            if ti_over_te_mode == 1
            else avg_tau if ti_over_te_mode == 2 else tau
        )
        if ti_over_te_mode == 2:
            ti_rcpte = avg_tau
        elif ti_over_te_mode == 3:
            ti_rcpte = tau

        if self.hes_settings.get("diffusion_coeff", 1) == 1:
            de = torch.ones_like(n) * float(params.norm_de)
            di = torch.ones_like(n) * float(params.norm_di)
        elif self.hes_settings.get("diffusion_coeff", 1) == 2:
            de = float(params.norm_de) * avg_n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = float(params.norm_di) * avg_n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif self.hes_settings.get("diffusion_coeff", 1) == 3:
            de = float(params.norm_de) * n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2)
            di = float(params.norm_di) * n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif self.hes_settings.get("diffusion_coeff", 1) == 4:
            de = float(params.norm_de) * avg_n / torch.sqrt(torch.clamp(te, min=1e-12)) / (b_field**2)
            di = float(params.norm_di) * avg_n / torch.sqrt(torch.clamp(ti, min=1e-12)) / (b_field**2)
        elif self.hes_settings.get("diffusion_coeff", 1) == 5:
            de = float(params.norm_de) * n / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = float(params.norm_di) * n / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        elif self.hes_settings.get("diffusion_coeff", 1) == 6:
            de = torch.ones_like(n) * float(params.norm_de) / torch.sqrt(torch.clamp(avg_te, min=1e-12)) / (b_field**2)
            di = torch.ones_like(n) * float(params.norm_di) / torch.sqrt(torch.clamp(avg_ti, min=1e-12)) / (b_field**2)
        else:
            raise ValueError("Unsupported diffusion_coeff option from BOUT-HESEL.")

        de, di = de * float(params.z_eff), di * float(params.z_eff)
        dn = de * (1.0 + ti_rcpte)
        if self.hes_settings.get("collisional_model", 0) == 0:
            collisional = {name: torch.zeros_like(term) for name, term in interchange.items()}
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif self.hes_settings.get("collisional_model", 0) == 1:
            collisional = {
                "lnn": float(params.norm_de) * (1.0 + ti0_over_te0) * lap(lnn),
                "lnpe": 2.0 / 3.0 * float(params.norm_de) * (1.0 + ti0_over_te0) * lap(lnpe),
                "lnpi": 2.0 / 3.0 * 2.0 * float(params.norm_di) * lap(lnti),
                "vort": float(params.norm_eta) * lap(vort),
            }
            u_r_x = torch.zeros_like(n)
            u_r_z = torch.zeros_like(n)
        elif self.hes_settings.get("collisional_model", 0) == 2:
            deln_rcpn = lap(lnn) + dlnn_dx * dlnn_dx + dlnn_dz * dlnn_dz
            delte_rcpte = lap(lnte) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = lap(lnti) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            gradn_gradte_rcppe = dlnn_dx * dlnte_dx + dlnn_dz * dlnte_dz
            gradn_gradti_rcppi = dlnn_dx * dlnti_dx + dlnn_dz * dlnti_dz
            collisional = {
                "lnn": dn * deln_rcpn,
                "lnpe": 2.0 / 3.0 * dn * (deln_rcpn + gradn_gradte_rcppe) + 2.0 / 3.0 * 29.0 / 12.0 * de * (delte_rcpte + gradn_gradte_rcppe),
                "lnpi": 2.0 / 3.0 * 5.0 / 2.0 * dn * (deln_rcpn + gradn_gradti_rcppi) + 2.0 / 3.0 * 2.0 * di * (delti_rcpti + gradn_gradti_rcppi),
                "vort": float(params.norm_eta) * lap(vort),
            }
            u_r_x, u_r_z = -dn * dlnn_dx, -dn * dlnn_dz
        elif self.hes_settings.get("collisional_model", 0) == 3:
            delte_rcpte = lap(lnte) + dlnte_dx * dlnte_dx + dlnte_dz * dlnte_dz
            delti_rcpti = lap(lnti) + dlnti_dx * dlnti_dx + dlnti_dz * dlnti_dz
            u_r_x = -de * ((1.0 + ti_rcpte) * dlnn_dx + dlnti_dx * ti_rcpte - 0.5 * dlnte_dx)
            u_r_z = -de * ((1.0 + ti_rcpte) * dlnn_dz + dlnti_dz * ti_rcpte - 0.5 * dlnte_dz)
            div_gamma_r_rcpn = dlnn_dx * u_r_x + dlnn_dz * u_r_z + ddx(u_r_x) + ddz(u_r_z)
            collisional = {
                "lnn": -div_gamma_r_rcpn,
                "lnpe": -2.0 / 3.0 * (div_gamma_r_rcpn + dlnte_dx * u_r_x + dlnte_dz * u_r_z) + 2.0 / 3.0 * 29.0 / 12.0 * ((ddx(de) + de * dlnn_dx) * dlnte_dx + (ddz(de) + de * dlnn_dz) * dlnte_dz + de * delte_rcpte),
                "lnpi": -2.0 / 3.0 * 5.0 / 2.0 * (div_gamma_r_rcpn + dlnti_dx * u_r_x + dlnti_dz * u_r_z) + 2.0 / 3.0 * 2.0 * ((ddx(di) + di * dlnn_dx) * dlnti_dx + (ddz(di) + di * dlnn_dz) * dlnti_dz + di * delti_rcpti),
                "vort": float(params.norm_eta) * lap(vort),
            }
        else:
            raise ValueError("Unsupported collisional_model option from BOUT-HESEL.")

        heat_exchange = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if self.hes_settings.get("perpend_heat_exchange", False):
            q_resist_rcppi = u_r_x * (dlnn_dx + dlnti_dx) + u_r_z * (dlnn_dz + dlnti_dz)
            if self.hes_settings.get("qdelta_approx", 0) == 0:
                qdelta_rcppe = torch.zeros_like(n)
            elif self.hes_settings.get("qdelta_approx", 0) == 1:
                qdelta_rcppe = 3.0 * me_over_mi * nuei_over_oci * (1.0 - ti_rcpte)
            elif self.hes_settings.get("qdelta_approx", 0) == 2:
                qdelta_rcppe = 3.0 * me_over_mi * nuei_over_oci * (1.0 - ti_rcpte) * avg_n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif self.hes_settings.get("qdelta_approx", 0) == 3:
                qdelta_rcppe = 3.0 * me_over_mi * nuei_over_oci * (1.0 - ti_rcpte) * n / te / torch.sqrt(torch.clamp(te, min=1e-12))
            elif self.hes_settings.get("qdelta_approx", 0) == 4:
                qdelta_rcppe = 3.0 * me_over_mi * nuei_over_oci * (1.0 - ti_rcpte) * n / avg_te / torch.sqrt(torch.clamp(avg_te, min=1e-12))
            else:
                raise ValueError("Unsupported qdelta_approx option from BOUT-HESEL.")
            heat_exchange["lnpe"] = -2.0 / 3.0 * q_resist_rcppi * ti_rcpte - 2.0 / 3.0 * qdelta_rcppe
            heat_exchange["lnpi"] = 2.0 / 3.0 * q_resist_rcppi + 2.0 / 3.0 * qdelta_rcppe / torch.clamp(ti_rcpte, min=1e-12)

        viscous = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if self.hes_settings.get("perpend_viscous_heating", False):
            field_sum = phi + pi
            qviscous = 3.0 / 10.0 * di * ((ddxx(field_sum) - ddzz(field_sum)) ** 2 + 4.0 * ddxz(field_sum) ** 2) / torch.clamp(ti, min=1e-12)
            viscous["lnpi"] = 2.0 / 3.0 * qviscous

        perpendicular = {name: collisional[name] + heat_exchange[name] + viscous[name] for name in interchange}
        if not self.hes_settings.get("perpendicular_dynamics", False):
            perpendicular = {name: torch.zeros_like(term) for name, term in interchange.items()}

        parallel = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if self.hes_settings.get("parallel_dynamics", self.hes_settings.get("parallel_transport", False)):
            if self.hes_settings.get("parallel_advection_damping", 0) == 0:
                damp_advection = torch.zeros_like(n)
            elif self.hes_settings.get("parallel_advection_damping", 0) == 1:
                damp_advection = torch.ones_like(n) / float(params.norm_taun)
            elif self.hes_settings.get("parallel_advection_damping", 0) == 2:
                damp_advection = avg_cs_hot / float(params.norm_taun)
            elif self.hes_settings.get("parallel_advection_damping", 0) == 3:
                damp_advection = cs_hot / float(params.norm_taun)
            else:
                raise ValueError("Unsupported parallel_advection_damping option from BOUT-HESEL.")
            parallel["lnn"] = parallel["lnn"] - sigma_open * damp_advection
            parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * 9.0 / 2.0 * sigma_open * damp_advection
            parallel["vort"] = parallel["vort"] - sigma_open * damp_advection * vort

            if self.hes_settings.get("parallel_sheath_damping", 0) == 0:
                damp_sheath = torch.zeros_like(n)
            elif self.hes_settings.get("parallel_sheath_damping", 0) == 1:
                damp_sheath = (1.0 / float(params.norm_lc)) * (1.0 - torch.exp(float(params.bohm_potential) - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif self.hes_settings.get("parallel_sheath_damping", 0) == 2:
                damp_sheath = avg_cs_hot / float(params.norm_lc) * (1.0 - torch.exp(float(params.bohm_potential) - avg_phi / torch.clamp(avg_te, min=1e-12)))
            elif self.hes_settings.get("parallel_sheath_damping", 0) == 3:
                damp_sheath = cs_hot / float(params.norm_lc) * (1.0 - torch.exp(float(params.bohm_potential) - phi / torch.clamp(te, min=1e-12)))
            else:
                raise ValueError("Unsupported parallel_sheath_damping option from BOUT-HESEL.")
            parallel["lnpi"] = parallel["lnpi"] + 2.0 / 3.0 * sigma_open * damp_sheath
            parallel["vort"] = parallel["vort"] + sigma_open * damp_sheath

            if self.hes_settings.get("parallel_conduction", 0) == 0:
                damp_she = torch.zeros_like(n)
                damp_shi = torch.zeros_like(n)
            elif self.hes_settings.get("parallel_conduction", 0) == 1:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / float(params.norm_taushe)
                damp_shi = torch.zeros_like(n)
            elif self.hes_settings.get("parallel_conduction", 0) == 2:
                damp_she = te * te * torch.sqrt(torch.clamp(te, min=1e-12)) / float(params.norm_taushe)
                damp_shi = ti * ti * torch.sqrt(torch.clamp(ti, min=1e-12)) / float(params.norm_taushi)
            else:
                raise ValueError("Unsupported parallel_conduction option from BOUT-HESEL.")

            pert_n, pert_te, pert_phi = n - avg_n, te - avg_te, phi - avg_phi
            if self.hes_settings.get("parallel_drift_wave", 0) == 0:
                driftwave = torch.zeros_like(n)
            elif self.hes_settings.get("parallel_drift_wave", 0) == 1:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / float(params.norm_taudw)
            elif self.hes_settings.get("parallel_drift_wave", 0) == 2:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / float(params.norm_taudw) * avg_te * torch.sqrt(torch.clamp(avg_te, min=1e-12))
            elif self.hes_settings.get("parallel_drift_wave", 0) == 3:
                driftwave = (pert_te + pert_n * (avg_te / torch.clamp(avg_n, min=1e-12)) - pert_phi) / float(params.norm_taudw) * te * torch.sqrt(torch.clamp(te, min=1e-12))
            else:
                raise ValueError("Unsupported parallel_drift_wave option from BOUT-HESEL.")
            parallel["vort"] = parallel["vort"] - sigma_closed * driftwave

            if self.hes_settings.get("reciprocal_approx", 1) == 1:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti
            elif self.hes_settings.get("reciprocal_approx", 1) == 2:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi / torch.clamp(avg_n, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave / torch.clamp(avg_n, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave
            elif self.hes_settings.get("reciprocal_approx", 1) == 3:
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * sigma_open * damp_she * (te - te_bck) / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_open * damp_shi * (te - ti_bck) / torch.clamp(pi, min=1e-12)
                parallel["lnn"] = parallel["lnn"] - sigma_closed * driftwave / torch.clamp(n, min=1e-12)
                parallel["lnpe"] = parallel["lnpe"] - 2.0 / 3.0 * 3.21 * sigma_closed * driftwave * avg_te / torch.clamp(pe, min=1e-12)
                parallel["lnpi"] = parallel["lnpi"] - 2.0 / 3.0 * sigma_closed * driftwave * avg_n * avg_ti / torch.clamp(pi, min=1e-12)
            else:
                raise ValueError("Unsupported reciprocal_approx option from BOUT-HESEL.")

        forcing = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if self.hes_settings.get("force_profiles", False):
            if not self.hes_settings.get("not_n_force", False):
                forcing["lnn"] = forcing["lnn"] + sigma_force * (init_n / torch.clamp(n, min=1e-12) - 1.0) / force_time
            if not self.hes_settings.get("not_p_force", False):
                forcing_multiplier = torch.ones_like(pi)
                if self.hes_settings.get("h_mode", False):
                    forcing_multiplier = 1.0 + (float(self.hes_settings.get("ramp_a", 2.0)) - 1.0) / 2.0 * (
                        torch.tanh((t_phys - float(self.hes_settings.get("ramp_t0", 0.0))) / float(self.hes_settings.get("ramp_trans", 5000.0)))
                        - torch.tanh((t_phys - float(self.hes_settings.get("ramp_t0", 0.0)) - float(self.hes_settings.get("ramp_peak", 50000.0))) / float(self.hes_settings.get("ramp_trans", 5000.0)))
                    )
                force_pe = sigma_force * (init_pe - pe) / force_time
                force_pi = sigma_force * (init_pi * forcing_multiplier - pi) / force_time
                forcing["lnpe"] = forcing["lnpe"] + force_pe / torch.clamp(pe, min=1e-12)
                forcing["lnpi"] = forcing["lnpi"] + force_pi / torch.clamp(pi, min=1e-12)

        floor_terms = {name: torch.zeros_like(term) for name, term in interchange.items()}
        if self.hes_settings.get("floor_profiles", False):
            floor_terms["lnn"] = floor_terms["lnn"] + torch.where(n < floor_n, (floor_n / torch.clamp(n, min=1e-12) - 1.0) / floor_time, torch.zeros_like(n))
            floor_terms["lnpe"] = floor_terms["lnpe"] + torch.where(pe < floor_pe, (floor_pe / torch.clamp(pe, min=1e-12) - 1.0) / floor_time, torch.zeros_like(pe))
            floor_terms["lnpi"] = floor_terms["lnpi"] + torch.where(pi < floor_pi, (floor_pi / torch.clamp(pi, min=1e-12) - 1.0) / floor_time, torch.zeros_like(pi))


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
