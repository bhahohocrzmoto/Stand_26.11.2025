#!/usr/bin/env python3
"""
Aggregate transformer metrics across all spiral variants into human-friendly summaries.

Outputs (written next to Address.txt under "Final Summary Transformer/"):
  - coverage_report.csv           : which folders are missing transformer_matrices.json
  - full_metrics_long.csv         : long table (one row per frequency per phase pair)
  - overview_ref_freq.csv         : single snapshot at reference frequency (includes SRF, capacitance)
  - coupling.xlsx                 : one sheet per frequency (k, M)
  - self_inductance.xlsx          : one sheet per frequency (L_pp, L_ss)
  - turns_ratio.xlsx              : one sheet per frequency (Np/Ns)
  - leakage.xlsx                  : one sheet per frequency (leakage inductances)
  - resistance.xlsx               : one sheet per frequency (R, Q)
  - capacitance.xlsx              : single sheet (self/mutual C + SRF estimates)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path so we can import PlotGeneration
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FastSolver.PlotGeneration import PlotGeneration as PG  # type: ignore  # noqa: E402


REF_FREQ = PG.REF_FREQ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate transformer metrics across spirals.")
    parser.add_argument(
        "address",
        nargs="?",
        help="Path to Address.txt (defaults to prompt if omitted).",
    )
    parser.add_argument(
        "--ref-freq",
        type=float,
        default=REF_FREQ,
        help=f"Reference frequency for SRF/overview (default: {REF_FREQ} Hz).",
    )
    return parser.parse_args()


def safe_sheet_name(freq_hz: float) -> str:
    """Build a short sheet name for a given frequency (Excel limit: 31 chars)."""
    hz = float(freq_hz)
    if hz >= 1e6:
        val = hz / 1e6
        unit = "MHz"
    elif hz >= 1e3:
        val = hz / 1e3
        unit = "kHz"
    else:
        val = hz
        unit = "Hz"
    name = f"f_{val:g}{unit}"
    return name[:31]


def build_phase_pairs(port_names: List[str]) -> List[Tuple[str, int, int]]:
    """Return (phase_key, p_idx, s_idx) tuples for every primary/secondary pairing."""
    roles = [PG.decode_port_role(name) for name in port_names]
    prim_by_phase: Dict[str, List[int]] = {}
    sec_by_phase: Dict[str, List[int]] = {}
    for idx, (role, phase_key) in enumerate(roles):
        if role == "primary":
            prim_by_phase.setdefault(phase_key, []).append(idx)
        elif role == "secondary":
            sec_by_phase.setdefault(phase_key, []).append(idx)

    pairs: List[Tuple[str, int, int]] = []
    for phase_key, prim_list in prim_by_phase.items():
        for p_idx in prim_list:
            for s_idx in sec_by_phase.get(phase_key, []):
                pairs.append((phase_key or "-", p_idx, s_idx))
    return pairs


def compute_srf(L_val: float, C_val: float) -> float:
    """Simple self-resonant frequency estimate from L and C (ignores losses)."""
    if L_val <= 0.0 or C_val <= 0.0:
        return math.nan
    try:
        return 1.0 / (2.0 * math.pi * math.sqrt(L_val * C_val))
    except Exception:
        return math.nan


def interpolate(values: np.ndarray, freq: np.ndarray, target: float) -> float:
    """1D interpolation with guard against empty arrays."""
    if values.size == 0 or freq.size == 0:
        return math.nan
    return float(np.interp(target, freq, values))


def load_matrix_payload(path: Path):
    data = PG.load_matrix_json(path)
    port_names: List[str] = list(data.get("port_names", []))
    freq = np.array(data.get("frequencies_Hz", []), dtype=float)
    matrices: Dict[str, List] = data.get("matrices", {})  # type: ignore[assignment]

    C_port = np.array(matrices.get("C_port", []), dtype=float)
    R_port = np.array(matrices.get("R_port", []), dtype=float)
    L_port = np.array(matrices.get("L_port", []), dtype=float)
    return port_names, freq, C_port, R_port, L_port


def process_transformer_file(
    spiral_name: str,
    json_path: Path,
    ref_freq: float,
    rows_full: List[Dict[str, object]],
    rows_ref: List[Dict[str, object]],
    rows_cap: List[Dict[str, object]],
) -> None:
    port_names, freq, C_port, R_port, L_port = load_matrix_payload(json_path)

    if not port_names or freq.size == 0:
        return

    n_ports = len(port_names)
    if (
        C_port.shape != (n_ports, n_ports)
        or R_port.ndim != 3
        or L_port.ndim != 3
        or R_port.shape[1] != n_ports
        or L_port.shape[1] != n_ports
    ):
        return

    pairs = build_phase_pairs(port_names)
    if not pairs:
        return

    c_self = {idx: abs(float(C_port[idx, idx])) for idx in range(n_ports)}

    for phase_key, p_idx, s_idx in pairs:
        L_pp_all = L_port[:, p_idx, p_idx]
        L_ss_all = L_port[:, s_idx, s_idx]
        M_ps_all = L_port[:, p_idx, s_idx]
        R_p_all = R_port[:, p_idx, p_idx]
        R_s_all = R_port[:, s_idx, s_idx]

        c_p = c_self.get(p_idx, math.nan)
        c_s = c_self.get(s_idx, math.nan)
        c_mutual = abs(float(C_port[p_idx, s_idx])) if C_port.size else math.nan

        # Reference-frequency snapshot (includes SRF and capacitance)
        L_pp_ref = interpolate(L_pp_all, freq, ref_freq)
        L_ss_ref = interpolate(L_ss_all, freq, ref_freq)
        M_ref = interpolate(M_ps_all, freq, ref_freq)
        R_p_ref = interpolate(R_p_all, freq, ref_freq)
        R_s_ref = interpolate(R_s_all, freq, ref_freq)

        k_ref = (
            M_ref / math.sqrt(L_pp_ref * L_ss_ref)
            if L_pp_ref > 0 and L_ss_ref > 0
            else math.nan
        )
        if math.isfinite(k_ref):
            k_ref = max(-1.0, min(1.0, k_ref))

        turns_ref = math.sqrt(L_pp_ref / L_ss_ref) if L_ss_ref > 0 else math.nan
        q_p_ref = (
            (2.0 * math.pi * ref_freq * L_pp_ref) / R_p_ref
            if math.isfinite(R_p_ref) and R_p_ref != 0.0
            else math.nan
        )
        q_s_ref = (
            (2.0 * math.pi * ref_freq * L_ss_ref) / R_s_ref
            if math.isfinite(R_s_ref) and R_s_ref != 0.0
            else math.nan
        )

        rows_ref.append(
            {
                "spiral_name": spiral_name,
                "phase_key": phase_key,
                "primary_port": port_names[p_idx],
                "secondary_port": port_names[s_idx],
                "ref_freq_Hz": ref_freq,
                "k_coupling": k_ref,
                "turns_ratio_NpNs": turns_ref,
                "L_pp_H": L_pp_ref,
                "L_ss_H": L_ss_ref,
                "M_ps_H": M_ref,
                "R_primary_ohm": R_p_ref,
                "R_secondary_ohm": R_s_ref,
                "Q_primary": q_p_ref,
                "Q_secondary": q_s_ref,
                "L_leak_primary_H": L_pp_ref - M_ref if math.isfinite(L_pp_ref) else math.nan,
                "L_leak_secondary_H": L_ss_ref - M_ref if math.isfinite(L_ss_ref) else math.nan,
                "C_self_primary_F": c_p,
                "C_self_secondary_F": c_s,
                "C_mutual_abs_F": c_mutual,
                "f_srf_primary_Hz": compute_srf(L_pp_ref, c_p),
                "f_srf_secondary_Hz": compute_srf(L_ss_ref, c_s),
            }
        )

        rows_cap.append(
            {
                "spiral_name": spiral_name,
                "phase_key": phase_key,
                "primary_port": port_names[p_idx],
                "secondary_port": port_names[s_idx],
                "C_self_primary_F": c_p,
                "C_self_secondary_F": c_s,
                "C_mutual_abs_F": c_mutual,
                "f_srf_primary_Hz": compute_srf(L_pp_ref, c_p),
                "f_srf_secondary_Hz": compute_srf(L_ss_ref, c_s),
            }
        )

        for idx, f in enumerate(freq):
            L_pp = float(L_pp_all[idx])
            L_ss = float(L_ss_all[idx])
            M_ps = float(M_ps_all[idx])
            R_p = float(R_p_all[idx])
            R_s = float(R_s_all[idx])

            if L_pp <= 0.0 or L_ss <= 0.0:
                k_c = math.nan
                turns_ratio = math.nan
            else:
                k_c = M_ps / math.sqrt(L_pp * L_ss)
                k_c = max(-1.0, min(1.0, k_c))
                turns_ratio = math.sqrt(L_pp / L_ss)

            q_p = (2.0 * math.pi * f * L_pp) / R_p if math.isfinite(R_p) and R_p != 0.0 else math.nan
            q_s = (2.0 * math.pi * f * L_ss) / R_s if math.isfinite(R_s) and R_s != 0.0 else math.nan

            rows_full.append(
                {
                    "spiral_name": spiral_name,
                    "phase_key": phase_key,
                    "primary_port": port_names[p_idx],
                    "secondary_port": port_names[s_idx],
                    "freq_Hz": float(f),
                    "k_coupling": k_c,
                    "turns_ratio_NpNs": turns_ratio,
                    "L_pp_H": L_pp,
                    "L_ss_H": L_ss,
                    "M_ps_H": M_ps,
                    "L_leak_primary_H": L_pp - M_ps,
                    "L_leak_secondary_H": L_ss - M_ps,
                    "R_primary_ohm": R_p,
                    "R_secondary_ohm": R_s,
                    "Q_primary": q_p,
                    "Q_secondary": q_s,
                    "C_self_primary_F": c_p,
                    "C_self_secondary_F": c_s,
                    "C_mutual_abs_F": c_mutual,
                }
            )


def write_per_param_workbook(rows_full: List[Dict[str, object]], out_path: Path, columns: List[str]) -> None:
    """Write one workbook with a sheet per frequency for the selected columns."""
    if not rows_full:
        return

    buckets: Dict[float, List[Dict[str, object]]] = {}
    for row in rows_full:
        f = float(row["freq_Hz"])
        buckets.setdefault(f, []).append(row)

    with pd.ExcelWriter(out_path) as writer:
        for freq_val in sorted(buckets.keys()):
            df = pd.DataFrame(buckets[freq_val])
            subset = [col for col in columns if col in df.columns]
            if not subset:
                continue
            sheet_df = df[subset]
            sheet_df.to_excel(writer, sheet_name=safe_sheet_name(freq_val), index=False)


def main() -> None:
    args = parse_args()
    if args.address:
        address_path = PG.normalize_address_path(args.address)
        if not address_path.exists():
            print(f"Address.txt not found: {address_path}")
            return
    else:
        address_path = PG.prompt_address_path()
        if address_path is None:
            return

    spirals = PG.read_addresses(address_path)
    summary_dir = address_path.parent / "Final Summary Transformer"
    summary_dir.mkdir(exist_ok=True)

    coverage_rows: List[Dict[str, object]] = []
    rows_full: List[Dict[str, object]] = []
    rows_ref: List[Dict[str, object]] = []
    rows_cap: List[Dict[str, object]] = []

    for spiral_path in spirals:
        spiral_name = spiral_path.name
        if not spiral_path.exists():
            coverage_rows.append({"spiral_name": spiral_name, "status": "missing_folder", "details": str(spiral_path)})
            continue

        matrices_dir = spiral_path / "Analysis" / "matrices"
        tx_json = matrices_dir / "transformer_matrices.json"
        if not tx_json.exists():
            coverage_rows.append({"spiral_name": spiral_name, "status": "missing_transformer_matrices", "details": str(tx_json)})
            continue

        coverage_rows.append({"spiral_name": spiral_name, "status": "ok", "details": str(tx_json)})
        try:
            process_transformer_file(spiral_name, tx_json, args.ref_freq, rows_full, rows_ref, rows_cap)
        except Exception as exc:  # noqa: BLE001
            coverage_rows.append({"spiral_name": spiral_name, "status": "error", "details": str(exc)})

    # Coverage report
    pd.DataFrame(coverage_rows).to_csv(summary_dir / "coverage_report.csv", index=False)

    if not rows_full:
        print("No transformer_matrices.json files processed. Summary not generated.")
        return

    # Core tables
    pd.DataFrame(rows_full).to_csv(summary_dir / "full_metrics_long.csv", index=False)
    pd.DataFrame(rows_ref).to_csv(summary_dir / "overview_ref_freq.csv", index=False)
    pd.DataFrame(rows_cap).to_excel(summary_dir / "capacitance.xlsx", index=False)

    # Per-parameter workbooks (one sheet per frequency)
    write_per_param_workbook(
        rows_full,
        summary_dir / "coupling.xlsx",
        ["spiral_name", "phase_key", "primary_port", "secondary_port", "k_coupling", "M_ps_H", "C_mutual_abs_F"],
    )
    write_per_param_workbook(
        rows_full,
        summary_dir / "self_inductance.xlsx",
        ["spiral_name", "phase_key", "primary_port", "secondary_port", "L_pp_H", "L_ss_H"],
    )
    write_per_param_workbook(
        rows_full,
        summary_dir / "turns_ratio.xlsx",
        ["spiral_name", "phase_key", "primary_port", "secondary_port", "turns_ratio_NpNs"],
    )
    write_per_param_workbook(
        rows_full,
        summary_dir / "leakage.xlsx",
        ["spiral_name", "phase_key", "primary_port", "secondary_port", "L_leak_primary_H", "L_leak_secondary_H"],
    )
    write_per_param_workbook(
        rows_full,
        summary_dir / "resistance.xlsx",
        ["spiral_name", "phase_key", "primary_port", "secondary_port", "R_primary_ohm", "R_secondary_ohm", "Q_primary", "Q_secondary"],
    )

    print(f"Final transformer summary written to: {summary_dir}")


if __name__ == "__main__":
    main()
