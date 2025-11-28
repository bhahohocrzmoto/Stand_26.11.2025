#!/usr/bin/env python3
"""
Script for generating plots and CSV summaries from FastSolver outputs,
with multi-port reduction (phases / windings) on top of per-trace matrices.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Frequencies where we record tabulated values
KEY_FREQS = [10e3, 50e3, 100e3, 200e3, 500e3, 1e6]
# Reference frequency for summaries / transformer metrics
REF_FREQ = 100e3

# Debug log key for JSON log files written by this module
DEBUG_LOG_NAME = "PlotGeneration_Debug.json"


def normalize_address_path(raw: Path | str) -> Path:
    """Return a resolved Address.txt path from user input."""
    cleaned = str(raw).strip().strip('"').strip("'")
    path = Path(cleaned).expanduser()
    if path.is_dir():
        path = path / "Address.txt"
    return path.resolve()


def read_addresses(address_path: Path) -> List[Path]:
    """Read non-empty, non-comment lines from Address.txt as paths."""
    addresses: List[Path] = []
    base = address_path.parent
    with address_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip().strip('"').strip("'")
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped)
            if not path.is_absolute():
                path = base / path
            addresses.append(path)
    return addresses


def ensure_analysis_dirs(spiral_path: Path) -> Dict[str, Path]:
    """Ensure Analysis/ subfolders exist and return a small path dict."""
    analysis = spiral_path / "Analysis"
    matrices_dir = analysis / "matrices"
    ports_dir = analysis / "ports"
    analysis.mkdir(exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    ports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "analysis": analysis,
        "matrices": matrices_dir,
        "ports": ports_dir,
        "ports_config": analysis / "ports_config.json",
        "summary_spiral": analysis / "summary_spiral.csv",
        "transformer_metrics": analysis / "transformer_metrics.csv",
        "debug_log": spiral_path.parent / DEBUG_LOG_NAME,
    }


def append_debug_entry(log_path: Path, *, spiral: str, stage: str, status: str, detail: str) -> None:
    """Append a structured debug entry to a JSON log file."""
    entry = {"spiral": spiral, "stage": stage, "status": status, "detail": detail}
    try:
        if log_path.exists():
            data = json.loads(log_path.read_text())
            if not isinstance(data, list):
                data = []
        else:
            data = []
        data.append(entry)
        log_path.write_text(json.dumps(data, indent=2))
    except Exception:
        return


def load_capacitance_matrix(path: Path) -> np.ndarray:
    """Load FasterCap-style text capacitance matrix into a dense numpy.array."""
    lines: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                numbers = [float(x) for x in stripped.replace(",", " ").split()]
            except ValueError:
                continue
            if numbers:
                lines.append(numbers)
    if not lines:
        raise ValueError("Capacitance matrix is empty or unreadable")
    return np.array(lines, dtype=float)


def select_first_match(data: dict, candidates: List[str]) -> str | None:
    for key in candidates:
        if key in data:
            return key
        for k in data:
            if k.lower() == key.lower():
                return k
    return None

def load_impedance_and_freq(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load frequency vector and impedance matrices from Zc.mat."""
    text = mat_path.read_text(encoding="utf-8", errors="ignore")
    if "Impedance matrix for frequency" in text:
        lines = text.splitlines()
        freqs, mats = [], []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Impedance matrix for frequency"):
                m = re.search(r"=\s*([^\s]+)\s+(\d+)\s*x\s*(\d+)", line)
                if not m: raise ValueError(f"Could not parse header: {line!r}")
                freq_val, nrows, ncols = float(m.group(1)), int(m.group(2)), int(m.group(3))
                i += 1
                rows = []
                for _ in range(nrows):
                    if i >= len(lines): raise ValueError("Unexpected EOF")
                    row_line = lines[i].strip()
                    tokens = row_line.split()
                    if len(tokens) != 2 * ncols: raise ValueError(f"Bad token count: {row_line!r}")
                    row_vals = [complex(tokens[2*c] + tokens[2*c+1]) for c in range(ncols)]
                    rows.append(row_vals)
                    i += 1
                mats.append(np.array(rows, dtype=complex))
                freqs.append(freq_val)
            else:
                i += 1
        if not mats:
            print("Warning: Found 'Impedance matrix' but no matrices; falling back to scipy.io.loadmat()")
        else:
            return np.array(freqs, dtype=float), np.stack(mats, axis=0)

    data = loadmat(mat_path)
    freq_key = select_first_match(data, ["freq", "frequency", "f"])
    z_key = select_first_match(data, ["Zc", "Z", "Z_matrix", "Zf"])
    if freq_key is None or z_key is None:
        raise ValueError("Could not find freq/impedance keys in Zc.mat")
    freq, Z = np.squeeze(np.array(data[freq_key], dtype=float)), np.array(data[z_key])
    if Z.ndim == 2: Z = Z[np.newaxis, :, :]
    if Z.shape[0] != freq.shape[0]:
        if Z.shape[-1] == freq.shape[0]: Z = np.moveaxis(Z, -1, 0)
        else: raise ValueError(f"Freq/impedance shape mismatch: {freq.shape[0]} vs {Z.shape}")
    return freq, Z


def compute_current_pattern(port_def: Dict[str, object], n: int) -> np.ndarray:
    """Convert a port definition into a conductor-current pattern alpha."""
    port_type = str(port_def.get("type", "")).lower()
    signs = np.array(port_def.get("signs", []), dtype=float).reshape(-1)
    if signs.size != n: raise ValueError(f"Expected {n} signs, got {signs.size}")
    active = signs != 0
    if port_type == "parallel":
        active_count = np.count_nonzero(active)
        if active_count == 0: raise ValueError("Parallel port has no active conductors")
        return np.where(active, signs / active_count, 0.0)
    return signs.astype(float)


def build_grouping_matrix_from_ports(ports: Dict[str, Dict[str, object]], n_conductors: int) -> Tuple[np.ndarray, List[str]]:
    """Build the N x P grouping matrix W from a ports dict."""
    if not ports: raise ValueError("No ports defined")
    port_names = sorted(ports.keys())
    cols = [compute_current_pattern(ports[name], n_conductors).reshape(-1) for name in port_names]
    return np.stack(cols, axis=1), port_names


def compute_R_L(freq: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split Z(f) into R(f) and L(f)."""
    R = np.real(Z)
    L = np.zeros_like(Z, dtype=float)
    non_zero_freq_indices = freq != 0
    L[non_zero_freq_indices] = np.imag(Z[non_zero_freq_indices]) / (2 * math.pi * freq[non_zero_freq_indices])
    return R, L


def effective_values_from_diag(freq: np.ndarray, R_diag: np.ndarray, L_diag: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute L, R, Q vectors for a single port."""
    R_eff, L_eff = np.asarray(R_diag, dtype=float).reshape(-1), np.asarray(L_diag, dtype=float).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = (2.0 * math.pi * freq * L_eff) / R_eff
    return L_eff, R_eff, Q


def find_resonance(freq: np.ndarray, Zin: np.ndarray) -> float:
    """Estimate the first series resonance."""
    imag_part = np.imag(Zin)
    signs = np.sign(imag_part)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    if sign_changes.size == 0: return float("nan")
    idx = sign_changes[0]
    f1, f2, y1, y2 = freq[idx], freq[idx+1], imag_part[idx], imag_part[idx+1]
    return float(f1 - y1 * (f2 - f1) / (y2 - y1)) if y2 != y1 else float(f1)


def interpolate_values(targets: List[float], freq: np.ndarray, values: np.ndarray) -> Dict[float, float]:
    """Interpolate values at target frequencies."""
    return {t: float(np.interp(t, freq, values)) for t in targets}


def build_matrix_payload(
    *, spiral_name: str, analysis_type: str, port_names: List[str], freq: np.ndarray,
    C_port: np.ndarray, R_port: np.ndarray, L_port: np.ndarray, source_files: Dict[str, str],
) -> Dict[str, object]:
    """Prepare a serialisable dict containing port-domain matrices."""
    return {
        "spiral_name": spiral_name, "analysis_type": analysis_type, "port_names": port_names,
        "frequencies_Hz": np.asarray(freq, dtype=float).tolist(),
        "units": {"C": "F", "R": "ohm", "L": "H"},
        "source_files": source_files,
        "matrices": {
            "C_port": np.asarray(C_port, dtype=float).tolist(),
            "R_port": np.asarray(R_port, dtype=float).tolist(),
            "L_port": np.asarray(L_port, dtype=float).tolist(),
        },
    }

def plot_vs_frequency(freq: np.ndarray, values: np.ndarray, ylabel: str, title: str, path: Path, logx: bool = True):
    plt.figure()
    plt.semilogx(freq, values) if logx else plt.plot(freq, values)
    plt.xlabel("Frequency (Hz)"), plt.ylabel(ylabel), plt.title(title)
    plt.grid(True, which="both"), plt.tight_layout(), plt.savefig(path), plt.close()

def get_port_analysis_type(port_name: str) -> str:
    """Determine analysis type from port name."""
    if port_name.startswith('p') or port_name.startswith('s'):
        return 'transformer'
    if 'Series' in port_name:
        return 'series_inductor'
    if 'Parallel' in port_name:
        return 'parallel_inductor'
    return 'unknown'

def process_spiral(
    spiral_path: Path, global_records: List[Dict[str, object]], *,
    ports_override: Optional[Dict[str, Dict[str, object]]] = None, debug_log_path: Optional[Path] = None
) -> None:
    spiral_name = spiral_path.name
    fastsolver = spiral_path / "FastSolver"
    if not (fastsolver.exists() and (fastsolver / "CapacitanceMatrix.txt").exists() and (fastsolver / "Zc.mat").exists()):
        if debug_log_path: append_debug_entry(debug_log_path, spiral=spiral_name, stage="precheck", status="FAILURE", detail="Missing FastSolver outputs")
        return

    dirs = ensure_analysis_dirs(spiral_path)
    try:
        C_trace = load_capacitance_matrix(dirs["analysis"].parent / "FastSolver" / "CapacitanceMatrix.txt")
        freq, Z_trace = load_impedance_and_freq(dirs["analysis"].parent / "FastSolver" / "Zc.mat")
    except Exception as exc:
        if debug_log_path: append_debug_entry(debug_log_path, spiral=spiral_name, stage="load", status="FAILURE", detail=str(exc))
        return

    n = C_trace.shape[0]
    if C_trace.shape[1] != n or Z_trace.shape[1] != n or Z_trace.shape[2] != n:
        if debug_log_path: append_debug_entry(debug_log_path, spiral=spiral_name, stage="shape", status="FAILURE", detail="Matrix dimension mismatch")
        return
    
    ports = json.loads(dirs["ports_config"].read_text()).get("ports", {})
    if not ports:
        if debug_log_path: append_debug_entry(debug_log_path, spiral=spiral_name, stage="ports", status="FAILURE", detail="No ports defined in ports_config.json")
        return

    R_trace, L_trace = compute_R_L(freq, Z_trace)
    
    # Group ports by analysis type
    ports_by_type = {}
    for name, definition in ports.items():
        analysis_type = get_port_analysis_type(name)
        if analysis_type == 'unknown': continue
        if analysis_type not in ports_by_type:
            ports_by_type[analysis_type] = {}
        ports_by_type[analysis_type][name] = definition

    # Process each analysis type separately
    for analysis_type, type_ports in ports_by_type.items():
        W, port_names = build_grouping_matrix_from_ports(type_ports, n)
        WT = W.T
        
        C_port = WT @ C_trace @ W
        R_port = np.einsum('ij,kjl,lm->kim', WT, R_trace, W)
        L_port = np.einsum('ij,kjl,lm->kim', WT, L_trace, W)

        payload = build_matrix_payload(
            spiral_name=spiral_name, analysis_type=analysis_type, port_names=port_names, freq=freq,
            C_port=C_port, R_port=R_port, L_port=L_port,
            source_files={"Zc": str(fastsolver / "Zc.mat"), "CapacitanceMatrix": str(fastsolver / "CapacitanceMatrix.txt")},
        )
        json_name = f"{analysis_type}_matrices.json"
        write_matrix_json(payload, dirs["matrices"] / json_name)
        if debug_log_path:
            append_debug_entry(debug_log_path, spiral=spiral_name, stage="analysis", status="SUCCESS", detail=f"Generated {json_name} with ports: {', '.join(port_names)}")

    # Restore per-port metrics and global records generation
    summary_rows_all: List[Dict[str, object]] = []
    W_all, port_names_all = build_grouping_matrix_from_ports(ports, n)
    WT_all = W_all.T
    R_port_all = np.einsum('ij,kjl,lm->kim', WT_all, R_trace, W)
    L_port_all = np.einsum('ij,kjl,lm->kim', WT_all, L_trace, W)

    for p_idx, port_name in enumerate(port_names_all):
        R_diag, L_diag = R_port_all[:, p_idx, p_idx], L_port_all[:, p_idx, p_idx]
        L_eff, R_eff, Q = effective_values_from_diag(freq, R_diag, L_diag)
        Zin = R_eff + 1j * 2 * math.pi * freq * L_eff
        resonance = find_resonance(freq, Zin)
        
        ref_L = float(np.interp(REF_FREQ, freq, L_eff))
        ref_R = float(np.interp(REF_FREQ, freq, R_eff))
        ref_Q = float(np.interp(REF_FREQ, freq, Q))

        global_records.append({
            "spiral_name": spiral_name, "port_name": port_name, "ref_freq_Hz": REF_FREQ,
            "L_eff_H": ref_L, "R_eff_ohm": ref_R, "Q": ref_Q, "first_resonance_Hz": resonance,
            "N_conductors": n, "N_ports": len(port_names_all),
            "system_type": json.loads(dirs["ports_config"].read_text()).get("system_type", "auto"),
        })

def write_matrix_json(payload: Dict[str, object], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

def main() -> None:
    address_path = normalize_address_path(input("Enter path to Address.txt: "))
    if not address_path.exists():
        print(f"Path does not exist: {address_path}")
        return
        
    spirals = read_addresses(address_path)
    debug_log = address_path.parent / DEBUG_LOG_NAME
    debug_log.unlink(missing_ok=True)
    global_records: List[Dict[str, object]] = []

    for spiral_path in spirals:
        if not spiral_path.exists():
            append_debug_entry(debug_log, spiral=spiral_path.name, stage="precheck", status="FAILURE", detail="Spiral folder missing")
            continue
        try:
            process_spiral(spiral_path, global_records, debug_log_path=debug_log)
        except Exception as exc:
            append_debug_entry(debug_log, spiral=spiral_path.name, stage="unexpected", status="FAILURE", detail=str(exc))
    print("Processing complete.")

if __name__ == "__main__":
    main()