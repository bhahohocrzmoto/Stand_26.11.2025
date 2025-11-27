#!/usr/bin/env python3
"""
Central orchestration GUI for spiral generation, solver automation, and plotting.

This panel keeps the existing specialised UIs but wires them together so a user can:
1) Launch the spiral batch UI to generate geometry + Address.txt
2) Verify Address.txt contents
3) Run FastSolver conversion + solver automation with a chosen permittivity
4) Configure ports in a friendlier popup and generate plots
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parents[1]
SPIRAL_UI = REPO_ROOT / "SpiralGeometryGeneration" / "Spiral_Batch_Variants_UI_16.11.2025.py"
FAST_UI = REPO_ROOT / "FastSolver" / "Automation" / "fast_solver_batch_ui.py"
AUTOMATE = REPO_ROOT / "FastSolver" / "Automation" / "automate_solvers.py"
PLOT_GEN = REPO_ROOT / "FastSolver" / "PlotGeneration" / "PlotGeneration.py"

sys.path.insert(0, str(REPO_ROOT))
from FastSolver.PlotGeneration import PlotGeneration as PG  # type: ignore  # noqa: E402


# ---------------- helpers -----------------

PHASE_LETTERS = ("A", "B", "C")


def read_address_entries(address_file: Path) -> List[Path]:
    cleaned = address_file.read_text().splitlines()
    entries: List[Path] = []
    for line in cleaned:
        stripped = line.strip().strip('"').strip("'")
        if not stripped:
            continue
        p = Path(stripped)
        if not p.is_absolute():
            p = address_file.parent / p
        entries.append(p.resolve())
    return entries


def parse_spiral_folder_name(name: str) -> List[Dict[str, object]]:
    """Extract layer metadata (layer index, K, direction) from a folder name."""

    matches = list(
        re.finditer(r"L(?P<layer>\d+)_K(?P<K>\d+)_N[^_]+_(?P<dir>CW|CCW)", name)
    )
    info: List[Dict[str, object]] = []
    offset = 0
    for m in matches:
        layer_idx = int(m.group("layer"))
        k = int(m.group("K"))
        direction = m.group("dir")
        info.append({"layer": layer_idx, "K": k, "direction": direction, "start": offset})
        offset += k
    return info


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_sign_vector(active_indices: Sequence[int], total: int, *, type_: str = "parallel") -> List[float]:
    """
    Build a raw ±1 sign vector for the selected conductors.

    IMPORTANT:
      - No normalisation is done here.
      - PlotGeneration.compute_current_pattern() will normalise 'parallel'
        ports so that the sum of magnitudes is 1 A.
    """
    signs = [0.0] * total
    for idx in active_indices:
        if 0 <= idx < total:
            signs[idx] = 1.0
    return signs



def log_subprocess(cmd: List[str], log_widget: tk.Text) -> bool:
    log_widget.insert("end", f"\n$ {' '.join(cmd)}\n")
    log_widget.see("end")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if proc.stdout:
            log_widget.insert("end", proc.stdout)
        if proc.stderr:
            log_widget.insert("end", proc.stderr)
        log_widget.see("end")
        return True
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        log_widget.insert("end", exc.stdout or "")
        log_widget.insert("end", exc.stderr or "")
        log_widget.insert("end", f"Command failed: {exc}\n")
        log_widget.see("end")
        messagebox.showerror("Command failed", f"{cmd[0]} exited with status {exc.returncode}")
        return False


# ---------------- Port configuration popup -----------------

class PortsPopup(tk.Toplevel):
    def __init__(self, master: tk.Tk, address_file: Path, log_widget: tk.Text):
        super().__init__(master)
        self.title("PlotGeneration configuration")
        self.address_file = address_file
        self.log = log_widget
        self.geometry("1020x640")
        self.transient(master)
        self.grab_set()

        self.spiral_paths = self._load_spiral_paths()
        self.layer_cache: Dict[Path, List[Dict[str, object]]] = {}

        self._build_ui()

    def _load_spiral_paths(self) -> List[Path]:
        try:
            paths = read_address_entries(self.address_file)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Address read error", str(exc), parent=self)
            return []
        existing = [p for p in paths if p.exists()]
        if not existing:
            messagebox.showwarning("No folders", "No folders from Address.txt are present.", parent=self)
        return existing

    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=False, padx=8, pady=8)

        ttk.Label(left, text="Spiral variations (from Address.txt)").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("name", "conductors"), show="headings", height=20)
        self.tree.heading("name", text="Folder")
        self.tree.heading("conductors", text="# conductors")
        self.tree.column("name", width=420)
        self.tree.column("conductors", width=100, anchor="center")
        self.tree.pack(fill="both", expand=True)
        for path in self.spiral_paths:
            n = self._count_conductors(path)
            self.tree.insert("", "end", iid=str(path), values=(path.name, n))

        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # --- Inductor options
        ind_frame = ttk.LabelFrame(right, text="Inductor analysis")
        ind_frame.pack(fill="x", pady=6)
        self.var_enable_inductor = tk.BooleanVar(value=True)
        ttk.Checkbutton(ind_frame, text="Enable inductor analysis", variable=self.var_enable_inductor).pack(anchor="w", padx=6, pady=2)
        series_row = ttk.Frame(ind_frame); series_row.pack(fill="x", padx=6, pady=2)
        self.var_series = tk.BooleanVar(value=True)
        ttk.Checkbutton(series_row, text="Series (Port_all_Series)", variable=self.var_series).pack(side="left")
        self.var_parallel = tk.BooleanVar(value=True)
        ttk.Checkbutton(series_row, text="Parallel (Port_all_Parallel)", variable=self.var_parallel).pack(side="left", padx=12)
        ttk.Label(ind_frame, text="Requires Wire_Sections.txt. Parallel also requires Zc.mat and CapacitanceMatrix.txt.").pack(anchor="w", padx=6)

        # --- Transformer options
        tx_frame = ttk.LabelFrame(right, text="Transformer analysis")
        tx_frame.pack(fill="x", pady=6)
        self.var_enable_tx = tk.BooleanVar(value=False)
        ttk.Checkbutton(tx_frame, text="Enable transformer analysis", variable=self.var_enable_tx).pack(anchor="w", padx=6, pady=2)

        row1 = ttk.Frame(tx_frame); row1.pack(fill="x", padx=6, pady=2)
        ttk.Label(row1, text="Primary layers (comma separated):").pack(side="left")
        self.var_primary_layers = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.var_primary_layers, width=18).pack(side="left", padx=4)

        row2 = ttk.Frame(tx_frame); row2.pack(fill="x", padx=6, pady=2)
        ttk.Label(row2, text="Secondary layers (comma separated):").pack(side="left")
        self.var_secondary_layers = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.var_secondary_layers, width=18).pack(side="left", padx=4)

        row3 = ttk.Frame(tx_frame); row3.pack(fill="x", padx=6, pady=2)
        ttk.Label(row3, text="Phases per side:").pack(side="left")
        self.var_phase_count = tk.StringVar(value="1")
        ttk.Combobox(row3, values=("1", "2", "3"), textvariable=self.var_phase_count, width=6, state="readonly").pack(side="left", padx=4)
        ttk.Label(row3, text="(must evenly divide K for each selected layer)").pack(side="left", padx=4)

        map_frame = ttk.Frame(tx_frame)
        map_frame.pack(fill="both", padx=6, pady=4)
        ttk.Label(map_frame, text="Optional custom port mapping (format: pA:0,6 | sA:3,9)").pack(anchor="w")
        self.var_custom_ports = tk.Text(map_frame, height=4)
        self.var_custom_ports.pack(fill="x", expand=True)

        # --- Summary box
        self.summary = tk.Text(right, height=12)
        self.summary.pack(fill="both", expand=True, pady=(6, 0))
        self._refresh_summary()

        ttk.Label(
            right,
            text=(
                "After filling the fields, click 'Run PlotGeneration' below to build ports, "
                "run analyses, and write results under each spiral's Analysis folder (plus Global_Report)."
            ),
            wraplength=580,
            foreground="#404040",
        ).pack(fill="x", pady=(6, 0))

        action = ttk.Frame(self)
        action.pack(fill="x", side="bottom", pady=8, padx=10)
        ttk.Button(action, text="Run PlotGeneration", command=self._run_plots).pack(side="right", padx=6)
        ttk.Button(action, text="Cancel", command=self.destroy).pack(side="right")

    def _count_conductors(self, path: Path) -> int:
        # Prefer solver outputs, fall back to Wire_Sections conductor count
        cap = path / "FastSolver" / "CapacitanceMatrix.txt"
        if cap.exists():
            try:
                matrix = PG.load_capacitance_matrix(cap)
                return matrix.shape[0]
            except Exception:
                pass
        wire_sections = path / "Wire_Sections.txt"
        if wire_sections.exists():
            try:
                lines = [ln for ln in wire_sections.read_text().splitlines() if ln.strip()]
                return len(lines)
            except Exception:
                return 0
        return 0

    def _refresh_summary(self):
        self.summary.delete("1.0", "end")
        self.summary.insert("end", "Inductor: " + ("enabled" if self.var_enable_inductor.get() else "disabled") + "\n")
        if self.var_enable_inductor.get():
            self.summary.insert("end", f"  Series: {self.var_series.get()} | Parallel: {self.var_parallel.get()}\n")
        self.summary.insert("end", "Transformer: " + ("enabled" if self.var_enable_tx.get() else "disabled") + "\n")
        if self.var_enable_tx.get():
            self.summary.insert(
                "end",
                f"  Primary layers: {self.var_primary_layers.get() or '-'}; Secondary layers: {self.var_secondary_layers.get() or '-'}; Phases: {self.var_phase_count.get()}\n",
            )
            text = self.var_custom_ports.get("1.0", "end").strip()
            if text:
                self.summary.insert("end", f"  Custom ports: {text}\n")
        self.summary.see("end")

    def _parse_layer_selection(self, raw: str) -> List[int]:
        layers: List[int] = []
        for token in re.split(r"[;,\s]+", raw.strip()):
            if not token:
                continue
            try:
                layers.append(int(token))
            except ValueError:
                continue
        return layers

    def _get_layers_info(self, path: Path) -> List[Dict[str, object]]:
        if path not in self.layer_cache:
            self.layer_cache[path] = parse_spiral_folder_name(path.name)
        return self.layer_cache[path]

    def _validate_series(self, layers: List[Dict[str, object]]) -> Tuple[bool, List[float]]:
        if not layers:
            return False, []
        total = sum(int(item["K"]) for item in layers)
        # Condition 1: only one arm per layer
        if any(int(item["K"]) != 1 for item in layers):
            return False, []
        # Condition 2: directions must alternate in the order given
        directions = [str(item["direction"]) for item in layers]
        for idx in range(1, len(directions)):
            if directions[idx] == directions[idx - 1]:
                return False, []
        # Sign vector based on direction (CW = -1, CCW = +1)
        signs: List[float] = [0.0] * total
        dir_to_sign = {"CCW": 1.0, "CW": -1.0}
        for info in layers:
            start = int(info["start"])
            signs[start] = dir_to_sign.get(str(info["direction"]), 1.0)
        return True, signs

    def _parse_custom_ports(self, text: str, total: int) -> Dict[str, Dict[str, object]]:
        ports: Dict[str, Dict[str, object]] = {}
        tokens = []
        for line in text.replace("|", "\n").splitlines():
            line = line.strip()
            if not line:
                continue
            tokens.append(line)
        for token in tokens:
            if ":" not in token:
                continue
            name, raw_indices = token.split(":", 1)
            name = name.strip()
            indices: List[int] = []
            for val in re.split(r"[,\s]+", raw_indices.strip()):
                if not val:
                    continue
                try:
                    indices.append(int(val))
                except ValueError:
                    continue
            ports[name] = {
                "type": "parallel",
                "signs": build_sign_vector(indices, total, type_="parallel"),
                "raw_indices": ",".join(str(i) for i in indices),
            }
        return ports

    def _build_transformer_ports(
        self, layers: List[Dict[str, object]], primary_layers: List[int], secondary_layers: List[int], phase_count: int
    ) -> Optional[Dict[str, Dict[str, object]]]:
        if not primary_layers and not secondary_layers:
            return None
        layer_numbers = {int(info["layer"]) for info in layers}
        if any(layer not in layer_numbers for layer in primary_layers + secondary_layers):
            return None
        total = sum(int(item["K"]) for item in layers)
        # Build mapping from layer -> info for quick lookup
        layer_map = {int(item["layer"]): item for item in layers}

        def validate_phase_counts(selected: List[int]) -> bool:
            for layer in selected:
                info = layer_map.get(layer)
                if info is None:
                    return False
                k = int(info["K"])
                if phase_count == 0 or k % phase_count != 0:
                    return False
            return True

        if not validate_phase_counts(primary_layers) or not validate_phase_counts(secondary_layers):
            return None

        # Optional custom mapping overrides auto grouping
        custom_text = self.var_custom_ports.get("1.0", "end").strip()
        if custom_text:
            return self._parse_custom_ports(custom_text, total)

        ports: Dict[str, Dict[str, object]] = {}
        letters = PHASE_LETTERS[:phase_count]
        # Helper to compute indices per phase for a given side
        def collect_indices(selected_layers: List[int], phase_idx: int) -> List[int]:
            collected: List[int] = []
            for layer in selected_layers:
                info = layer_map[layer]
                start = int(info["start"])
                k = int(info["K"])
                per_phase = k // phase_count
                offset_start = start + phase_idx * per_phase
                collected.extend(list(range(offset_start, offset_start + per_phase)))
            return collected

        for idx, letter in enumerate(letters):
            if primary_layers:
                p_indices = collect_indices(primary_layers, idx)
                ports[f"p{letter}"] = {
                    "type": "parallel",
                    "signs": build_sign_vector(p_indices, total, type_="parallel"),
                    "raw_indices": ",".join(str(i) for i in p_indices),
                }
            if secondary_layers:
                s_indices = collect_indices(secondary_layers, idx)
                ports[f"s{letter}"] = {
                    "type": "parallel",
                    "signs": build_sign_vector(s_indices, total, type_="parallel"),
                    "raw_indices": ",".join(str(i) for i in s_indices),
                }

        return ports

    def _run_plots(self):
        self._refresh_summary()
        if not self.spiral_paths:
            messagebox.showwarning("No folders", "No spiral folders were loaded.", parent=self)
            return

        enable_inductor = self.var_enable_inductor.get() and (self.var_series.get() or self.var_parallel.get())
        enable_tx = self.var_enable_tx.get()
        if not enable_inductor and not enable_tx:
            messagebox.showwarning("Nothing to run", "Select at least one analysis mode.", parent=self)
            return

        base_dir = self.address_file.parent
        na_path = base_dir / "NotAnalyzed.txt"
        na_series = base_dir / "NotAnalyzedSeries.txt"
        na_parallel = base_dir / "NotAnalyzedParallel.txt"
        na_tx = base_dir / "NotAnalyzedTransformer.txt"

        try:
            phase_count = int(self.var_phase_count.get())
        except ValueError:
            phase_count = 1

        primary_layers = self._parse_layer_selection(self.var_primary_layers.get())
        secondary_layers = self._parse_layer_selection(self.var_secondary_layers.get())

        debug_log = self.address_file.parent / PG.DEBUG_LOG_NAME
        debug_log.unlink(missing_ok=True)

        records: List[Dict[str, object]] = []
        for path in self.spiral_paths:
            layers = self._get_layers_info(path)
            total = sum(int(item["K"]) for item in layers)
            if not total:
                append_line(na_path, path.name)
                PG.append_debug_entry(debug_log, spiral=path.name, stage="precheck", reason="No conductors found")
                continue
            wire_sections = path / "Wire_Sections.txt"
            if not wire_sections.exists():
                append_line(na_path, path.name)
                PG.append_debug_entry(debug_log, spiral=path.name, stage="precheck", reason="Wire_Sections.txt missing")
                continue

            ports: Dict[str, Dict[str, object]] = {}

            if enable_inductor:
                # Parallel mode
                if self.var_parallel.get():
                    zc_path = path / "FastSolver" / "Zc.mat"
                    cap_path = path / "FastSolver" / "CapacitanceMatrix.txt"
                    if not (zc_path.exists() and cap_path.exists()):
                        append_line(na_parallel, path.name)
                        append_line(na_path, path.name)
                        PG.append_debug_entry(debug_log, spiral=path.name, stage="inductor", reason="Missing Zc.mat/CapacitanceMatrix.txt for parallel analysis")
                    else:
                        indices = list(range(total))
                        ports["Port_all_Parallel"] = {
                            "type": "parallel",
                            "signs": build_sign_vector(indices, total, type_="parallel"),
                            "raw_indices": ",".join(str(i) for i in indices),
                        }

                # Series mode
                if self.var_series.get():
                    ok_series, signs = self._validate_series(layers)
                    if not ok_series:
                        append_line(na_series, path.name)
                        append_line(na_path, path.name)
                        PG.append_debug_entry(debug_log, spiral=path.name, stage="inductor", reason="Series validation failed")
                    else:
                        ports["Port_all_Series"] = {
                            "type": "series",
                            "signs": signs,
                            "raw_indices": ",".join(str(i) for i in range(total)),
                        }

            if enable_tx:
                tx_ports = self._build_transformer_ports(layers, primary_layers, secondary_layers, phase_count)
                if tx_ports is None:
                    append_line(na_tx, path.name)
                    PG.append_debug_entry(debug_log, spiral=path.name, stage="transformer", reason="Transformer port validation failed")
                else:
                    ports.update(tx_ports)

            if not ports:
                continue

            dirs = PG.ensure_analysis_dirs(path)
            system_type = "hybrid" if enable_inductor and enable_tx else ("transformer" if enable_tx else "inductor")
            dirs["ports_config"].write_text(json.dumps({"ports": ports, "system_type": system_type}, indent=2))
            PG.process_spiral(
                path,
                records,
                ports_override=ports,
                auto_reuse_ports=False,
                debug_log_path=debug_log,
            )

        if records:
            PG.write_global_summary(self.address_file.parent, records)
            self.log.insert("end", "Plot generation complete. Global_Report updated.\n")
        else:
            self.log.insert("end", "Plot generation finished. No analyzable folders found.\n")
        self.log.see("end")
        self.destroy()


# ---------------- Main app -----------------

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spirals main panel")
        self.geometry("940x720")

        self.var_address = tk.StringVar()
        self.var_eps = tk.StringVar(value="3.5")
        self.var_matrix_json = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        top = ttk.LabelFrame(self, text="1) Geometry generation")
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Use the existing batch UI to generate spirals and Address.txt").pack(side="left", padx=6)
        ttk.Button(top, text="Open generator", command=self._launch_spiral_ui).pack(side="right", padx=6)

        mid = ttk.LabelFrame(self, text="2) Address & solver setup")
        mid.pack(fill="x", padx=10, pady=8)

        row = ttk.Frame(mid); row.pack(fill="x", pady=4, padx=6)
        ttk.Label(row, text="Address.txt:").pack(side="left")
        ttk.Entry(row, textvariable=self.var_address, width=80).pack(side="left", padx=6)
        ttk.Button(row, text="Browse…", command=self._browse_address).pack(side="left")
        ttk.Button(row, text="Verify", command=self._verify_address).pack(side="left", padx=4)

        eps_row = ttk.Frame(mid); eps_row.pack(fill="x", pady=4, padx=6)
        ttk.Label(eps_row, text="Permittivity (eps_r):").pack(side="left")
        ttk.Entry(eps_row, textvariable=self.var_eps, width=12).pack(side="left", padx=6)

        solver = ttk.LabelFrame(self, text="3) Solve")
        solver.pack(fill="x", padx=10, pady=8)
        ttk.Button(solver, text="Run conversion + solvers", command=self._run_pipeline).pack(side="left", padx=6, pady=6)
        ttk.Button(solver, text="Configure ports / plots", command=self._open_ports_popup).pack(side="left", padx=6)

        viewer = ttk.LabelFrame(self, text="4) Matrix review")
        viewer.pack(fill="x", padx=10, pady=8)
        row_json = ttk.Frame(viewer); row_json.pack(fill="x", pady=4, padx=6)
        ttk.Label(row_json, text="Matrix JSON:").pack(side="left")
        ttk.Entry(row_json, textvariable=self.var_matrix_json, width=70).pack(side="left", padx=6)
        ttk.Button(row_json, text="Browse…", command=self._browse_matrix_json).pack(side="left")
        ttk.Button(viewer, text="Export readable CSV/Excel", command=self._export_matrix_json).pack(side="left", padx=6, pady=2)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = tk.Text(log_frame, wrap="word")
        self.log.pack(fill="both", expand=True)

    def _launch_spiral_ui(self):
        if not SPIRAL_UI.exists():
            messagebox.showerror("Missing script", f"Cannot find {SPIRAL_UI}")
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, str(SPIRAL_UI)],
                cwd=str(SPIRAL_UI.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Launch failed", str(exc))
            return

        self.log.insert("end", f"Launched spiral generator UI (pid {proc.pid}).\n")
        self.log.see("end")

        def _check_proc():
            ret = proc.poll()
            if ret is None:
                return
            out, err = proc.communicate()
            if ret != 0:
                messagebox.showerror(
                    "Generator exited", err or out or f"Exited with status {ret}", parent=self
                )
                self.log.insert("end", err or out or f"Generator exited with {ret}\n")
            elif out or err:
                self.log.insert("end", (out or "") + (err or ""))
            self.log.see("end")

        # Surface immediate failures instead of silently ignoring them
        self.after(1200, _check_proc)

    def _browse_address(self):
        path = filedialog.askopenfilename(title="Select Address.txt", filetypes=[("Address", "Address.txt"), ("Text", "*.txt")])
        if path:
            self.var_address.set(path)

    def _browse_matrix_json(self):
        path = filedialog.askopenfilename(title="Select matrix JSON", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.var_matrix_json.set(path)

    def _verify_address(self):
        path = Path(self.var_address.get())
        if not path.is_file():
            messagebox.showerror("Address missing", "Select a valid Address.txt first.")
            return False
        try:
            entries = read_address_entries(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid Address.txt", str(exc))
            return False
        missing = [p for p in entries if not p.exists()]
        if missing:
            messagebox.showwarning("Missing folders", "\n".join(str(m) for m in missing))
            return False
        messagebox.showinfo("Address check", f"{len(entries)} folders found.")
        return True

    def _run_pipeline(self):
        if not self._verify_address():
            return
        addr = Path(self.var_address.get())
        eps = self.var_eps.get().strip() or "1"
        # 1) Convert Wire_Sections to solver formats
        ok = log_subprocess([sys.executable, str(FAST_UI), "--non-interactive", str(addr)], self.log)
        if not ok:
            return
        # 2) Run solvers
        ok = log_subprocess([sys.executable, str(AUTOMATE), str(addr), eps], self.log)
        if ok:
            messagebox.showinfo("Solvers complete", "FastHenry/FasterCap runs finished.")
            self._open_ports_popup()

    def _open_ports_popup(self):
        if not self.var_address.get():
            messagebox.showwarning("Address needed", "Select Address.txt first.")
            return
        popup = PortsPopup(self, Path(self.var_address.get()), self.log)
        popup.wait_window()

    def _export_matrix_json(self):
        raw = self.var_matrix_json.get().strip()
        if not raw:
            messagebox.showwarning("Select JSON", "Choose a matrix JSON file first.")
            return
        json_path = Path(raw)
        if not json_path.exists():
            messagebox.showerror("File missing", f"Cannot find {json_path}")
            return
        try:
            output = PG.export_matrix_json_to_excel(json_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))
            return
        self.log.insert("end", f"Readable workbook created: {output}\n")
        self.log.see("end")
        messagebox.showinfo("Export complete", f"Readable workbook saved to\n{output}")


def main():
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
