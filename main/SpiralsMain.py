#!/usr/bin/env python3
"""
Central orchestration GUI for spiral generation, solver automation, and plotting.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parents[1]
SPIRAL_UI = REPO_ROOT / "SpiralGeometryGeneration" / "Spiral_Batch_Variants_UI_16.11.2025.py"
AUTOMATE = REPO_ROOT / "FastSolver" / "Automation" / "automate_solvers.py"
PLOT_GEN = REPO_ROOT / "FastSolver" / "PlotGeneration" / "PlotGeneration.py"
ANALYSIS_SCRIPT = REPO_ROOT / "BatchAnalysis" / "design_analyzer.py"
INDUCTOR_ANALYSIS_SCRIPT = REPO_ROOT / "BatchAnalysis" / "inductor_analyzer.py"

sys.path.insert(0, str(REPO_ROOT))
from FastSolver.PlotGeneration import PlotGeneration as PG

PHASE_LETTERS = ("A", "B", "C")

def read_address_entries(address_file: Path) -> List[Path]:
    """Reads and resolves paths from an Address.txt file."""
    cleaned = address_file.read_text().splitlines()
    entries: List[Path] = []
    for line in cleaned:
        stripped = line.strip().strip('"').strip("'")
        if not stripped: continue
        p = Path(stripped)
        if not p.is_absolute():
            p = address_file.parent / p
        entries.append(p.resolve())
    return entries

def parse_spiral_folder_name(name: str) -> List[Dict[str, object]]:
    """Extract layer metadata (layer index, K, direction) from a folder name."""
    matches = list(re.finditer(r"L(?P<layer>\d+)_K(?P<K>\d+)_N[^_]+_(?P<dir>CW|CCW)", name))
    info: List[Dict[str, object]] = []
    offset = 0
    for m in matches:
        layer_idx, k, direction = int(m.group("layer")), int(m.group("K")), m.group("dir")
        info.append({"layer": layer_idx, "K": k, "direction": direction, "start": offset})
        offset += k
    return info

def build_sign_vector(active_indices: Sequence[int], total: int) -> List[float]:
    """Build a raw +1 sign vector for the selected conductors."""
    signs = [0.0] * total
    for idx in active_indices:
        if 0 <= idx < total:
            signs[idx] = 1.0
    return signs

def log_subprocess(cmd: List[str], log_widget: tk.Text) -> bool:
    """Runs a command and logs its output to a text widget."""
    log_widget.insert("end", f"\n$ {' '.join(cmd)}\n")
    log_widget.see("end")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        if proc.stdout: log_widget.insert("end", proc.stdout)
        if proc.stderr: log_widget.insert("end", proc.stderr)
        log_widget.see("end")
        return True
    except subprocess.CalledProcessError as exc:
        log_widget.insert("end", exc.stdout or "")
        log_widget.insert("end", exc.stderr or "")
        log_widget.insert("end", f"Command failed: {exc}\n")
        log_widget.see("end")
        messagebox.showerror("Command failed", f"{cmd[0]} exited with status {exc.returncode}")
        return False

class PortsPopup(tk.Toplevel):
    """A popup window for configuring port analyses."""
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
        except Exception as exc:
            messagebox.showerror("Address read error", str(exc), parent=self)
            return []
        existing = [p for p in paths if p.exists()]
        if not existing:
            messagebox.showwarning("No folders", "No valid folders found in Address.txt.", parent=self)
        return existing

    def _build_ui(self):
        # UI layout remains the same as before...
        left = ttk.Frame(self); left.pack(side="left", fill="both", expand=False, padx=8, pady=8)
        ttk.Label(left, text="Spiral variations (from Address.txt)").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("name", "conductors"), show="headings", height=20)
        self.tree.heading("name", text="Folder"), self.tree.heading("conductors", text="# conductors")
        self.tree.column("name", width=420), self.tree.column("conductors", width=100, anchor="center")
        self.tree.pack(fill="both", expand=True)
        for path in self.spiral_paths:
            self.tree.insert("", "end", iid=str(path), values=(path.name, self._count_conductors(path)))

        right = ttk.Frame(self); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        
        ind_frame = ttk.LabelFrame(right, text="Inductor analysis"); ind_frame.pack(fill="x", pady=6)
        self.var_enable_inductor = tk.BooleanVar(value=True)
        ttk.Checkbutton(ind_frame, text="Enable inductor analysis", variable=self.var_enable_inductor).pack(anchor="w", padx=6, pady=2)
        series_row = ttk.Frame(ind_frame); series_row.pack(fill="x", padx=6, pady=2)
        self.var_series = tk.BooleanVar(value=True)
        ttk.Checkbutton(series_row, text="Series (Port_all_Series)", variable=self.var_series).pack(side="left")
        self.var_parallel = tk.BooleanVar(value=True)
        ttk.Checkbutton(series_row, text="Parallel (Port_all_Parallel)", variable=self.var_parallel).pack(side="left", padx=12)

        tx_frame = ttk.LabelFrame(right, text="Transformer analysis"); tx_frame.pack(fill="x", pady=6)
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
        
        map_frame = ttk.Frame(tx_frame); map_frame.pack(fill="both", padx=6, pady=4)
        ttk.Label(map_frame, text="Optional custom port mapping (format: pA:0,6 | sA:3,9)").pack(anchor="w")
        self.var_custom_ports = tk.Text(map_frame, height=4); self.var_custom_ports.pack(fill="x", expand=True)
        
        self.summary = tk.Text(right, height=10); self.summary.pack(fill="both", expand=True, pady=(6, 0))
        self._refresh_summary()

        action = ttk.Frame(self); action.pack(fill="x", side="bottom", pady=8, padx=10)
        ttk.Button(action, text="Run PlotGeneration", command=self._run_plots).pack(side="right", padx=6)
        ttk.Button(action, text="Cancel", command=self.destroy).pack(side="right")

    def _count_conductors(self, path: Path) -> int:
        cap_path = path / "FastSolver" / "CapacitanceMatrix.txt"
        if cap_path.exists():
            try: return PG.load_capacitance_matrix(cap_path).shape[0]
            except Exception: pass
        wire_sections = path / "Wire_Sections.txt"
        if wire_sections.exists():
            try: return len([ln for ln in wire_sections.read_text().splitlines() if ln.strip()])
            except Exception: return 0
        return 0

    def _refresh_summary(self):
        self.summary.delete("1.0", "end")
        self.summary.insert("end", f"Inductor analysis: {'enabled' if self.var_enable_inductor.get() else 'disabled'}\n")
        if self.var_enable_inductor.get(): self.summary.insert("end", f"  - Series: {self.var_series.get()}, Parallel: {self.var_parallel.get()}\n")
        self.summary.insert("end", f"Transformer analysis: {'enabled' if self.var_enable_tx.get() else 'disabled'}\n")
        if self.var_enable_tx.get():
            self.summary.insert("end", f"  - Primary: {self.var_primary_layers.get() or '-'}\n  - Secondary: {self.var_secondary_layers.get() or '-'}\n  - Phases: {self.var_phase_count.get()}\n")
            custom = self.var_custom_ports.get("1.0", "end").strip()
            if custom: self.summary.insert("end", f"  - Custom ports: {custom}\n")
        self.summary.see("end")

    def _parse_layer_selection(self, raw: str) -> List[int]:
        return [int(token) for token in re.split(r"[;,\s]+", raw.strip()) if token.isdigit()]

    def _get_layers_info(self, path: Path) -> List[Dict[str, object]]:
        if path not in self.layer_cache:
            self.layer_cache[path] = parse_spiral_folder_name(path.name)
        return self.layer_cache[path]

    def _validate_series(self, layers: List[Dict[str, object]]) -> Tuple[bool, List[float]]:
        if not layers or len(layers) < 2: return False, []
        if any(int(layer["K"]) != 1 for layer in layers): return False, []
        directions = [str(layer["direction"]) for layer in layers]
        if any(directions[i] == directions[i+1] for i in range(len(directions) - 1)): return False, []
        
        total_k = sum(int(layer["K"]) for layer in layers)
        signs: List[float] = [0.0] * total_k
        dir_to_sign = {"CCW": 1.0, "CW": -1.0}
        for info in layers:
            signs[int(info["start"])] = dir_to_sign.get(str(info["direction"]), 1.0)
        return True, signs

    def _parse_custom_ports(self, text: str, total: int) -> Dict[str, Dict[str, object]]:
        ports = {}
        for line in text.replace("|", "\n").splitlines():
            if ":" not in line: continue
            name, raw_indices = line.split(":", 1)
            indices = [int(val) for val in re.split(r"[Kohls,\s]+", raw_indices.strip()) if val.isdigit()]
            ports[name.strip()] = {"type": "parallel", "signs": build_sign_vector(indices, total), "raw_indices": ",".join(map(str, indices))}
        return ports

    def _build_transformer_ports(
        self, layers: List[Dict[str, object]], primary_layers: List[int], secondary_layers: List[int], phase_count: int
    ) -> Optional[Dict[str, Dict[str, object]]]:
        if not primary_layers and not secondary_layers: return None
        layer_map = {int(info["layer"]): info for info in layers}
        if any(layer not in layer_map for layer in primary_layers + secondary_layers): return None

        def validate_phase_counts(selected: List[int]) -> bool:
            return all(layer_map.get(layer) and int(layer_map[layer]["K"]) % phase_count == 0 for layer in selected)

        if not validate_phase_counts(primary_layers) or not validate_phase_counts(secondary_layers): return None

        total = sum(int(info["K"]) for info in layers)
        custom_text = self.var_custom_ports.get("1.0", "end").strip()
        if custom_text: return self._parse_custom_ports(custom_text, total)

        ports = {}
        letters = PHASE_LETTERS[:phase_count]
        def collect_indices(selected_layers: List[int], phase_idx: int) -> List[int]:
            return [i for layer in selected_layers for i in range(int(layer_map[layer]["start"]) + phase_idx * (int(layer_map[layer]["K"]) // phase_count), int(layer_map[layer]["start"]) + (phase_idx + 1) * (int(layer_map[layer]["K"]) // phase_count))]

        for idx, letter in enumerate(letters):
            if primary_layers:
                p_indices = collect_indices(primary_layers, idx)
                ports[f"p{letter}"] = {"type": "parallel", "signs": build_sign_vector(p_indices, total), "raw_indices": ",".join(map(str, p_indices))}
            if secondary_layers:
                s_indices = collect_indices(secondary_layers, idx)
                ports[f"s{letter}"] = {"type": "parallel", "signs": build_sign_vector(s_indices, total), "raw_indices": ",".join(map(str, s_indices))}
        return ports

    def _run_plots(self):
        self._refresh_summary()
        if not self.spiral_paths:
            messagebox.showwarning("No folders", "No spiral folders were loaded.", parent=self)
            return

        enable_inductor, enable_tx = self.var_enable_inductor.get(), self.var_enable_tx.get()
        if not enable_inductor and not enable_tx:
            messagebox.showwarning("Nothing to run", "Select at least one analysis mode.", parent=self)
            return

        debug_log = self.address_file.parent / PG.DEBUG_LOG_NAME
        debug_log.unlink(missing_ok=True)
        
        try: phase_count = int(self.var_phase_count.get())
        except ValueError: phase_count = 1
        primary_layers = self._parse_layer_selection(self.var_primary_layers.get())
        secondary_layers = self._parse_layer_selection(self.var_secondary_layers.get())

        records: List[Dict[str, object]] = []
        for path in self.spiral_paths:
            layers = self._get_layers_info(path)
            total = sum(int(item["K"]) for item in layers)
            if not total:
                PG.append_debug_entry(debug_log, spiral=path.name, stage="precheck", status="FAILURE", detail="No conductors found")
                continue

            ports: Dict[str, Dict[str, object]] = {}
            if enable_inductor:
                if self.var_parallel.get():
                    if (path / "FastSolver" / "Zc.mat").exists() and (path / "FastSolver" / "CapacitanceMatrix.txt").exists():
                        indices = list(range(total))
                        ports["Port_all_Parallel"] = {"type": "parallel", "signs": build_sign_vector(indices, total), "raw_indices": ",".join(str(i) for i in indices)}
                    else:
                        PG.append_debug_entry(debug_log, spiral=path.name, stage="inductor_parallel", status="FAILURE", detail="Missing Zc.mat or CapacitanceMatrix.txt")
                
                if self.var_series.get():
                    ok_series, signs = self._validate_series(layers)
                    if ok_series:
                        ports["Port_all_Series"] = {"type": "series", "signs": signs, "raw_indices": ",".join(str(i) for i in range(total))}
                    else:
                        PG.append_debug_entry(debug_log, spiral=path.name, stage="inductor_series", status="FAILURE", detail="Series validation failed (K=1 per layer and alternating directions required)")

            if enable_tx:
                tx_ports = self._build_transformer_ports(layers, primary_layers, secondary_layers, phase_count)
                if tx_ports:
                    ports.update(tx_ports)
                else:
                    PG.append_debug_entry(debug_log, spiral=path.name, stage="transformer", status="FAILURE", detail="Transformer port validation failed")
            
            if not ports:
                PG.append_debug_entry(debug_log, spiral=path.name, stage="ports", status="FAILURE", detail="No valid port configurations were generated based on UI settings.")
                continue

            dirs = PG.ensure_analysis_dirs(path)
            system_type = "hybrid" if enable_inductor and enable_tx and len(ports) > 1 else ("transformer" if enable_tx else "inductor")
            dirs["ports_config"].write_text(json.dumps({"ports": ports, "system_type": system_type}, indent=2))
            PG.process_spiral(path, records, debug_log_path=debug_log)

        if records:
            self.log.insert("end", f"Plot generation complete. Processed {len(records)} ports across all designs.\n")
        else:
            self.log.insert("end", "Plot generation finished. No analyzable folders found.\n")
        self.log.see("end")
        self.destroy()

class MainApp(tk.Tk):
    # MainApp class remains largely the same...
    def __init__(self):
        super().__init__(); self.title("Spirals main panel"); self.geometry("940x720")
        self.var_address = tk.StringVar(); self.var_eps = tk.StringVar(value="3.5"); self.var_matrix_json = tk.StringVar(); self.var_analysis_freq = tk.StringVar()
        self.var_label_mode = tk.StringVar(value="hover"); self.var_show_plot = tk.BooleanVar(value=False)
        self._build_ui()

    def _build_ui(self):
        top = ttk.LabelFrame(self, text="1) Geometry generation"); top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Use the existing batch UI to generate spirals and Address.txt").pack(side="left", padx=6)
        ttk.Button(top, text="Open generator", command=self._launch_spiral_ui).pack(side="right", padx=6)
        mid = ttk.LabelFrame(self, text="2) Address & solver setup"); mid.pack(fill="x", padx=10, pady=8)
        row = ttk.Frame(mid); row.pack(fill="x", pady=4, padx=6)
        ttk.Label(row, text="Address.txt:").pack(side="left")
        ttk.Entry(row, textvariable=self.var_address, width=80).pack(side="left", padx=6)
        ttk.Button(row, text="Browse…", command=self._browse_address).pack(side="left")
        ttk.Button(row, text="Verify", command=self._verify_address).pack(side="left", padx=4)
        eps_row = ttk.Frame(mid); eps_row.pack(fill="x", pady=4, padx=6)
        ttk.Label(eps_row, text="Permittivity (eps_r):").pack(side="left")
        ttk.Entry(eps_row, textvariable=self.var_eps, width=12).pack(side="left", padx=6)
        solver = ttk.LabelFrame(self, text="3) Solve"); solver.pack(fill="x", padx=10, pady=8)
        ttk.Button(solver, text="Run conversion + solvers", command=self._run_pipeline).pack(side="left", padx=6, pady=6)
        ttk.Button(solver, text="Configure ports / plots", command=self._open_ports_popup).pack(side="left", padx=6)
        viewer = ttk.LabelFrame(self, text="4) Matrix review"); viewer.pack(fill="x", padx=10, pady=8)
        row_json = ttk.Frame(viewer); row_json.pack(fill="x", pady=4, padx=6)
        ttk.Label(row_json, text="Matrix JSON:").pack(side="left")
        ttk.Entry(row_json, textvariable=self.var_matrix_json, width=70).pack(side="left", padx=6)
        ttk.Button(row_json, text="Browse…", command=self._browse_matrix_json).pack(side="left")
        analysis_frame = ttk.LabelFrame(self, text="5) Final Analysis"); analysis_frame.pack(fill="x", padx=10, pady=8)
        freq_row = ttk.Frame(analysis_frame); freq_row.pack(fill="x", padx=6, pady=(4, 6))
        ttk.Label(freq_row, text="Analysis Frequency (Hz):").pack(side="left")
        ttk.Entry(freq_row, textvariable=self.var_analysis_freq, width=20).pack(side="left", padx=6)
        ttk.Label(freq_row, text="(leave empty for highest)").pack(side="left")
        label_row = ttk.Frame(analysis_frame); label_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(label_row, text="Design labels:").pack(side="left")
        ttk.Combobox(label_row, values=("hover", "static", "none"), textvariable=self.var_label_mode, width=10, state="readonly").pack(side="left", padx=4)
        ttk.Checkbutton(label_row, text="Open interactive plot window", variable=self.var_show_plot).pack(side="left", padx=12)
        button_frame = ttk.Frame(analysis_frame); button_frame.pack(fill="x", side="bottom", padx=6, pady=6)
        ttk.Button(button_frame, text="Finalize Inductor Analysis", command=self._run_inductor_analysis).pack(side="left", padx=6)
        ttk.Button(button_frame, text="Finalize Transformer Analysis", command=self._run_full_analysis).pack(side="right", padx=6)
        log_frame = ttk.LabelFrame(self, text="Log"); log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = tk.Text(log_frame, wrap="word"); self.log.pack(fill="both", expand=True)

    def _launch_spiral_ui(self):
        if not SPIRAL_UI.exists(): messagebox.showerror("Missing script", f"Cannot find {SPIRAL_UI}"); return
        try:
            proc = subprocess.Popen([sys.executable, str(SPIRAL_UI)], cwd=str(SPIRAL_UI.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        except Exception as exc: messagebox.showerror("Launch failed", str(exc)); return
        self.log.insert("end", f"Launched spiral generator UI (pid {proc.pid}).\n"); self.log.see("end")
        self.after(1200, lambda: self._check_proc(proc))

    def _check_proc(self, proc):
        if proc.poll() is None: return
        out, err = proc.communicate()
        if proc.returncode != 0: messagebox.showerror("Generator exited", err or out or f"Exited with status {proc.returncode}", parent=self)
        if out or err: self.log.insert("end", (out or "") + (err or "")); self.log.see("end")

    def _browse_address(self):
        path = filedialog.askopenfilename(title="Select Address.txt", filetypes=[("Address", "Address.txt"), ("Text", "*.txt")])
        if path: self.var_address.set(path)

    def _browse_matrix_json(self):
        path = filedialog.askopenfilename(title="Select matrix JSON", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path: self.var_matrix_json.set(path)

    def _verify_address(self) -> bool:
        path = Path(self.var_address.get())
        if not path.is_file(): messagebox.showerror("Address missing", "Select a valid Address.txt first."); return False
        try: entries = read_address_entries(path)
        except Exception as exc: messagebox.showerror("Invalid Address.txt", str(exc)); return False
        missing = [p for p in entries if not p.exists()]
        if missing: messagebox.showwarning("Missing folders", "\n".join(str(m) for m in missing)); return False
        messagebox.showinfo("Address check", f"{len(entries)} folders found."); return True

    def _run_pipeline(self):
        if not self._verify_address(): return
        addr, eps = Path(self.var_address.get()), self.var_eps.get().strip() or "1"
        if log_subprocess([sys.executable, str(REPO_ROOT / "FastSolver" / "Automation" / "fast_solver_batch_ui.py"), "--non-interactive", str(addr)], self.log):
            if log_subprocess([sys.executable, str(AUTOMATE), str(addr), eps], self.log):
                messagebox.showinfo("Solvers complete", "FastHenry/FasterCap runs finished.")
                self._open_ports_popup()

    def _display_log_summary(self, log_path: Path):
        """Reads the debug log and displays a summary in the UI log."""
        if not log_path.is_file():
            return  # No log file, nothing to summarize

        self.log.insert("end", f"\n\n--- Summary of Plot Generation ---\n")
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)

            event_counter = Counter()
            for entry in logs:
                event_key = (
                    entry.get('stage', 'unknown'),
                    entry.get('status', 'unknown'),
                    entry.get('detail', 'unknown')
                )
                event_counter[event_key] += 1

            if not event_counter:
                self.log.insert("end", "Log is empty.\n")
            else:
                for (stage, status, detail), count in sorted(event_counter.items()):
                    self.log.insert("end", f"\n- Event: [{stage}] / Status: [{status}]\n")
                    self.log.insert("end", f"  Detail: {detail}\n")
                    self.log.insert("end", f"  Occurrences: {count} times\n")

        except Exception as e:
            self.log.insert("end", f"Could not read or parse summary log: {e}\n")
        
        self.log.insert("end", f"--- End of Summary ---\n")
        self.log.see("end")

    def _open_ports_popup(self):
        if not self.var_address.get(): messagebox.showwarning("Address needed", "Select Address.txt first."); return
        address_path = Path(self.var_address.get())
        popup = PortsPopup(self, address_path, self.log)
        popup.wait_window()

        # After popup closes, find and display the summary from the debug log
        debug_log_path = address_path.parent / PG.DEBUG_LOG_NAME
        self._display_log_summary(debug_log_path)

    def _run_full_analysis(self):
        addr_path = self.var_address.get(); freq = self.var_analysis_freq.get().strip()
        if not addr_path or not Path(addr_path).is_file(): messagebox.showerror("Address missing", "Select a valid Address.txt first."); return
        if not ANALYSIS_SCRIPT.exists(): messagebox.showerror("Missing script", f"Cannot find {ANALYSIS_SCRIPT}"); return
        cmd = [sys.executable, str(ANALYSIS_SCRIPT), addr_path, "--label-mode", self.var_label_mode.get()]
        if freq: cmd.extend(["--frequency", freq])
        if self.var_show_plot.get(): cmd.append("--show-plot")
        if log_subprocess(cmd, self.log): messagebox.showinfo("Analysis Complete", "KPI analysis finished. Check the 'FinalTransformerAnalysis' folder.")

    def _run_inductor_analysis(self):
        addr_path = self.var_address.get(); freq = self.var_analysis_freq.get().strip()
        if not addr_path or not Path(addr_path).is_file(): messagebox.showerror("Address missing", "Select a valid Address.txt first."); return
        if not INDUCTOR_ANALYSIS_SCRIPT.exists(): messagebox.showerror("Missing script", f"Cannot find {INDUCTOR_ANALYSIS_SCRIPT}"); return
        cmd = [sys.executable, str(INDUCTOR_ANALYSIS_SCRIPT), addr_path, "--label-mode", self.var_label_mode.get()]
        if freq: cmd.extend(["--frequency", freq])
        if self.var_show_plot.get(): cmd.append("--show-plot")
        if log_subprocess(cmd, self.log): messagebox.showinfo("Analysis Complete", "KPI analysis finished. Check the 'FinalInductorAnalysis' folder.")

def main():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()