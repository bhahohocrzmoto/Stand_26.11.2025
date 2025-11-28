import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def sum_k_from_name(name: str) -> float:
    """Return the sum of all K-values encoded in the folder name."""
    k_matches = re.findall(r"K(\d+(?:\.\d+)?)", name)
    return float(np.sum([float(k) for k in k_matches])) if k_matches else 0.0

def get_freq_index(frequencies, target_freq=None):
    """Finds the index of the target frequency, or the highest frequency if not specified."""
    if target_freq is None: return np.argmax(frequencies)
    try:
        target_freq = float(target_freq)
        return np.argmin(np.abs(frequencies - target_freq))
    except (ValueError, TypeError):
        return np.argmax(frequencies)

def analyze_inductor_json(json_path: Path, folder_name: str, target_freq: float | None) -> dict | None:
    """Analyzes a single inductor matrix JSON file and returns a KPI dictionary."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        analysis_type = data.get('analysis_type', 'unknown')
        port_name = data.get('port_names', ['unknown'])[0]
        
        port_type = 'unknown'
        if 'series' in analysis_type: port_type = 'series'
        elif 'parallel' in analysis_type: port_type = 'parallel'

        frequencies = np.array(data['frequencies_Hz'])
        L_port = np.array(data['matrices']['L_port'])
        R_port = np.array(data['matrices']['R_port'])
        C_port = np.array(data['matrices']['C_port'])

        freq_idx = get_freq_index(frequencies, target_freq)
        f_selected = frequencies[freq_idx]
        min_freq_idx = np.argmin(frequencies)
        omega = 2 * np.pi * f_selected

        L = L_port[freq_idx] if L_port.ndim == 3 else L_port
        R = R_port[freq_idx] if R_port.ndim == 3 else R_port
        C = C_port
        R_dc = R_port[min_freq_idx] if R_port.ndim == 3 else R_port

        l_eff, r_ac, c_self, r_dc_val = L[0, 0], R[0, 0], C[0, 0], R_dc[0, 0]

        q_factor = (omega * l_eff) / r_ac if r_ac > 0 else 0
        ac_dc_ratio = r_ac / r_dc_val if r_dc_val > 0 else 0
        srf_mhz = (1 / (2 * np.pi * np.sqrt(l_eff * c_self))) / 1e6 if l_eff > 0 and c_self > 0 else 0

        return {
            'folder': folder_name, 'port_name': port_name, 'port_type': port_type,
            'frequency_Hz': f_selected, 'effective_inductance_uH': l_eff * 1e6,
            'quality_factor_Q': q_factor, 'ac_dc_resistance_ratio': ac_dc_ratio,
            'self_capacitance_pF': c_self * 1e12, 'estimated_srf_MHz': srf_mhz,
            'total_k_sum': sum_k_from_name(folder_name)
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error processing {json_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run KPI analysis on inductor designs.")
    parser.add_argument("address_file", help="Path to the Address.txt file.")
    parser.add_argument("--frequency", help="Optional: Specific frequency in Hz.", default=None)
    parser.add_argument(
        "--label-mode",
        choices=["hover", "static", "none"],
        default="none",
        help=(
            "How to display design names: 'hover' shows tooltips using mplcursors, "
            "'static' writes labels on the plot, and 'none' hides labels."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open an interactive window for exploring the plot (useful with hover labels).",
    )
    args = parser.parse_args()

    address_path = Path(args.address_file)
    if not address_path.is_file():
        print(f"Error: Address file not found at {address_path}"); return

    base_dir = address_path.parent
    output_dir = base_dir / "FinalInductorAnalysis"
    series_dir = output_dir / "Series"; parallel_dir = output_dir / "Parallel"
    series_dir.mkdir(parents=True, exist_ok=True); parallel_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        folder_paths = [base_dir / line.strip() for line in address_path.read_text().splitlines() if line.strip()]
    except Exception as e:
        print(f"Error reading {address_path}: {e}"); return

    results = []
    for folder in folder_paths:
        if not folder.exists(): continue
        matrices_dir = folder / 'Analysis' / 'matrices'
        if not matrices_dir.exists(): continue
        
        for json_file in matrices_dir.glob('*_inductor_matrices.json'):
            kpis = analyze_inductor_json(json_file, folder.name, args.frequency)
            if kpis: results.append(kpis)
    
    print(f"\nFound and processed {len(results)} inductor analysis files.")

    if not results:
        print("No valid inductor data found to process. Exiting."); return

    df = pd.DataFrame(results)
    for port_type in ['series', 'parallel']:
        df_type = df[df['port_type'] == port_type]
        if df_type.empty:
            print(f"\nNo data found for '{port_type}' inductors."); continue

        target_dir = series_dir if port_type == 'series' else parallel_dir
        csv_path = target_dir / f'{port_type}_comparison.csv'
        df_type.to_csv(csv_path, index=False)
        print(f"\nSuccessfully saved '{port_type}' analysis to {csv_path}")

        plt.figure(figsize=(14, 10))
        scatter_ax = sns.scatterplot(
            data=df_type,
            x='effective_inductance_uH',
            y='quality_factor_Q',
            hue='estimated_srf_MHz',
            palette='viridis',
            size='total_k_sum',
            sizes=(50, 250),
            legend='auto'
        )
        plt.title(f"{port_type.capitalize()} Inductor Performance @ {args.frequency or 'Max'} Hz")
        plt.xlabel('Effective Inductance (uH)'), plt.ylabel('Quality Factor (Q)'), plt.grid(True)

        def _add_static_labels():
            for _, row in df_type.iterrows():
                plt.text(
                    row['effective_inductance_uH'] * 1.01,
                    row['quality_factor_Q'],
                    row['folder'],
                    fontsize=9
                )

        if args.label_mode == 'static':
            _add_static_labels()
        elif args.label_mode == 'hover':
            try:
                import mplcursors

                cursor = mplcursors.cursor(scatter_ax.collections[0], hover=True)
                cursor.connect(
                    "add", lambda sel: sel.annotation.set_text(df_type.iloc[sel.index]['folder'])
                )
            except ImportError:
                print("mplcursors not installed; falling back to static labels.")
                _add_static_labels()

        plt.tight_layout()

        plot_path = target_dir / f'{port_type}_performance_plot.png'
        plt.savefig(plot_path)
        print(f"Successfully saved '{port_type}' performance plot to {plot_path}")

    if args.show_plot:
        plt.show()

if __name__ == "__main__":
    main()
