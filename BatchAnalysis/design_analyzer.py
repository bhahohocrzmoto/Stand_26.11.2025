import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def get_freq_index(frequencies, target_freq=None):
    """Finds the index of the target frequency, or the highest frequency if not specified."""
    if target_freq is None: return np.argmax(frequencies)
    try:
        target_freq = float(target_freq)
        return np.argmin(np.abs(frequencies - target_freq))
    except (ValueError, TypeError):
        return np.argmax(frequencies)

def analyze_design_folder(folder_path: Path, target_freq: float | None) -> dict | None:
    """Analyzes a single transformer_matrices.json file and returns a KPI dictionary."""
    json_path = folder_path / 'Analysis' / 'matrices' / 'transformer_matrices.json'
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        port_names = data.get('port_names', [])
        p_ports = [i for i, name in enumerate(port_names) if name.startswith('p')]
        s_ports = [i for i, name in enumerate(port_names) if name.startswith('s')]
        
        if not p_ports or not s_ports:
            return None # Not a valid transformer file for this analysis
        
        p_idx, s_idx = p_ports[0], s_ports[0]

        frequencies = np.array(data['frequencies_Hz'])
        L_port_all = np.array(data['matrices']['L_port'])
        R_port_all = np.array(data['matrices']['R_port'])
        C_port_all = np.array(data['matrices']['C_port'])

        freq_idx = get_freq_index(frequencies, target_freq)
        f_selected = frequencies[freq_idx]
        min_freq_idx = np.argmin(frequencies)
        omega = 2 * np.pi * f_selected

        L, R, C = L_port_all[freq_idx], R_port_all[freq_idx], C_port_all
        R_dc = R_port_all[min_freq_idx]

        k = L[p_idx, s_idx] / np.sqrt(L[p_idx, p_idx] * L[s_idx, s_idx]) if L[p_idx, p_idx] > 0 and L[s_idx, s_idx] > 0 else 0
        Q = (omega * L[p_idx, p_idx]) / R[p_idx, p_idx] if R[p_idx, p_idx] != 0 else 0
        ac_dc_ratio = R[p_idx, p_idx] / R_dc[p_idx, p_idx] if R_dc[p_idx, p_idx] != 0 else 0
        
        primary_inductances = [L[i, i] for i in p_ports]
        symmetry_score = np.std(primary_inductances) if len(primary_inductances) > 1 else 0.0

        srf_mhz = (1 / (2 * np.pi * np.sqrt(L[p_idx, p_idx] * C[p_idx, p_idx]))) if L[p_idx, p_idx] > 0 and C[p_idx, p_idx] > 0 else 0
        srf_mhz /= 1e6

        return {
            'folder': folder_path.name, 'frequency_Hz': f_selected, 'coupling_coefficient_k': k,
            'quality_factor_Q': Q, 'ac_dc_resistance_ratio': ac_dc_ratio, 'symmetry_score': symmetry_score,
            'primary_self_capacitance_pF': C[p_idx, p_idx] * 1e12,
            'inter_winding_capacitance_pF': C[p_idx, s_idx] * 1e12,
            'estimated_srf_MHz': srf_mhz
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error processing {json_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run KPI analysis on transformer designs.")
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
    output_dir = base_dir / "FinalTransformerAnalysis"
    output_dir.mkdir(exist_ok=True)
    
    try:
        folder_paths = [base_dir / line.strip() for line in address_path.read_text().splitlines() if line.strip()]
    except Exception as e:
        print(f"Error reading {address_path}: {e}"); return

    results = [kpis for folder in folder_paths if folder.exists() and (kpis := analyze_design_folder(folder, args.frequency)) is not None]
    
    print(f"\nFound and processed {len(results)} transformer-compatible JSON files out of {len(folder_paths)} total designs.")

    if not results:
        print("No valid transformer data found to process. Exiting."); return

    df = pd.DataFrame(results)
    csv_path = output_dir / 'design_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved analysis to {csv_path}")

    plt.figure(figsize=(12, 8))
    # A more relevant plot for transformers might be k vs Q
    scatter_ax = sns.scatterplot(
        data=df,
        x='coupling_coefficient_k',
        y='quality_factor_Q',
        hue='estimated_srf_MHz',
        palette='viridis',
        size='estimated_srf_MHz',
        sizes=(50, 250),
        legend='auto'
    )
    plt.title(f"Pareto Plot: Design Comparison @ {args.frequency or 'Max'} Hz")
    plt.xlabel('Coupling Coefficient (k)')
    plt.ylabel('Quality Factor (Q)')
    plt.grid(True)

    def _add_static_labels():
        for _, row in df.iterrows():
            plt.text(
                row['coupling_coefficient_k'] * 1.001,
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
                "add", lambda sel: sel.annotation.set_text(df.iloc[sel.index]['folder'])
            )
        except ImportError:
            print("mplcursors not installed; falling back to static labels.")
            _add_static_labels()

    plt.tight_layout()
    plot_path = output_dir / 'design_pareto_plot.png'
    plt.savefig(plot_path)
    print(f"Successfully saved Pareto plot to {plot_path}")

    if args.show_plot:
        plt.show()

if __name__ == "__main__":
    main()
