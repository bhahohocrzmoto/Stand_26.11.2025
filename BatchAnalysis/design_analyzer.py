import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def get_freq_index(frequencies, target_freq=None):
    """Finds the index of the target frequency, or the highest frequency if not specified."""
    if target_freq is None:
        return np.argmax(frequencies)
    try:
        target_freq = float(target_freq)
        # Find the index of the frequency closest to the target
        return np.argmin(np.abs(frequencies - target_freq))
    except (ValueError, TypeError):
        # Fallback to highest frequency if conversion fails
        return np.argmax(frequencies)

def analyze_design_folder(folder_path, target_freq=None):
    """
    Analyzes a single design folder to calculate KPIs from a JSON file for a specific frequency.
    """
    json_path = os.path.join(folder_path, 'Analysis', 'matrices', 'transformer_matrices.json')

    if not os.path.exists(json_path):
        print(f"Warning: No JSON file found at {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"An error occurred while reading {json_path}: {e}")
        return None

    try:
        frequencies = np.array(data['frequencies_Hz'])
        L_port = np.array(data['matrices']['L_port'])
        R_port = np.array(data['matrices']['R_port'])
        C_port = np.array(data['matrices']['C_port'])

        # C is frequency-independent, so it's always the direct 2D matrix
        C = C_port

        # Handle frequency-dependent L and R
        if L_port.ndim == 2:
            L = L_port
            R = R_port
            f_selected = frequencies.item() if frequencies.size == 1 else frequencies[0]
            R_dc = R # Only one R matrix, so AC/DC ratio will be 1
        else: # L_port is 3D
            freq_idx = get_freq_index(frequencies, target_freq)
            f_selected = frequencies[freq_idx]
            L = L_port[freq_idx]
            R = R_port[freq_idx]
            min_freq_idx = np.argmin(frequencies)
            R_dc = R_port[min_freq_idx]

        # --- KPI Calculations (L & R) ---
        k = L[0, 3] / np.sqrt(L[0, 0] * L[3, 3]) if L[0,0] > 0 and L[3,3] > 0 else 0
        omega = 2 * np.pi * f_selected
        Q = (omega * L[0, 0]) / R[0, 0] if R[0, 0] != 0 else 0
        
        r_dc_val = R_dc[0, 0] if R_dc.ndim > 1 else R_dc[0] 
        ac_dc_ratio = R[0, 0] / r_dc_val if r_dc_val != 0 else 0

        primary_self_inductances = [L[0, 0], L[1, 1], L[2, 2]]
        symmetry_score = np.std(primary_self_inductances)

        isolation_db = 20 * np.log10(L[0, 3] / L[0, 4]) if L[0, 3] > 0 and L[0, 4] > 0 else -np.inf

        # --- KPI Calculations (C) ---
        c_self_pf = C[0, 0] * 1e12
        c_interwinding_pf = C[0, 3] * 1e12
        c_crosstalk_pf = C[0, 4] * 1e12

        # Estimated Self-Resonant Frequency (SRF)
        srf_mhz = 0
        if L[0, 0] > 0 and C[0, 0] > 0:
            srf_mhz = (1 / (2 * np.pi * np.sqrt(L[0, 0] * C[0, 0]))) / 1e6


        return {
            'folder': os.path.basename(folder_path),
            'frequency_Hz': f_selected,
            'coupling_coefficient_k': k,
            'quality_factor_Q': Q,
            'ac_dc_resistance_ratio': ac_dc_ratio,
            'symmetry_score': symmetry_score,
            'isolation_dB': isolation_db,
            'primary_self_capacitance_pF': c_self_pf,
            'inter_winding_capacitance_pF': c_interwinding_pf,
            'capacitive_crosstalk_pF': c_crosstalk_pf,
            'estimated_srf_MHz': srf_mhz
        }

    except (KeyError, IndexError) as e:
        print(f"Error: Data structure incorrect in {json_path}. Missing key or index: {e}")
        return None

def main():
    """
    Main function to run the batch analysis.
    """
    parser = argparse.ArgumentParser(description="Run KPI analysis on transformer designs.")
    parser.add_argument("address_file", help="Path to the Address.txt file.")
    parser.add_argument("--frequency", help="Optional: Specific frequency in Hz to analyze. Defaults to the highest available.", default=None)
    args = parser.parse_args()

    address_path = Path(args.address_file)
    if not address_path.is_file():
        print(f"Error: Address file not found at {address_path}")
        return

    base_dir = address_path.parent
    output_dir = base_dir / "FinalTransformerAnalysis"
    output_dir.mkdir(exist_ok=True)
    
    try:
        with open(address_path, 'r') as f:
            folder_paths = [str(base_dir / line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {address_path} not found.")
        return

    results = []
    missing_json_folders = []

    for folder in folder_paths:
        if not Path(folder).exists():
            print(f"Warning: Folder not found: {folder}")
            missing_json_folders.append(folder)
            continue
        kpis = analyze_design_folder(folder, target_freq=args.frequency)
        if kpis:
            results.append(kpis)
        else:
            missing_json_folders.append(folder)

    if not results:
        print("No valid data found to process. Exiting.")
        return

    df = pd.DataFrame(results)
    
    # Save results in the new dedicated folder
    csv_path = output_dir / 'design_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved analysis to {csv_path}")

    # Generate and save plot in the new folder
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='coupling_coefficient_k',
        y='isolation_dB',
        hue='quality_factor_Q',
        palette='viridis',
        size='quality_factor_Q',
        sizes=(50, 250),
        legend='auto'
    )
    plt.title(f"Pareto Plot: Design Comparison @ {args.frequency or 'Max'} Hz")
    plt.xlabel('Coupling Coefficient (k)')
    plt.ylabel('Isolation (dB)')
    plt.grid(True)
    
    for i, row in df.iterrows():
        plt.text(row['coupling_coefficient_k'] + 0.001, row['isolation_dB'], row['folder'], fontsize=9)

    plot_path = output_dir / 'design_pareto_plot.png'
    plt.savefig(plot_path)
    print(f"Successfully saved Pareto plot to {plot_path}")

    if missing_json_folders:
        print("\nCould not process the following folders (missing JSON or folder not found):")
        for folder in missing_json_folders:
            print(f"- {folder}")

if __name__ == "__main__":
    main()
