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
    Analyzes a single design folder to calculate inductor KPIs from a JSON file.
    """
    json_path = os.path.join(folder_path, 'Analysis', 'matrices', 'inductor_matrices.json')

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

        # C is frequency-independent
        C = C_port

        if L_port.ndim == 2: # Frequency-independent
            L = L_port
            R = R_port
            f_selected = frequencies.item() if frequencies.size == 1 else frequencies[0]
            R_dc = R
        else: # Frequency-dependent (3D)
            freq_idx = get_freq_index(frequencies, target_freq)
            f_selected = frequencies[freq_idx]
            L = L_port[freq_idx]
            R = R_port[freq_idx]
            min_freq_idx = np.argmin(frequencies)
            R_dc = R_port[min_freq_idx]
        
        omega = 2 * np.pi * f_selected
        num_ports = L.shape[0]
        kpis_list = []

        for i in range(num_ports):
            l_eff = L[i, i]
            r_ac = R[i, i]
            c_self = C[i, i]
            
            # Safely get DC resistance
            r_dc_val = R_dc[i, i] if R_dc.ndim == 2 else R_dc[0]

            q_factor = (omega * l_eff) / r_ac if r_ac > 0 else 0
            ac_dc_ratio = r_ac / r_dc_val if r_dc_val > 0 else 0
            
            srf_mhz = 0
            if l_eff > 0 and c_self > 0:
                srf_mhz = (1 / (2 * np.pi * np.sqrt(l_eff * c_self))) / 1e6
            
            kpis_list.append({
                'folder': os.path.basename(folder_path),
                'port': i + 1,
                'frequency_Hz': f_selected,
                'effective_inductance_uH': l_eff * 1e6,
                'quality_factor_Q': q_factor,
                'ac_dc_resistance_ratio': ac_dc_ratio,
                'self_capacitance_pF': c_self * 1e12,
                'estimated_srf_MHz': srf_mhz,
            })
            
        return kpis_list

    except (KeyError, IndexError) as e:
        print(f"Error: Data structure incorrect in {json_path}. Missing key or index: {e}")
        return None

def main():
    """
    Main function to run the batch analysis for inductors.
    """
    parser = argparse.ArgumentParser(description="Run KPI analysis on inductor designs.")
    parser.add_argument("address_file", help="Path to the Address.txt file.")
    parser.add_argument("--frequency", help="Optional: Specific frequency in Hz to analyze. Defaults to the highest available.", default=None)
    args = parser.parse_args()

    address_path = Path(args.address_file)
    if not address_path.is_file():
        print(f"Error: Address file not found at {address_path}")
        return

    base_dir = address_path.parent
    output_dir = base_dir / "FinalInductorAnalysis"
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
            results.extend(kpis)
        else:
            missing_json_folders.append(folder)

    if not results:
        print("No valid data found to process. Exiting.")
        return

    df = pd.DataFrame(results)
    
    csv_path = output_dir / 'inductor_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved analysis to {csv_path}")

    # Generate and save plot
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df,
        x='effective_inductance_uH',
        y='quality_factor_Q',
        hue='estimated_srf_MHz',
        palette='viridis',
        size='estimated_srf_MHz',
        sizes=(50, 250),
        legend='auto'
    )
    plt.title(f"Inductor Performance Comparison @ {args.frequency or 'Max'} Hz")
    plt.xlabel('Effective Inductance (uH)')
    plt.ylabel('Quality Factor (Q)')
    plt.grid(True)
    
    for i, row in df.iterrows():
        label = f"{row['folder']} (P{row['port']})"
        plt.text(row['effective_inductance_uH'] * 1.01, row['quality_factor_Q'], label, fontsize=9)

    plot_path = output_dir / 'inductor_performance_plot.png'
    plt.savefig(plot_path)
    print(f"Successfully saved performance plot to {plot_path}")

    if missing_json_folders:
        print("\nCould not process the following folders (missing JSON or folder not found):")
        for folder in missing_json_folders:
            print(f"- {folder}")

if __name__ == "__main__":
    main()
