import math

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def get_float(prompt, default):
    """Read a float from input, with a default value."""
    txt = input(f"{prompt} [{default}]: ").strip()
    return float(txt) if txt else float(default)


def get_int(prompt, default):
    """Read an int from input, with a default value."""
    txt = input(f"{prompt} [{default}]: ").strip()
    return int(txt) if txt else int(default)


def compute_for_n_series(
    N_series_units,
    S_tot_MVA,
    Vp_LL_kV,
    Vs_LL_kV,
    N_units_total,
    N_systems_sec,
    N_PCB_per_unit,
    I_max_PCB_Trace,
):
    """
    Compute all electrical quantities for a given number of units in series.
    Returns a dict with all relevant results.
    """

    # Basic conversions
    Vp_LL = Vp_LL_kV * 1e3  # [V]
    Vs_LL = Vs_LL_kV * 1e3  # [V]

    # Power per system
    S_sys_MVA = S_tot_MVA / N_systems_sec
    S_sys_VA = S_sys_MVA * 1e6

    # System currents (primary & secondary)
    I_p_sys = S_sys_VA / (math.sqrt(3) * Vp_LL)
    I_s_sys = S_sys_VA / (math.sqrt(3) * Vs_LL)

    # System L-N voltages
    Vp_LN_sys = Vp_LL / math.sqrt(3.0)
    Vs_LN_sys = Vs_LL / math.sqrt(3.0)

    # Units distribution
    units_per_system = N_units_total / N_systems_sec
    N_parallel = int(units_per_system // N_series_units)
    N_used = N_systems_sec * N_series_units * N_parallel
    N_spare = N_units_total - N_used

    # Unit voltages
    Vp_unit_LL = Vp_LL / N_series_units
    Vs_unit_LL = Vs_LL / N_series_units
    Vp_unit_LN = Vp_unit_LL / math.sqrt(3.0)
    Vs_unit_LN = Vs_unit_LL / math.sqrt(3.0)

    if N_parallel > 0:
        # Currents per string / per unit
        I_s_string = I_s_sys / N_parallel
        I_p_string = I_p_sys / N_parallel
        I_s_unit = I_s_string
        I_p_unit = I_p_string

        # Apparent power per unit (secondary side)
        S_unit_VA = math.sqrt(3) * Vs_unit_LL * I_s_unit
        S_unit_kVA = S_unit_VA / 1e3

        # Consistency check
        S_sys_check_MVA = N_series_units * N_parallel * S_unit_VA / 1e6
        S_tot_check_MVA = N_systems_sec * S_sys_check_MVA
    else:
        I_s_string = I_p_string = I_s_unit = I_p_unit = 0.0
        S_unit_VA = S_unit_kVA = 0.0
        S_sys_check_MVA = S_tot_check_MVA = 0.0

    # Per-PCB currents & number of traces
    if N_parallel > 0 and N_PCB_per_unit > 0:
        I_p_PCB = I_p_unit / N_PCB_per_unit
        I_s_PCB = I_s_unit / N_PCB_per_unit
        if I_max_PCB_Trace > 0:
            N_s_Trace = round(I_s_PCB / I_max_PCB_Trace)
            N_p_Trace = round(I_p_PCB / I_max_PCB_Trace)
        else:
            N_s_Trace = N_p_Trace = 0
    else:
        I_p_PCB = I_s_PCB = 0.0
        N_s_Trace = N_p_Trace = 0

    return {
        "N_series_units": N_series_units,
        "S_sys_MVA": S_sys_MVA,
        "S_sys_VA": S_sys_VA,
        "Vp_LL": Vp_LL,
        "Vs_LL": Vs_LL,
        "Vp_LN_sys": Vp_LN_sys,
        "Vs_LN_sys": Vs_LN_sys,
        "I_p_sys": I_p_sys,
        "I_s_sys": I_s_sys,
        "units_per_system": units_per_system,
        "N_parallel": N_parallel,
        "N_used": N_used,
        "N_spare": N_spare,
        "Vp_unit_LL": Vp_unit_LL,
        "Vs_unit_LL": Vs_unit_LL,
        "Vp_unit_LN": Vp_unit_LN,
        "Vs_unit_LN": Vs_unit_LN,
        "I_p_unit": I_p_unit,
        "I_s_unit": I_s_unit,
        "S_unit_kVA": S_unit_kVA,
        "S_sys_check_MVA": S_sys_check_MVA,
        "S_tot_check_MVA": S_tot_check_MVA,
        "I_p_PCB": I_p_PCB,
        "I_s_PCB": I_s_PCB,
        "N_p_Trace": N_p_Trace,
        "N_s_Trace": N_s_Trace,
    }


def main():
    print("=== Modular Unit System Calculator ===\n")
    print("All voltages are line-to-line unless otherwise stated.\n")

    # ---- User inputs ----
    S_tot_MVA = get_float("Total apparent power S_tot in MVA", 300.0)
    Vp_LL_kV = get_float("Primary system voltage (line-line) in kV", 400.0)
    Vs_LL_kV = get_float("Secondary system voltage (line-line) in kV", 10.0)
    N_units_total = get_int("Total number of units", 190000)
    N_systems_sec = get_int("Number of secondary systems (e.g. 18 for 10 kV case)", 18)
    N_series_units = get_int("Number of units in series per string", 2000)
    N_PCB_per_unit = get_int("Number of PCB boards per unit", 25)
    I_max_PCB_Trace = get_float(
        "Continuous current allowed in a single trace in A", 0.6
    )

    # Base-case calculation
    base = compute_for_n_series(
        N_series_units,
        S_tot_MVA,
        Vp_LL_kV,
        Vs_LL_kV,
        N_units_total,
        N_systems_sec,
        N_PCB_per_unit,
        I_max_PCB_Trace,
    )

    # ---- OUTPUT: base case ----
    print("\n=== INPUT SUMMARY ===")
    print(f"S_tot            = {S_tot_MVA:.3f} MVA")
    print(f"Primary voltage  = {Vp_LL_kV:.3f} kV (line-line)")
    print(f"Secondary voltage= {Vs_LL_kV:.3f} kV (line-line)")
    print(f"Total units      = {N_units_total}")
    print(f"Secondary systems= {N_systems_sec}")
    print(f"Units in series  = {N_series_units}")
    print(f"Units per system ≈ {base['units_per_system']:.2f}")

    print("\n=== SYSTEM LEVEL ===")
    print(f"Power per system           = {base['S_sys_MVA']:.3f} MVA")
    print(f"Primary current per system = {base['I_p_sys']:.3f} A")
    print(f"Secondary current per system = {base['I_s_sys']:.3f} A")
    print(f"Primary voltage L-N (system)   = {base['Vp_LN_sys']:.1f} V")
    print(f"Secondary voltage L-N (system) = {base['Vs_LN_sys']:.1f} V")

    print("\n=== UNIT TOPOLOGY ===")
    print(f"Parallel strings per system = {base['N_parallel']}")
    print(f"Units actually used         = {base['N_used']}")
    print(f"Spare units                 = {base['N_spare']}")

    print("\n=== PER-UNIT ELECTRICAL DATA ===")
    print(
        f"Primary voltage per unit:   {base['Vp_unit_LL']:.3f} V (L-L), "
        f"{base['Vp_unit_LN']:.3f} V (L-N)"
    )
    print(
        f"Secondary voltage per unit: {base['Vs_unit_LL']:.3f} V (L-L), "
        f"{base['Vs_unit_LN']:.3f} V (L-N)"
    )
    print(f"Primary current per unit:   {base['I_p_unit']:.3f} A")
    print(f"Secondary current per unit: {base['I_s_unit']:.3f} A")
    print(f"Apparent power per unit:    {base['S_unit_kVA']:.3f} kVA")

    print("\n=== PER-PCB ELECTRICAL DATA ===")
    print(f"Primary current per PCB:   {base['I_p_PCB']:.3f} A")
    print(f"Primary num trace per PCB per phase:   {base['N_p_Trace']}")
    print(f"Secondary current per PCB: {base['I_s_PCB']:.3f} A")
    print(f"Secondary trace per PCB per phase: {base['N_s_Trace']} ")

    if base["N_parallel"] > 0:
        print("\n=== CONSISTENCY CHECK ===")
        print(
            f"Power per system from units = {base['S_sys_check_MVA']:.3f} MVA"
        )
        print(
            f"Total power from all units  = {base['S_tot_check_MVA']:.3f} MVA"
        )

    # ---- SWEEP over N_series_units ----
    print("\n=== SWEEP SETTINGS (Number of units in series per string) ===")
    sweep_min = get_int(
        "Sweep: minimum units in series (0 = skip sweep)", 0
    )
    sweep_max = get_int(
        "Sweep: maximum units in series (ignored if min <= 0)", 0
    )

    if sweep_min > 0 and sweep_max >= sweep_min:
        if plt is None:
            print(
                "\nmatplotlib is not installed, so plots cannot be generated."
                "\nInstall it with 'pip install matplotlib' and run again "
                "if you want the sweep plots."
            )
        else:
            n_vals = []
            vp_unit_vals = []
            is_unit_vals = []

            for n_series in range(sweep_min, sweep_max + 1):
                res = compute_for_n_series(
                    n_series,
                    S_tot_MVA,
                    Vp_LL_kV,
                    Vs_LL_kV,
                    N_units_total,
                    N_systems_sec,
                    N_PCB_per_unit,
                    I_max_PCB_Trace,
                )
                # Only include cases where at least 1 string exists
                if res["N_parallel"] > 0:
                    n_vals.append(n_series)
                    vp_unit_vals.append(res["Vp_unit_LL"])
                    is_unit_vals.append(res["I_s_unit"])

            if n_vals:
                # Plot: Primary voltage per unit vs N_series
                plt.figure()
                plt.plot(n_vals, vp_unit_vals, marker="o")
                plt.xlabel("Number of units in series per string")
                plt.ylabel("Primary voltage per unit [V line-line]")
                plt.grid(True)
                plt.tight_layout()

                # Plot: Secondary current per unit vs N_series
                plt.figure()
                plt.plot(n_vals, is_unit_vals, marker="o")
                plt.xlabel("Number of units in series per string")
                plt.ylabel("Secondary current per unit [A]")
                plt.grid(True)
                plt.tight_layout()

                plt.show()
            else:
                print(
                    "\nNo valid sweep points (probably too few units per system "
                    "for the chosen N_series range)."
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
#this is a test