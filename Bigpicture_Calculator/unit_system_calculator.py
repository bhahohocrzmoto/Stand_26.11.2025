import math

def get_float(prompt, default):
    """Read a float from input, with a default value."""
    txt = input(f"{prompt} [{default}]: ").strip()
    return float(txt) if txt else float(default)

def get_int(prompt, default):
    """Read an int from input, with a default value."""
    txt = input(f"{prompt} [{default}]: ").strip()
    return int(txt) if txt else int(default)

def main():
    print("=== Modular Unit System Calculator ===\n")
    print("All voltages are line-to-line unless otherwise stated.\n")

    # ---- User inputs (with your typical values as defaults) ----
    S_tot_MVA      = get_float("Total apparent power S_tot in MVA", 300.0)
    Vp_LL_kV       = get_float("Primary system voltage (line-line) in kV", 400.0)
    Vs_LL_kV       = get_float("Secondary system voltage (line-line) in kV", 10.0)
    N_units_total  = get_int  ("Total number of units", 190000)
    N_systems_sec  = get_int  ("Number of secondary systems (e.g. 18 for 10 kV case)", 18)
    N_series_units = get_int  ("Number of units in series per string", 2000)
    N_PCB_per_unit = get_int  ("Number of PCB boards per unit", 25)
    I_max_PCB_Trace = get_float("Continous current allowed in a single trace in A", 0.6)
    u_t = Vp_LL_kV/Vs_LL_kV

    # ---- Basic conversions ----
    Vp_LL = Vp_LL_kV * 1e3  # [V]
    Vs_LL = Vs_LL_kV * 1e3  # [V]

    # ---- Power per system ----
    S_sys_MVA = S_tot_MVA / N_systems_sec
    S_sys_VA  = S_sys_MVA * 1e6

    # ---- System currents (primary & secondary) ----
    I_p_sys = S_sys_VA / (math.sqrt(3) * Vp_LL)  # [A]
    I_s_sys = S_sys_VA / (math.sqrt(3) * Vs_LL)  # [A]

    # ---- System line-to-neutral voltages ----
    Vp_LN_sys = Vp_LL / math.sqrt(3.0)
    Vs_LN_sys = Vs_LL / math.sqrt(3.0)

    # ---- Units distribution: series & parallel ----
    units_per_system = N_units_total / N_systems_sec
    N_parallel = int(units_per_system // N_series_units)
    N_used     = N_systems_sec * N_series_units * N_parallel
    N_spare    = N_units_total - N_used

    # ---- Unit voltages (depend on N_series_units) ----
    Vp_unit_LL = Vp_LL / N_series_units
    Vs_unit_LL = Vs_LL / N_series_units
    Vp_unit_LN = Vp_unit_LL / math.sqrt(3.0)
    Vs_unit_LN = Vs_unit_LL / math.sqrt(3.0)

    # ---- Currents per string and per unit ----
    if N_parallel > 0:
        I_s_string = I_s_sys / N_parallel  # same as unit current in series string
        I_p_string = I_p_sys / N_parallel
        I_s_unit   = I_s_string
        I_p_unit   = I_p_string

        # Apparent power per unit (use secondary side)
        S_unit_VA  = math.sqrt(3) * Vs_unit_LL * I_s_unit
        S_unit_kVA = S_unit_VA / 1e3

        # Consistency check (what MVA do we get from this topology?)
        S_sys_check = N_series_units * N_parallel * S_unit_VA / 1e6
        S_tot_check = N_systems_sec * S_sys_check
    else:
        I_s_string = I_p_string = I_s_unit = I_p_unit = 0.0
        S_unit_VA = S_unit_kVA = 0.0
        S_sys_check = S_tot_check = 0.0
        
    I_p_PCB = I_p_unit/N_PCB_per_unit
    I_s_PCB = I_s_unit/N_PCB_per_unit
    N_s_Trace = round(I_s_PCB/I_max_PCB_Trace)
    N_p_Trace = round(I_p_PCB/I_max_PCB_Trace)

    # ---- Output ----
    print("\n=== INPUT SUMMARY ===")
    print(f"S_tot            = {S_tot_MVA:.3f} MVA")
    print(f"Primary voltage  = {Vp_LL_kV:.3f} kV (line-line)")
    print(f"Secondary voltage= {Vs_LL_kV:.3f} kV (line-line)")
    print(f"Total units      = {N_units_total}")
    print(f"Secondary systems= {N_systems_sec}")
    print(f"Units in series  = {N_series_units}")
    print(f"Units per system ≈ {units_per_system:.2f}")

    print("\n=== SYSTEM LEVEL ===")
    print(f"Power per system           = {S_sys_MVA:.3f} MVA")
    print(f"Primary current per system = {I_p_sys:.3f} A")
    print(f"Secondary current per system = {I_s_sys:.3f} A")
    print(f"Primary voltage L-N (system)   = {Vp_LN_sys:.1f} V")
    print(f"Secondary voltage L-N (system) = {Vs_LN_sys:.1f} V")

    print("\n=== UNIT TOPOLOGY ===")
    print(f"Parallel strings per system = {N_parallel}")
    print(f"Units actually used         = {N_used}")
    print(f"Spare units                 = {N_spare}")

    print("\n=== PER-UNIT ELECTRICAL DATA ===")
    print(f"Primary voltage per unit:   {Vp_unit_LL:.3f} V (L-L), {Vp_unit_LN:.3f} V (L-N)")
    print(f"Secondary voltage per unit: {Vs_unit_LL:.3f} V (L-L), {Vs_unit_LN:.3f} V (L-N)")
    print(f"Primary current per unit:   {I_p_unit:.3f} A")
    print(f"Secondary current per unit: {I_s_unit:.3f} A")
    print(f"Apparent power per unit:    {S_unit_kVA:.3f} kVA")
    
    print("\n=== PER-PCB ELECTRICAL DATA ===")
    print(f"Primary current per PCB:   {I_p_PCB:.3f} A")
    print(f"Primary num trace per PCB per phase:   {N_p_Trace}")
    print(f"Secondary current per PCB: {I_s_PCB:.3f} A")
    print(f"Secondary trace per PCB per phase: {N_s_Trace} ")

    if N_parallel > 0:
        print("\n=== CONSISTENCY CHECK ===")
        print(f"Power per system from units = {S_sys_check:.3f} MVA")
        print(f"Total power from all units  = {S_tot_check:.3f} MVA")

    print("\nDone.")

if __name__ == "__main__":
    main()
