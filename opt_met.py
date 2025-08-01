# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:23:59 2025
Functions for data processing ratios between optical clocks

@author: mar
"""
import numpy as np
import os
import datetime
from timet import data_path, sec2mjd
from mpmath import mp, mpf
mp.dps = 25

def ratio_format(ratio: mpf, uncertainty: float) -> str:
    """
    Format a frequency ratio with uncertainty using 18 decimal places and parenthesis notation.

    Parameters:
        ratio (mpf): The ratio value.
        uncertainty (float): The uncertainty in the same units.

    Returns:
        str: Formatted ratio string like '0.999999999999999990(4)'
    """
    scaled_ratio = ratio * mpf(1e18)
    scaled_unc = uncertainty * 1e18

    int_ratio = int(scaled_ratio)
    int_unc = int(round(scaled_unc))

    formatted = f"{int_ratio}({int_unc})"
    if ratio > 1:
        formatted = formatted[0] + "." + formatted[1:]
    else:
        formatted = "0." + formatted

    return formatted


def ratio_dat(data_dict, paths, ky, output_folder='ratio_data/'):
    """
    Computes the product of all intermediate ratios for a given path.

    Parameters:
        data_dict (dict): dictionary with all campaign data.
        paths (dict): dictionary with information about paths
        ky (str): name of the comparison to make (N_ini-N_end)

    Returns:
        np.ndarray: Array of R (float) values computed from the input arrays.
    """
    # Generate data arrays of the link
    arrays_dict = data_path(data_dict, paths[ky]['folders'])
    
    # Obtain constants from the path
    folders = paths[ky]['folders']          # Folders of the path> OSCA -> OSCB 
    directions = paths[ky]['directions']    # Directions of the folders
    constants = paths[ky]['r_constants']    # constants R_ij
    
    # Constants related to the ratio
    nuA = paths[ky]['nuA']
    R0 = paths[ky]['nuB']/paths[ky]['nuA']
    grsA = paths[ky]['grsA']
    grsB = paths[ky]['grsB']
    uA_opt = paths[ky]['uA_sys']
    uB_opt = paths[ky]['uB_sys']
    

    # Uncertainty arrays for the ratio data
    if arrays_dict[folders[0]].shape[1] >= 4:
        usysA = arrays_dict[folders[0]][:, 3]
    else:
        usysA = np.ones(arrays_dict[folders[0]].shape[0], dtype=np.float64) * uA_opt
    
    if arrays_dict[folders[-1]].shape[1] >= 4:
        usysB = arrays_dict[folders[-1]][:, 3]
    else:
        usysB = np.ones(arrays_dict[folders[0]].shape[0], dtype=np.float64) * uB_opt
    
    #Array with MJD
    t_total = arrays_dict[folders[0]][:,0]
    mjd_tot = sec2mjd(t_total)
    
    # Initialize total ratio as mpf zeros
    ratio_sum = [mpf('0') for _ in range(arrays_dict[folders[0]].shape[0])]
    
    # Loop over folders and accumulate
    for i, folder in enumerate(folders):
        direction = mpf(directions[i])
        arr = arrays_dict[folder]
    
        for j, f in enumerate(arr[:, 1]):
            ratio_sum[j] += mpf(f) * direction * mpf(constants['s'][i]) * mpf(constants['R_constants'][i]) / mpf(nuA)
    
    # Apply the final scaling and correction
    rho_tot = mpf(constants['rho_tot'])
    R0 = mpf(R0)
    grsA = mpf(grsA)
    grsB = mpf(grsB)
    
    # Final ratio calculation
    R = [((r + 1) * rho_tot / R0 - 1) - grsA + grsB for r in ratio_sum]
    R = np.array([float(r) for r in R])
    # Package output (you may want to keep R as a list of mpf or convert to float depending on needs)
    ratio_data = {
        'mjd': mjd_tot,
        'ratio': R,
        'usysA': usysA,
        'usysB': usysB,
        'r0': R0
    }
    
    save_ratio_by_day(output_folder, ratio_data, usysA, usysB, ky[0], ky[1])
    
    return ratio_data

def save_ratio_by_day(output_folder, ratio_data, usysA, usysB, source_osc, target_osc):
    """
    Saves ratio data to separate .dat files per MJD day, with comments and headers.

    Parameters:
        output_folder (str): Folder where the .dat files will be saved.
        ratio_data (dict): Output from ratio_dat() with keys 'mjd', 'ratio', 'r0'.
        usysA (np.ndarray): Array of system uncertainty A values.
        usysB (np.ndarray): Array of system uncertainty B values.
        source_osc (str): Oscillator A name.
        target_osc (str): Oscillator B name.
    """
    output_folder = output_folder+f"/{source_osc}-{target_osc}"
    os.makedirs(output_folder, exist_ok=True)

    mjd = ratio_data['mjd']
    R = ratio_data['ratio']

    # Convert MJD to int day numbers
    mjd_days = np.floor(mjd).astype(int)
    unique_days = np.unique(mjd_days)

    today_str = datetime.date.today().isoformat()
    link_name = f"{source_osc}-{target_osc}".replace(" ", "_")

    for day in unique_days:
        mask = mjd_days == day

        mjd_day = mjd[mask]
        R_day = R[mask]
        usysA_day = usysA[mask]
        usysB_day = usysB[mask]
        flags = np.ones_like(mjd_day, dtype=int)  # All valid if passed here

        # Convert MJD to YYYY-MM-DD for filename
        date_str = (datetime.date(1858, 11, 17) + datetime.timedelta(days=int(day))).isoformat()
        filename = f"{date_str}_{link_name}.dat"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, 'w') as f:
            f.write(f"# Data for {source_osc}-{target_osc}\n")
            f.write(f"# File generated on: {today_str}\n")
            f.write("#\n")
            f.write("# t\tDelta[A->B]\tflag\tusys_A\tusys_B\n")

            for t, r, fl, ua, ub in zip(mjd_day, R_day, flags, usysA_day, usysB_day):
                f.write(f"{t:.12f}\t{r:.12e}\t{fl}\t{ua:.3e}\t{ub:.3e}\n")

    print(f"Saved: ratio {source_osc}-{target_osc}")

    
    
    
    
    
    
    
