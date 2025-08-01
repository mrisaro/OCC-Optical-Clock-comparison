
"""
Created on Mon Jul 14 14:00:51 2025
TiMet -> library with functions to deal about timing issues like:
    time conversion, time matching, time stamp, etc
@author: mar
"""

import numpy as np

def mjd2sec(mjd_values, mjd_ref=48282.0):
    """
    Converts Modified Julian Dates to seconds since a reference epoch.

    Parameters:
        mjd_values (float, list, or np.ndarray): MJD value(s) to convert.
        mjd_ref (float): Reference MJD to subtract from. Default is 48282.0 (March 5, 1991).

    Returns:
        np.ndarray: Time values in seconds since mjd_ref.
    """

    mjd_array = np.asarray(mjd_values, dtype=float)
    seconds = np.round((mjd_array - mjd_ref) * 86400)
    return seconds

def sec2mjd(sec_values, mjd_ref=48282.0):
    """
    Converts Seconds to MJD since a reference epoch.

    Parameters:
        sec_values (float, list, or np.ndarray): Seconds value(s) to convert.
        mjd_ref (float): Reference MJD to subtract from. Default is 48282.0 (March 5, 1991).

    Returns:
        np.ndarray: Time values in MJD.
    """

    mjd_array = np.asarray(sec_values, dtype=float)
    mjd_array = sec_values/86400 + mjd_ref
    
    return mjd_array

def valid_mjd(data_dict, path_keys):
    """
    Finds the MJDs that are present in all arrays in `path_keys` where confidence == 1.

    Returns:
        np.ndarray: Sorted array of common MJDs
    """
    mjd_sets = []

    for key in path_keys:
        arr = data_dict[key]
        conf_mask = arr[:, 2] == 1
        mjds = np.unique(arr[conf_mask, 0])
        mjd_sets.append(set(mjds))

    common_mjds = set.intersection(*mjd_sets)
    return np.array(sorted(common_mjds))


def data_path(data_dict, path_keys):
    """
    Filters each array in data_dict to retain only rows where:
    - time is in common_times
    - confidence/flag == 1

    Returns:
        dict: key -> filtered array (with exact matching times)
    """
    filtered_data = {}
    common_times = valid_mjd(data_dict, path_keys)

    for key in path_keys:
        arr = data_dict[key]

        # Ensure times are rounded
        times = np.round(arr[:, 0]).astype(int)
        arr[:, 0] = times

        # Apply mask
        mask = (arr[:, 2] == 1) & np.isin(times, common_times)
        filtered_arr = arr[mask]

        # Remove duplicates per timestamp (keep first)
        _, unique_idx = np.unique(filtered_arr[:, 0], return_index=True)
        filtered_arr = filtered_arr[sorted(unique_idx)]

        if len(filtered_arr) != len(common_times):
            print(f"Warning: {key} expected {len(common_times)} points but got {len(filtered_arr)}")

        filtered_data[key] = filtered_arr

    return filtered_data