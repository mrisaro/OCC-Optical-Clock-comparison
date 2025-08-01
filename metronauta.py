"""
Created on Thu Jul 17 10:47:48 2025
Metronauta is a library with functions to deal with the statistical analysis
of optical clock comparisons.
@author: mar
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import allantools
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
from uncertainties import ufloat
from mpmath import mpf
from opt_met import ratio_format
import yaml

#------- functions for data loading -------

def load_yaml(yaml_path):
    """Load a YAML file and return its content as a dictionary."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)
    
    
#------- functions for ratio managing -------
def ratio_outlier(data, n_sigma=5, p_low=5, p_high=95, scale=1e15, plot=False, title=None):
    """
    Determine outliers for a ratio dataset using both Gaussian and percentile methods.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'ratio' (and optionally 'mjd', etc.).
    n_sigma : float, optional
        Sigma threshold for Gaussian-based outlier removal.
    p_low : float, optional
        Lower percentile threshold (default: 5).
    p_high : float, optional
        Upper percentile threshold (default: 95).
    scale : float, optional
        Scaling factor applied to ratio values before analysis (default: 1e15).
    plot : bool, optional
        Whether to plot the histogram and threshold lines.
    title : str, optional
        Title for the plot (used as label for the key).

    Returns
    -------
    masks : dict
        Dictionary with two boolean masks:
        - 'gaussian_mask'
        - 'percentile_mask'
    """
    R = data['ratio']
    R_scaled = R * scale

    # --- Gaussian Fit Method ---
    mu, std = norm.fit(R_scaled)
    lower_gauss = mu - n_sigma * std
    upper_gauss = mu + n_sigma * std
    gaussian_mask = (R_scaled >= lower_gauss) & (R_scaled <= upper_gauss)

    # --- Percentile Method ---
    p_low_val = np.percentile(R_scaled, p_low)
    p_high_val = np.percentile(R_scaled, p_high)
    delta = (p_high_val - p_low_val) * 2
    lower_perc = mu - delta
    upper_perc = mu + delta
    percentile_mask = (R_scaled >= lower_perc) & (R_scaled <= upper_perc)

    # --- Plot if requested ---
    if plot:
        plt.figure(figsize=(10, 6), num=f"histogram ratio - {title if title else ''}")

        # Histogram
        counts, bins = np.histogram(R_scaled, bins=50, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        outlier_mask = (bin_centers < lower_perc) | (bin_centers > upper_perc)

        # Bar plot with per-bin color
        for i in range(len(counts)):
            color = 'orange' if outlier_mask[i] else 'gray'
            plt.bar(bin_centers[i], counts[i], width=(bins[1] - bins[0]),
                    color=color, alpha=0.6, log=True)

        # Gaussian fit line
        pdf = norm.pdf(bin_centers, mu, std)
        pdf = pdf*np.max(counts)/np.max(pdf)
        plt.plot(bin_centers, pdf, 'r-', lw=2, label='Gaussian fit')

        # Threshold lines
        plt.axvline(lower_gauss, color='blue', linestyle='--', label=f'Gaussian -{n_sigma}σ')
        plt.axvline(upper_gauss, color='blue', linestyle='--')

        plt.axvline(lower_perc, color='green', linestyle='--', label=f'Percentile bounds ({p_low}-{p_high}) x2')
        plt.axvline(upper_perc, color='green', linestyle='--')

        # Labels
        plt.title(f'Ratio Histogram {f"– {title}" if title else ""} mu = {mu:.2f}')
        plt.xlabel(f'Ratio × {scale:.0e}')
        plt.ylabel('Density (log scale)')
        plt.legend()
        plt.ylim(bottom=1e-6, top = 2*np.max(counts))  # reasonable log-scale floor
        plt.grid(linestyle='--')
        plt.tight_layout()

    return {
        'gaussian_mask': gaussian_mask,
        'percentile_mask': percentile_mask
    }

def mask_perc(data):
    """
    Determine valid points based on a percentile criterion.
    
    A point is considered valid if it lies within twice the range between 
    the 5th and 95th percentiles, centered around the mean of the data.
    
    Parameters
    ----------
    data : array-like
        Input data to evaluate validity.
    
    Returns
    -------
    mask : array of bool
        Boolean mask where True indicates a valid point and False an outlier.
    """
    
    mu, std = norm.fit(data)
    limit = (np.percentile(data, 95) - np.percentile(data, 5)) * 2
    val_mask = np.abs(data - mu) < limit
    return val_mask

#----- Functions about statistic analysis -----

def compute_adev_summary(R, fs=1.0, white_index=10):
    """
    Compute Allan deviation, confidence bounds, and white noise floor estimate.

    Parameters
    ----------
    R : array-like
        Frequency ratio data (filtered).
    fs : float
        Sampling rate in Hz. Default is 1.0.
    white_index : int
        Index of tau to use for white noise estimate. Default is 10 to have
        tau = 2**10s = 1024s. A criteria used by M. Pizzocaro.

    Returns
    -------
    dict containing:
        - tau: averaging times
        - adev: Allan deviation
        - admin: lower confidence bound
        - admax: upper confidence bound
        - taua: tau array for totdev
        - ada: total deviation
        - w_fit: white noise estimate at tau=1s
        - uA: statistical uncertainty
    """
    taua, ada, _, _ = allantools.totdev(R, rate=fs, data_type="freq", taus="octave")
    tau, adev, _, adn = allantools.oadev(R, rate=fs, data_type="freq", taus="octave")

    # Compute EDF and error bars
    N = adn
    m = tau
    edf = (3 * (N - 1) / (2 * m) - 2 * (N - 2) / N) * 4 * m**2 / (4 * m**2 + 5)
    edf = np.clip(edf, 1, None)

    admax = adev * np.sqrt(edf / scipy.stats.chi2.ppf(0.1, edf))
    admin = adev * np.sqrt(edf / scipy.stats.chi2.ppf(0.9, edf))

    # White noise estimation
    w_fit = adev[white_index] * np.sqrt(1024)
    uA = w_fit / np.sqrt(len(R) / fs)

    return {'tau': tau, 'adev': adev,'admax': admax,'admin': admin,
        'taua': taua, 'ada': ada,'w_fit': w_fit, 'uA': uA}

def compute_uGRS(link_key, grs_data, name_map):
    """
    Compute GRS uncertainty for a link.

    Parameters
    ----------
    link_key : tuple
        Tuple like ('INSTITUTE_CLOCKA', 'INSTITUTE_CLOCKB')
    grs_data : dict
        Dictionary loaded from clock_uGRS.yml
    name_map : dict
        Dictionary mapping long names to short names (e.g., for YAML)

    Returns
    -------
    uGRS : float
        Total GRS uncertainty (quadrature sum of clock uncertainties)
    """
    a, b = link_key
    a_short = name_map.get(a, a)
    b_short = name_map.get(b, b)

    # Check if they belong to the same institute
    institute_a = a_short.split('-')[0]
    institute_b = b_short.split('-')[0]
    same_institute = institute_a == institute_b
    
    try:
        if same_institute:
            # use local GRS uncertainty
            ug1 = ug2 = grs_data[a_short][1]
        else:
            ug1 = grs_data[a_short][0]
            ug2 = grs_data[b_short][0]
    except KeyError as e:
        raise ValueError(f"Missing GRS data for: {e}")

    return np.sqrt(ug1**2 + ug2**2)

def compute_birge(R, t, usysA, usysB, uGRS, w_fit, mu_val, 
                        grsc=0.0, min_points_per_day=864):
    """
    Compute Birge ratio from daily-binned data.

    Parameters
    ----------
    R : array
        Frequency ratio data (filtered).
    t : array
        Timestamps (MJD or seconds). 
    usysA, usysB : array
        Systematic uncertainties. 
    uGRS : float
        Combined GRS uncertainty.
    w_fit : float
        White noise amplitude at 1s.
    mu_val : float
        Mean of filtered data.
    grsc : float
        Gravity shift correction. Already made that in my code
    min_points_per_day : int
        Minimum points per day to be considered valid.

    Returns
    -------
    dict with:
        - birge: Birge ratio
        - uAstar: modified uA
        - ndof: degrees of freedom
        - daily_ustat, daily_usys
        - daily_means
        - daily_counts
        - full_days: list of [start, stop] times
    """
    
    if np.max(t) > 1e5:
        t_days = t // 86400
    else:
        t_days = np.floor(t)

    unique_days = np.unique(t_days)
    daily_means, daily_counts, daily_usys, full_days = [], [], [], []

    for day in unique_days:
        mask = t_days == day
        if np.sum(mask) >= min_points_per_day:
            daily_means.append(np.mean(R[mask]))
            daily_counts.append(np.sum(mask))
            sys_day = np.sqrt(usysA[mask]**2 + usysB[mask]**2 + uGRS**2)
            daily_usys.append(np.mean(sys_day))
            full_days.append([day * 86400, (day + 1) * 86400])

    daily_means = np.array(daily_means)
    daily_counts = np.array(daily_counts)
    daily_usys = np.array(daily_usys)
    full_days = np.array(full_days)

    if len(daily_means) > 1:
        daily_ustat = w_fit / np.sqrt(daily_counts)
        total_daily_u = np.sqrt(daily_ustat**2 + daily_usys**2)
        chi2 = np.sum(((daily_means + grsc - mu_val) ** 2) / total_daily_u**2)
        ndof = len(daily_means) - 1
        chi2red = chi2 / ndof
        birge = np.sqrt(chi2red)
        uAstar = w_fit / np.sqrt(len(R)) * max(birge, 1)
    else:
        birge = np.nan
        uAstar = w_fit / np.sqrt(len(R))
        ndof = 0
        daily_ustat = np.array([])

    return {
        'birge': birge,
        'uAstar': uAstar,
        'ndof': ndof,
        'daily_means': daily_means,
        'daily_counts': daily_counts,
        'daily_usys': daily_usys,
        'daily_ustat': daily_ustat,
        'full_days': full_days
    }

def report_uncertainty(mu_val, uAstar, uB, R0):
    """
    Final uncertainty reporting and frequency ratio formatting.

    Parameters
    ----------
    mu_val : float
        Mean frequency ratio offset.
    uAstar : float
        Inflated statistical uncertainty.
    uB : float
        Combined systematic uncertainty.
    R0 : float
        Nominal frequency ratio.

    Returns
    -------
    dict with:
        - final_u
        - final_ratio (ufloat)
        - ratio_value (mpmath.mpf)
        - ratio_uncertainty (float)
        - rout (formatted string)
    """
    final_u = np.sqrt(uAstar**2 + uB**2)
    final = ufloat(mu_val, final_u)

    ratio_val = (mpf(mu_val) + mpf(1)) * R0
    ratio_unc = float(ratio_val) * final_u
    rout = ratio_format(ratio_val, ratio_unc)

    return {
        'final_u': final_u,
        'final_mu': final,
        'ratio_value': ratio_val,
        'ratio_uncertainty': ratio_unc,
        'rout': rout
    }


def compute_uncertainty(
    data, link_name, fs=1.0, window_sec=3600*3,
    grsc=0.0,
    min_points_per_day=864,
    white_index=10,
    plot=False
):
    
    # Load dictionary data into different arrays.
    R = np.array(data['ratio'])
    t = np.array(data['mjd'])
    usysA = np.array(data['usysA'])
    usysB = np.array(data['usysB'])
    R0 = data['r0']
    
    #Compute the valid pointswith the mask_perc function.
    val_mask = mask_perc(R)
    R = R[val_mask]
    t = t[val_mask]
    usysA = usysA[val_mask]
    usysB = usysB[val_mask]

    # --- Deviation calculations ---
    adev_info = compute_adev_summary(R, fs=fs, white_index=white_index)
    
    tau, adev = adev_info['tau'], adev_info['adev']         # OADEV compute
    admax, admin = adev_info['admax'], adev_info['admin']   # OADEV uncertainty
    
    taua, ada = adev_info['taua'], adev_info['ada']         # TotalAdev
    w_fit, uA = adev_info['w_fit'], adev_info['uA']
    
    # ---- Compute uGRS ------
    # Load GRS uncertainties
    clock_uGRS = load_yaml("clock_uGRS.yml")

    # Load clock name map
    name_map = load_yaml("clock_name_map.yml")
    uGRS = compute_uGRS(link_name, clock_uGRS, name_map)
    
    uB1 = np.mean(usysA)
    uB2 = np.mean(usysB)
    uB = np.sqrt(uB1**2 + uB2**2 + uGRS**2)

    mu_val = np.mean(R)

    # Birge binning
    birge_info = compute_birge(R, t, usysA, usysB, uGRS, w_fit, mu_val)
    
    uAstar = birge_info['uAstar']
    birge = birge_info['birge']
    ndof = birge_info['ndof']
    daily_means = birge_info['daily_means']
    full_days = birge_info['full_days']
    daily_ustat = birge_info['daily_ustat']
    daily_usys = birge_info['daily_usys']
    
    # Final uncertainty report
    report = report_uncertainty(mu_val, uAstar, uB, R0)
    final_u = report['final_u']
    final = report['final_mu']
    ratio_val = report['ratio_value']
    ratio_unc = report['ratio_uncertainty']
    rout = report['rout']
    
    if plot:
        plot_uncertainty_summary(
            link_name, t, R, grsc, final, final_u,
            uAstar, uB1, uB2, uGRS, birge, ndof, rout,
            full_days, np.column_stack([np.mean(full_days, axis=1), daily_means]),
            daily_ustat, daily_usys,
            tau, adev, admax, admin,
            white=w_fit, n_outliers=len(val_mask) - sum(val_mask), 
            taufit=tau, ada=ada, taua=taua
        )

    return {
        'mu': mu_val,
        'uA': uA,
        'uAstar': uAstar,
        'uB': uB,
        'birge_ratio': birge,
        'final_u': final_u,
        'final_ratio': final,
        'ratio_value': ratio_val,
        'ratio_uncertainty': ratio_unc,
        'uptime_seconds': np.sum(val_mask) / fs,
        'outlier_count': len(val_mask) - np.sum(val_mask),
        'n_days': len(daily_means),
        'taua': taua,
        'ada': ada,
        'tau': tau,
        'adev': adev,
        'admax': admax,
        'admin': admin
    }


def plot_uncertainty_summary(
    link_name, t_valid, R_valid, grsc, final, final_u,
    uAstar, uB1, uB2, uGRS, birge, ndof, rout,
    days, ddata, daily_ustat, daily_usys,
    tau, adev, admax, admin,
    white, n_outliers, taufit=None, ada=None, taua=None
    ):
    """
    function to plot and report uncertainty.
    
    Returns
    -------
    None.

    """
    # some recurrent variables.
    mu_val = final.nominal_value 
    
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)

    fig.suptitle(
        f"{link_name}: Tot time: {len(R_valid)} sec, outliers = {n_outliers} \n"
        f"y = {final:.2uS} (uA* = {uAstar:.1e}, uB1 = {uB1:.1e}, uB2 = {uB2:.1e}, "
        f"uGRS = {uGRS:.1e}, Birge = {birge:.1f}, ndof = {ndof})\n"
        f"R = {rout}",
        fontsize=12
    )
    
    # ---- Raw y(t) ----
    ax0 = fig.add_subplot(gs[0, 0])
    mjd = t_valid
    ax0.plot(mjd - 60000, R_valid + grsc, ".", label="Raw", rasterized=True)
    ax0.plot(mjd - 60000, uniform_filter1d(R_valid + grsc, size=10000), ".", label="Rolling mean", rasterized=True)
    ax0.set_ylabel("y")
    ax0.set_xlabel("MJD - 60000")
    ax0.grid(linestyle='--')
    ax0.set_xlim(60745 - 60000, 60780 - 60000)
    ax0.set_xticks(np.arange(60745, 60781, 5) - 60000)
    ax0.legend()

    # ---- Daily averages ----
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    if len(ddata) > 0:
        timetags = np.mean(days, axis=1) / 86400
        ax1.axhspan(mu_val - final_u, mu_val + final_u, color="C2", alpha=0.5, label="Final uncertainty")

        if len(daily_ustat) == len(ddata):
            total_err = np.sqrt(daily_ustat**2 + daily_usys**2)
            ax1.errorbar(timetags - 60000, ddata[:, 1] + grsc, yerr=total_err, fmt="o", label="Stat + Sys")
            ax1.errorbar(timetags - 60000, ddata[:, 1] + grsc, yerr=daily_ustat, fmt="o", label="Stat only")
        else:
            ax1.plot(timetags - 60000, ddata[:, 1] + grsc, "o", label="Daily mean")

    ax1.set_ylabel("y (averaged)")
    ax1.set_xlabel("MJD - 60000")
    ax1.grid(linestyle='--')
    ax1.set_xlim(60745 - 60000, 60780 - 60000)
    ax1.set_xticks(np.arange(60745, 60781, 5) - 60000)
    ax1.legend()

    # ---- ADEV ----
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.errorbar(tau, adev, yerr=[adev - admin, admax - adev], fmt="o", label="ADEV")
    if taufit is not None and white is not None:
        ax2.loglog(taufit, white / np.sqrt(taufit), "-", label=f"White noise = {white:.2e}")
    if taua is not None and ada is not None:
        ax2.loglog(taua, ada, "-", label="Tot dev")
    ax2.set_ylabel("ADEV")
    ax2.set_xlabel("Tau [s]")
    ax2.grid(True, which="both")
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])  # space for suptitle
