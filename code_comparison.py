# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:06:40 2025

Script for comparison Between Marco and Matias. 

@author: mar
"""
import datamet as dm
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import metronauta as mt
from opt_met import ratio_dat
from uncertainties import ufloat
from pathlib import Path

data_folder =  "data_2025_03_TOCK/"
data_dict = dm.load_all_data(data_folder)

G, folders, paths = dm.build_clock_network(
    data_folder=data_folder,
    yaml_path="oscillators.yml",
    pair_file="oscillator_pairs.txt"
)

t0 = time.time()
ratio_campaign = {}

for i,key in enumerate(list(paths.keys())):
    print(f'Computing ratio for {key[0]} - {key[1]}')
    print(f'Comparator {i} of {len(paths.keys())}')
    try:
        ratio_data = ratio_dat(data_dict, paths, key)
        ratio_campaign[key] = ratio_data
    except Exception as e:
        print(f"❌ Failed for {key[0]} - {key[1]}: {e}")

t1 = time.time()
print(f"Done computing all ratios in {t1 - t0:.2f} seconds.")

#%%
summary_lines = ["#Ratio\ty(u)\ty\tuTot\tuA*\tuB1\tuB2\tuGRS\tbirge\tndof\tdays"]

output_dir = Path("ratio_plots_tris")
output_dir.mkdir(exist_ok=True)

name_map = {
    "PTB_In_CombKnoten": "PTB-In1",
    "PTB_Al_CombAl": "PTB-Al+",
    "PTB_Yb_CombKnoten": "PTB-Yb1E3",
    "PTB_Sr3_CombKnoten": "PTB-Sr3",
    "INRIM_PTBSr4": "PTB-Sr4",
    "INRIM-ITYb1": "IT-Yb1",
    "NPL-YbE3": "NPL-E3Yb+3"
}
i = 0
for key in ratio_campaign.keys():
    print(f"Working on link {key}")
    data = ratio_campaign[key]

    try:
        # Compute uncertainty and plot
        result = mt.compute_uncertainty(
            data,
            link_name=key,
            grsc=0.0,
            plot=False
        )
        
        # Save the plot
        plot_path = output_dir / f"{i} {key[0]}-{key[1]}.png"
        plt.savefig(plot_path)
        plt.close()

        final_y = ufloat(result['mu'], result['final_u'])

        # Append row to summary
        summary_lines.append(
            f"{name_map.get(key[1], key[1])}/{name_map.get(key[0], key[0])}\t"
            f"{final_y:.2uS}\t"
            f"{result['mu']:.3e}\t"
            f"{result['final_u']:.1e}\t"
            f"{result['uAstar']:.1e}\t"
            f"{result['uB']:.1e}\t"
            f"{result['uB']:.1e}\t"
            f"{2e-17:.1e}\t"
            f"{result['birge_ratio']:.2f}\t"
            f"{result['n_days'] - 1}\t"
            f"{result['n_days']:.1f}"
        )

    except Exception as e:
        print(f"⚠️ Skipping {key} due to error: {e}")
        continue
    
    i=i+1
    
# Save summary file
summary_file = "summary_matias.txt"
with open(summary_file, "w") as f:
    f.write("\n".join(summary_lines))

print("✅ All plots and summary saved.")


#%% Pick a test path
tot_time_list = []
key = list(paths.keys())
t0 = time.time()

for i in key:
    print(f'Testing the comparator {i}')
    folders_ky = paths[i]['folders']
    
    common_times = valid_mjd(data_dict, folders_ky)
    
    tot_time_list.append([i[0]+'-'+i[1], len(common_times)])
        
t1 = time.time()
np.savetxt('total_time_mati.txt',tot_time_list, fmt='%s', delimiter=',', newline='\n', header='Link, Total time')

data_mati = np.genfromtxt('total_time_mati.txt', delimiter=",", dtype=str, comments="#")
links_mati = data_mati[:,0]
times_mati = data_mati[:,1].astype(float)

data_marco = np.genfromtxt('total_time_links.txt', delimiter=",", dtype=str, comments="#")
links_marco = data_marco[:,0]
times_marco = data_marco[:,1].astype(float)
outliers_marco = data_marco[:,2].astype(float)

import csv

def normalize_link(link):
    """Ensure the link is always formatted as 'A-B' with A < B."""
    parts = link.split('-')
    return '-'.join(sorted(parts))

def compare_links_to_csv_with_outliers(links_mati, times_mati, links_marco, times_marco, outliers_marco, output_file="comparison.csv"):
    # Normalize and create dicts
    normalized_mati = {normalize_link(link): time for link, time in zip(links_mati, times_mati)}
    normalized_marco = {normalize_link(link): (time, outlier) for link, time, outlier in zip(links_marco, times_marco, outliers_marco)}

    # Find common links
    common_links = sorted(set(normalized_mati.keys()) & set(normalized_marco.keys()))

    # Prepare rows
    rows = []
    for link in common_links:
        time_mati = normalized_mati[link]
        time_marco, outlier = normalized_marco[link]
        diff = time_mati - time_marco  # signed difference
        rows.append([link, time_mati, time_marco, diff, outlier])

    # Write to CSV
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Link", "Time_Mati", "Time_Marco", "Difference", "Marco_Outliers"])
        writer.writerows(rows)

    print(f"✅ CSV file saved as '{output_file}' with {len(rows)} matched links.")


#%%
from scipy.stats import norm
import allantools
import pandas as pd
from matplotlib.gridspec import GridSpec
from timet import sec2mjd
import matplotlib.gridspec as gridspec

def compute_valid_mask(R):
    mu, std = norm.fit(R)
    limit = (np.percentile(R, 95) - np.percentile(R, 5)) * 2
    val_mask = np.abs(R - mu) < limit
    return val_mask

def plot_ratio_comparison(key, ratio_campaign, ratio_marco, shared_keys, 
                          save_path=None):
    data_marco = np.array(ratio_marco[shared_keys[key]].data)
    
    t1, R1 = sec2mjd(data_marco[:, 0]), data_marco[:, 1]
    t2, R2 = ratio_campaign[key]['mjd'], ratio_campaign[key]['ratio']
    rho0 = float(ratio_campaign[key]['r0'])
    
    mask1 = compute_valid_mask(R1)
    mask2 = compute_valid_mask(R2)

    R1_valid, t1_valid = R1[mask1], t1[mask1]
    R2_valid, t2_valid = R2[mask2], t2[mask2]

    (tau1, adev1, _, _) = allantools.oadev(R1_valid, rate=1.0, data_type='freq', taus=None)
    (tau2, adev2, _, _) = allantools.oadev(R2_valid, rate=1.0, data_type='freq', taus=None)
    
    # Summary stats
    mean1, total1, outliers1 = np.mean(R1), len(R1), len(R1) - np.sum(mask1)
    mean2, total2, outliers2 = np.mean(R2), len(R2), len(R2) - np.sum(mask2)

    # Create subplot layout: 2 rows, 3 columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 6]})

    # Table in the first row (merged across 3 columns)
    ax_table = axes[0, 0]
    for ax in axes[0]: ax.axis('off')  # hide all axes in row 0
    table_data = [
        [f"{key[1]}/{key[0]}", "Mean", "Total Points", "Outliers"],
        ["Marco", f"{mean1:.3e}", f"{total1}", f"{outliers1}"],
        ["Campaign", f"{mean2:.3e}", f"{total2}", f"{outliers2}"]
    ]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1, 1)
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Plots
    ax1, ax2, ax3 = axes[1, 0], axes[1, 1], axes[1, 2]

    ax1.plot(t1_valid, R1_valid,'.', label="Marco", alpha=0.7)
    ax1.set_title("Marco Raw Ratio")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Ratio")
    ax1.grid(True)

    ax2.plot(t2_valid, R2_valid,'.', label="Matias", alpha=0.7, color='orange')
    ax2.set_title("Campaign Raw Ratio")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    ax3.loglog(tau1, adev1/rho0, 'o-', label="Marco ADEV")
    ax3.loglog(tau2, adev2/rho0, 's-', label="Campaign ADEV")
    ax3.set_title("Allan Deviation")
    ax3.set_xlabel("Averaging Time τ [s]")
    ax3.set_ylabel("σ_y(τ)")
    ax3.grid(True, which="both", ls="--")
    ax3.legend()

    plt.tight_layout()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        nameA, nameB = key
        fname = f"{nameA}__{nameB}.png"
        plt.savefig(os.path.join(save_path, fname), dpi=300)
        plt.close()
    else:
        plt.show()
    
#%% Step 0: Your name map
from joblib import load
ratio_marco = load("ratio_marco.joblib")

long_names = {
    "IT-Yb1": "INRIM_ITYb1",
    "PTB-In1": "PTB_In_CombKnoten",
    "PTB-Al+": "PTB_Al_CombAl",
    "PTB-Yb1E3": "PTB_Yb_CombKnoten",
    "PTB-Sr3": "PTB_Sr3_CombKnoten",
    "PTB-Sr4": "INRIM_PTBSr4",
    "OBSPARIS-Sr2": "OBSPARIS_Sr2",
    "OBSPARIS-SrB": "OBSPARIS_SrB",
    "PTB-Yb1E3E2": "PTB_Yb1E2_CombYb",
    "NPL-E3Yb+3": "NPL_YbE3",
    "NPL-Sr1": "NPL_Sr1",
}

# Invert long_names to go from long to short
inverse_long_names = {v: k for k, v in long_names.items()}

# Build mapping from ratio_campaign keys to both possible string key forms
def build_key_mappings(ratio_campaign, inverse_map, ratio_marco):
    shared_keys = {}
    unmatched_keys = {}

    for key in ratio_campaign.keys():
        longA, longB = key
        shortA = inverse_map.get(longA, longA)
        shortB = inverse_map.get(longB, longB)

        key1 = f"{shortB}/{shortA}"  # B over A
        key2 = f"{shortA}/{shortB}"  # A over B

        if key1 in ratio_marco:
            shared_keys[key] = key1
        elif key2 in ratio_marco:
            shared_keys[key] = key2
        else:
            unmatched_keys[key] = (key1, key2)  # Neither match found

    return shared_keys, unmatched_keys

# Run the comparison
shared_keys, unmatched_keys = build_key_mappings(ratio_campaign, inverse_long_names, ratio_marco)

# Also check which keys in marco are completely unmatched
mapped_values = set(shared_keys.values())
unmatched_in_marco = set(ratio_marco.keys()) - mapped_values

# Print results
print(f"✅ Matched keys: {len(shared_keys)}")
print(f"❌ Unmatched keys from ratio_campaign: {len(unmatched_keys)}")
print(f"❌ Unmatched keys from ratio_marco: {len(unmatched_in_marco)}")

# Preview unmatched campaign keys
print("\n⚠️ Example unmatched keys in ratio_campaign:")
for k, v in list(unmatched_keys.items())[:10]:
    print(f"  {k} -> Tried: {v[0]} or {v[1]}")

# Preview unmatched marco keys
print("\n⚠️ Example unmatched keys in ratio_marco:")
for k in list(unmatched_in_marco)[:10]:
    print(f"  {k}")
#%% Comparing each pair
example_key = list(shared_keys.keys())[-5]

# Plot the comparison
plot_ratio_comparison(example_key, ratio_campaign, ratio_marco, shared_keys)
#%%
output_folder = "comparison_plots"

for key in shared_keys:
    plot_ratio_comparison(key, ratio_campaign, ratio_marco, shared_keys, save_path=output_folder)

print("✅ All plots saved.")
