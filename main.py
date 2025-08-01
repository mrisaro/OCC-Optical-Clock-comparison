"""
Created on Thu Jul 10 16:31:35 2025

@author: mar
"""

import datamet as dm
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import metronauta as mt
from opt_met import ratio_dat
from uncertainties import ufloat

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
        ratio_data = ratio_dat(data_dict, paths, key, output_folder='ratio_data/')
        ratio_campaign[key] = ratio_data
    except Exception as e:
        print(f"Failed for {key[0]} - {key[1]}: {e}")

t1 = time.time()
print(f"Done computing all ratios in {t1 - t0:.2f} seconds.")

#%% Data test for the ratio analysis
summary_lines = ["#Ratio\ty(u)\ty\tuTot\tuA*\tuB1\tuB2\tuGRS\tbirge\tndof\tdays"]

output_dir = Path("ratio_plots")
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
            plot=True
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
        print(f"Skipping {key} due to error: {e}")
        continue
    
    i=i+1
    
# Save summary file
summary_file = "summary_matias.txt"
with open(summary_file, "w") as f:
    f.write("\n".join(summary_lines))

print("All plots and summary saved.")



