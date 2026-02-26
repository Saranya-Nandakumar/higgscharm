"""
Script to analyze c-jets (hadronFlavour == 4) in SomeSMSignal parquet files.
1. Find the number of c-jets in each event
2. For events with exactly 1 c-jet, plot the position (eta vs phi)
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from collections import Counter

# Find all SomeSMSignal parquet files
parquet_pattern = "/eos/user/s/snandaku/higgscharm/outputs/hplusc/*/SomeSMSignal*/base/*.parquet"
parquet_files = glob.glob(parquet_pattern)

print(f"Found {len(parquet_files)} parquet files")

# Collect data
all_n_cjets = []  # Number of c-jets per event
single_cjet_eta = []  # Eta of c-jet for events with exactly 1 c-jet
single_cjet_phi = []  # Phi of c-jet for events with exactly 1 c-jet

for i, pf in enumerate(parquet_files):
    if i % 100 == 0:
        print(f"Processing file {i+1}/{len(parquet_files)}...")

    try:
        table = pq.read_table(pf)

        jet_eta = table['jet_eta'].to_pylist()
        jet_phi = table['jet_phi'].to_pylist()
        jet_flavour = table['jet_hadronFlavour'].to_pylist()

        for evt_idx in range(len(jet_eta)):
            evt_eta = jet_eta[evt_idx]
            evt_phi = jet_phi[evt_idx]
            evt_flav = jet_flavour[evt_idx]

            # Find c-jets (flavour == 4)
            cjet_indices = [j for j, flav in enumerate(evt_flav) if flav == 4]
            n_cjets = len(cjet_indices)
            all_n_cjets.append(n_cjets)

            # For events with exactly 1 c-jet, store position
            if n_cjets == 1:
                idx = cjet_indices[0]
                single_cjet_eta.append(evt_eta[idx])
                single_cjet_phi.append(evt_phi[idx])

    except Exception as e:
        print(f"Error processing {pf}: {e}")
        continue

print(f"\nTotal events processed: {len(all_n_cjets)}")

# Distribution of number of c-jets per event
print("\n=== Distribution of c-jets per event ===")
cjet_counts = Counter(all_n_cjets)
for n in sorted(cjet_counts.keys()):
    pct = 100 * cjet_counts[n] / len(all_n_cjets)
    print(f"  {n} c-jets: {cjet_counts[n]} events ({pct:.1f}%)")

print(f"\nEvents with exactly 1 c-jet: {len(single_cjet_eta)}")

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Distribution of number of c-jets per event
ax1 = axes[0]
n_cjets_arr = np.array(all_n_cjets)
max_cjets = min(max(n_cjets_arr), 10)
ax1.hist(n_cjets_arr, bins=np.arange(-0.5, max_cjets + 1.5, 1),
         edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of c-jets per event')
ax1.set_ylabel('Number of events')
ax1.set_title('Distribution of c-jets (flavour=4) per event')
ax1.set_xticks(range(max_cjets + 1))

# Plot 2: Eta-Phi position for single c-jet events
ax2 = axes[1]
if len(single_cjet_eta) > 0:
    ax2.scatter(single_cjet_eta, single_cjet_phi, alpha=0.3, s=5)
    ax2.set_xlabel(r'$\eta$ (c-jet)')
    ax2.set_ylabel(r'$\phi$ (c-jet)')
    ax2.set_title(f'c-jet position (1 c-jet events, N={len(single_cjet_eta)})')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-np.pi, np.pi)
else:
    ax2.text(0.5, 0.5, 'No single c-jet events', ha='center', va='center')

# Plot 3: 2D histogram of eta-phi
ax3 = axes[2]
if len(single_cjet_eta) > 0:
    h = ax3.hist2d(single_cjet_eta, single_cjet_phi,
                   bins=[25, 25],
                   range=[[-2.5, 2.5], [-np.pi, np.pi]],
                   cmap='viridis')
    plt.colorbar(h[3], ax=ax3, label='Events')
    ax3.set_xlabel(r'$\eta$ (c-jet)')
    ax3.set_ylabel(r'$\phi$ (c-jet)')
    ax3.set_title('c-jet position (2D histogram)')

plt.tight_layout()
plt.savefig('/afs/cern.ch/user/s/snandaku/Higgscharmfresh/higgscharm/scripts/cjet_position_plot.png',
            dpi=150, bbox_inches='tight')
plt.savefig('/afs/cern.ch/user/s/snandaku/Higgscharmfresh/higgscharm/scripts/cjet_position_plot.pdf',
            bbox_inches='tight')
print(f"\nPlots saved to:")
print(f"  cjet_position_plot.png")
print(f"  cjet_position_plot.pdf")

plt.show()
