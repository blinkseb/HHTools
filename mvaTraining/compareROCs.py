import os
import plotTools
from common import *
import keras

import matplotlib.pyplot as plt

inputs = [
        "jj_pt", 
        "ll_pt",
        "ll_M",
        "ll_DR_l_l",
        "jj_DR_j_j",
        "llmetjj_DPhi_ll_jj",
        "llmetjj_minDR_l_j",
        "llmetjj_MTformula",
        "isSF"
        ]

cut = "(91 - ll_M) > 15"

# FIXME: Put b-tagging SFs back once they are correct
weight = {
            # '__base__': "event_weight * trigeff * jjbtag_heavy * jjbtag_light * llidiso * pu",
            '__base__': "event_weight * trigeff * llidiso * pu",
            'DYJetsToLL_M.*': "dy_nobtag_to_btagM_weight"
}

mass = 900

dataset = DatasetManager(inputs, weight, cut)
dataset.load_resonant_signal(masses=[mass], add_mass_column=True)
dataset.load_backgrounds(add_mass_column=True)

models = [
    # {
        # 'file': 'hh_resonant_trained_models/2017-01-18_400_dy_estimation_from_BDT_on_GPU_deeper_150epochs/hh_resonant_trained_model.h5',
        # 'legend': 'NN training with M=400 GeV',
        # 'color': '#8E2800',
        # 'no_mass_column': True
    # },

    {
        'file': 'hh_resonant_trained_models/2017-01-18_900_dy_estimation_from_BDT_on_GPU_deeper/hh_resonant_trained_model.h5',
        'legend': 'NN training with M=900 GeV',
        'color': '#8E2800',
        'no_mass_column': True
    },

    {
        'file': 'hh_resonant_trained_models/2017-01-19_400_650_900_dy_estimation_from_BDT_on_GPU_deeper_lr_scheduler_100epochs/hh_resonant_trained_model.h5',
        'legend': 'NN training with M=400, 650, 900 GeV',
        'color': '#468966'
    }
]

fig = plt.figure(1, figsize=(7, 7))
ax = fig.add_subplot(111)

for m in models:
    print("Evaluating predictions from %r" % m['file'])
    model = keras.models.load_model(m['file'])

    ignore_n_last_columns = 0
    if 'no_mass_column' in m and m['no_mass_column']:
        ignore_n_last_columns = 1

    print("Signal...")
    signal_predictions = dataset.get_signal_predictions(model, ignore_last_columns=ignore_n_last_columns)
    print("Background...")
    background_predictions = dataset.get_background_predictions(model, ignore_last_columns=ignore_n_last_columns)
    print("Done.")

    n_signal, _, binning = plotTools.binDataset(signal_predictions, dataset.get_signal_weights(), bins=50, range=[0, 1])
    n_background, _, _ = plotTools.binDataset(background_predictions, dataset.get_background_weights(), bins=binning)

    x, y = plotTools.get_roc(n_signal, n_background)
    ax.plot(x, y, '-', color=m['color'], lw=2, label=m['legend'])

   
ax.set_xlabel("Background efficiency", fontsize='large')
ax.set_ylabel("Signal efficiency", fontsize='large')

ax.margins(0.05)
fig.set_tight_layout(True)

ax.legend(loc='lower right', numpoints=1, frameon=False)

output_dir = '.'
output_name = 'roc_comparison.pdf'

fig.savefig(os.path.join(output_dir, output_name))
print("Comparison plot saved as %r" % os.path.join(output_dir, output_name))
