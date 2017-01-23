import argparse
import os
import re

import plotTools
from common import *
import keras

import matplotlib.pyplot as plt

from sklearn import metrics

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

add_mass_column = True
signal_masses = [400, 650, 900]

style = {
        400:
        {
            'legend': 'M=400 GeV',
            'color': '#8E2800'
            },

        650:
        {
            'legend': 'M=650 GeV',
            'color': '#468966'
            },

        900:
        {
            'legend': 'M=900 GeV',
            'color': '#FFB03B'
            },
        }


parser = argparse.ArgumentParser(description='Plot NN output and ROC curve of a training for various masses.')
parser.add_argument('input', metavar='FILE', help='Trained model H5 file', type=str)
args = parser.parse_args()

model = keras.models.load_model(args.input)

# Extract suffix from input name
# input = hh_(non)resonant_trained_model_<suffix>.h5
# pattern = re.compile(r"hh_(?:non)?resonant_trained_model_(.*)\.h5$")
# groups = pattern.search(args.input)

# suffix = groups.group(1)
suffix = "test"

output_folder = '.'

dataset = DatasetManager(inputs, weight, cut)
dataset.load_resonant_signal(masses=[400], add_mass_column=True)
dataset.load_backgrounds(add_mass_column=True)

roc_fig = plt.figure(1, figsize=(7, 7))
roc_ax = roc_fig.add_subplot(111)
roc_ax.set_xlabel("Background efficiency", fontsize='large')
roc_ax.set_ylabel("Signal efficiency", fontsize='large')

roc_ax.margins(0.05)
roc_fig.set_tight_layout(True)

auc = []

for mass in signal_masses:
    dataset.load_resonant_signal(masses=[mass], add_mass_column=True)
    dataset.update_background_mass_column()

    dataset.split()

    # training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
    # training_weights = dataset.get_training_combined_weights()

    # testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
    # testing_weights = dataset.get_testing_combined_weights()

    print("Evaluating model performances...")

    training_signal_weights, training_background_weights = dataset.get_training_weights()
    testing_signal_weights, testing_background_weights = dataset.get_testing_weights()

    training_signal_predictions, testing_signal_predictions = dataset.get_training_testing_signal_predictions(model)
    training_background_predictions, testing_background_predictions = dataset.get_training_testing_background_predictions(model)

    signal_predictions = np.concatenate((testing_signal_predictions, training_signal_predictions), axis=0)
    background_predictions = np.concatenate((testing_background_predictions, training_background_predictions), axis=0)

    n_signal, _, binning = plotTools.binDataset(signal_predictions, dataset.get_signal_weights(), bins=100, range=[0, 1])
    n_background, _, _ = plotTools.binDataset(background_predictions, dataset.get_background_weights(), bins=binning)

    x, y = plotTools.get_roc(n_signal, n_background)
    roc_ax.plot(x, y, '-', color=style[mass]['color'], lw=2, label=style[mass]['legend'])

    auc.append(metrics.auc(x, y, reorder=True))

    print("Done")

roc_ax.legend(loc='lower right', numpoints=1, frameon=False)

auc_fig = plt.figure(2, figsize=(7, 7))
auc_ax = auc_fig.add_subplot(111)
auc_ax.set_ylabel("AUC", fontsize='large')
auc_ax.set_xlabel("Signal mass hypothesis", fontsize='large')

auc_ax.margins(0.05)
auc_fig.set_tight_layout(True)

auc_ax.plot(signal_masses, auc, linestyle='-', marker='o', color=style[650]['color'], lw=2, markersize=10)

output_dir = '.'
output_name = 'roc_comparison_same_training_%s.pdf' % suffix

roc_fig.savefig(os.path.join(output_dir, output_name))
auc_fig.savefig(os.path.join(output_dir, 'auc_vs_signal_masses_%s.pdf' % suffix))
print("Comparison plot saved as %r" % os.path.join(output_dir, output_name))
