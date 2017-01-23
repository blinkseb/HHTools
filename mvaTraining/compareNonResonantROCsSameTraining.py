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

add_parameters_columns = True

style = {
        (1, 1):
        {
            'legend': '$\kappa_\lambda=1, \kappa_t=1$ (SM)',
            'color': '#8E2800'
            },

        (-15, 0.5):
        {
            'legend': '$\kappa_\lambda=-15, \kappa_t=0.5$',
            'color': '#468966'
            },

        (1, 0.5):
        {
            'legend': '$\kappa_\lambda=1, \kappa_t=0.5$',
            'color': '#FFF0A5'
            },

        (5, 2.5):
        {
            'legend': '$\kappa_\lambda=5, \kappa_t=2.5$',
            'color': '#FFB03B'
            },
        }

parameters_list = style.keys()

parser = argparse.ArgumentParser(description='Plot NN output and ROC curve of a training for various masses.')
parser.add_argument('input', metavar='FILE', help='Trained model H5 file', type=str)
args = parser.parse_args()

model = keras.models.load_model(args.input)

suffix = "non_resonant"

output_folder = '.'

dataset = DatasetManager(inputs, weight, cut)
dataset.load_nonresonant_signal(parameters_list=[parameters_list[0]], add_parameters_columns=add_parameters_columns, fraction=1)
dataset.load_backgrounds(add_parameters_columns=add_parameters_columns)

roc_fig = plt.figure(1, figsize=(7, 7))
roc_ax = roc_fig.add_subplot(111)
roc_ax.set_xlabel("Background efficiency", fontsize='large')
roc_ax.set_ylabel("Signal efficiency", fontsize='large')

roc_ax.margins(0.05)
roc_fig.set_tight_layout(True)

auc = []
xlabels = []

for parameters in parameters_list:
    dataset.load_nonresonant_signal(parameters_list=[parameters], add_parameters_columns=add_parameters_columns, fraction=1)
    dataset.update_background_parameters_column()

    dataset.split()

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
    roc_ax.plot(x, y, '-', color=style[parameters]['color'], lw=2, label=style[parameters]['legend'])

    auc.append(metrics.auc(x, y, reorder=True))
    xlabels.append(str(parameters))

    print("Done")

roc_ax.legend(loc='lower right', numpoints=1, frameon=False)

xticks = np.arange(len(xlabels))

auc_fig = plt.figure(2, figsize=(7, 7))
auc_ax = auc_fig.add_subplot(111)
auc_ax.set_ylabel("AUC", fontsize='large')
auc_ax.set_xlabel("Parameters", fontsize='large')

auc_ax.margins(0.05)
auc_fig.set_tight_layout(True)

auc_ax.plot(xticks, auc, linestyle='none', marker='o', color=style.values()[0]['color'], lw=2, markersize=10)
auc_ax.set_xticks(xticks)
auc_ax.set_xticklabels(xlabels)

output_dir = '.'
output_name = 'roc_comparison_same_training_%s.pdf' % suffix

roc_fig.savefig(os.path.join(output_dir, output_name))
auc_fig.savefig(os.path.join(output_dir, 'auc_vs_signal_parameters_%s.pdf' % suffix))
print("Comparison plot saved as %r" % os.path.join(output_dir, output_name))
