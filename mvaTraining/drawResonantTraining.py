import glob
import os
import pickle
import gzip
import datetime
import argparse

import plotTools

from common import *

import keras

parser = argparse.ArgumentParser(description='Plot NN output and ROC curve of a training for various resonant masses.')
parser.add_argument('input', metavar='FILE', help='Trained model H5 file', type=str)
parser.add_argument('output', metavar='DIR', help='Output directory', type=str)
args = parser.parse_args()

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

signal_masses = [400]
has_mass_column = False

output_folder = args.output
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset = DatasetManager(inputs, weight, cut)
dataset.load_resonant_signal(masses=signal_masses, add_mass_column=has_mass_column, fraction=1)
dataset.load_backgrounds(add_mass_column=has_mass_column)
dataset.split()

model = keras.models.load_model(args.input)

draw_resonant_training_plots(model, dataset, output_folder, split_by_mass=has_mass_column)
