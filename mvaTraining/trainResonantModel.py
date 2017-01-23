import glob
import os
import pickle
import math
import gzip

import plotTools
import datetime
from timeit import default_timer as timer

from common import *

import keras


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

def lr_scheduler(epoch):
    default_lr = 0.001
    drop = 0.1
    epochs_drop = 50.0
    lr = default_lr * math.pow(drop, min(1, math.floor((1 + epoch) / epochs_drop)))

    return lr

add_mass_column = True
# add_mass_column = False

signal_masses = [400, 650, 900]
# signal_masses = [400]

suffix = "dy_estimation_from_BDT_on_GPU_deeper_lr_scheduler_100epochs"
output_suffix = '{:%Y-%m-%d}_{}_{}'.format(datetime.date.today(), '_'.join([str(x) for x in signal_masses]), suffix)
output_folder = os.path.join('hh_resonant_trained_models', output_suffix)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset = DatasetManager(inputs, weight, cut)
dataset.load_resonant_signal(masses=signal_masses, add_mass_column=add_mass_column, fraction=1)
dataset.load_backgrounds(add_mass_column=add_mass_column)
dataset.split()

training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
training_weights = dataset.get_training_combined_weights()

testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
testing_weights = dataset.get_testing_combined_weights()

callbacks = []

output_model_filename = 'hh_resonant_trained_model.h5'
output_model_filename = os.path.join(output_folder, output_model_filename)

callbacks.append(keras.callbacks.ModelCheckpoint(output_model_filename, monitor='val_loss', verbose=False, save_best_only=True, mode='auto'))

# output_logs_folder = os.path.join('hh_resonant_trained_models', 'logs', output_suffix)
# callbacks.append(keras.callbacks.TensorBoard(log_dir=output_logs_folder, histogram_freq=1, write_graph=True, write_images=False))

callbacks.append(keras.callbacks.LearningRateScheduler(lr_scheduler))

training = True
training_time = None

# Load model
if training:
    n_inputs = len(inputs)
    if add_mass_column:
        n_inputs += 1

    model = create_resonant_model(n_inputs)

    batch_size = 5000
    nb_epoch = 100

    start_time = timer()

    history = model.fit(training_dataset, training_targets, sample_weight=training_weights, batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=True, validation_data=(testing_dataset, testing_targets, testing_weights), callbacks=callbacks)

    end_time = timer()
    training_time = datetime.timedelta(seconds=(end_time - start_time))

    save_training_parameters(output_folder, model,
            batch_size=batch_size, nb_epoch=nb_epoch,
            training_time=str(training_time),
            masses=signal_masses,
            with_mass_column=add_mass_column,
            inputs=inputs,
            cut=cut,
            weights=weight)

    plotTools.draw_keras_history(history, output_dir=output_folder, output_name="loss.pdf")

    # Save history
    print("Saving model training history...")
    output_history_filename = 'hh_resonant_trained_model_history.pklz'
    output_history_filename = os.path.join(output_folder, output_history_filename)
    with gzip.open(output_history_filename, 'wb') as f:
        pickle.dump(history.history, f)
    print("Done.")

model = keras.models.load_model(output_model_filename)

draw_resonant_training_plots(model, dataset, output_folder, split_by_mass=add_mass_column)

print("All done. Training time: %s" % str(training_time))
