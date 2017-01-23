import glob
import os
import re
import math
import socket
import json

import array
import numpy as np
import numpy.lib.recfunctions as recfunc

from ROOT import TChain, TFile, TTree

from root_numpy import tree2array

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, safe_indexing

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, merge
from keras.layers.core import Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
# from keras.utils.visualize_util import plot

from keras import backend as K
import tensorflow as tf

import plotTools

INPUT_FOLDER = '/home/fynu/sbrochet/scratch/Framework/CMSSW_8_0_24_patch1_HH_Analysis/src/cp3_llbb/HHTools/mvaTraining/inputs/2017-01-10_with_dy_estimate_from_bdt'

HAVE_GPU = 'ingrid-ui8' in socket.gethostname()

def format_nonresonant_parameters(param):
    kl = str(param[0])
    kt = str(param[1])
    X_Y =  (kl + "_" + kt).replace(".", "p")

    return X_Y

backgrounds = [
        {
            'input': 'TTTo2L2Nu_13TeV-powheg_*_histos.root',
        },

        {
            'input': 'DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_*_histos.root',
        },

        {
            'input': 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_*_histos.root',
        },

        # {
            # 'inputs': ['ST_tW_top_5f_inclusiveDecays_13TeV-powheg_*_histos.root'],
        # },

        # {
            # 'inputs': ['ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg_*_histos.root'],
        # },
        
        ]

resonant_signals = {}
resonant_signal_masses = [400, 650, 900]
for m in resonant_signal_masses:
    resonant_signals[m] = 'GluGluToRadionToHHTo2B2VTo2L2Nu_M-%d_narrow_Spring16MiniAODv2_*_histos.root' % m

# Key is k_lambda, k_top
nonresonant_parameters = [(kl, kt) for kl in [-15, -5, -1, 0.0001, 1, 5, 15] for kt in [0.5, 1, 1.75, 2.5]]

# Create map of signal points
nonresonant_signals = {}
for grid_point in nonresonant_parameters:
    X_Y = format_nonresonant_parameters(grid_point)
    nonresonant_signals[grid_point] = 'GluGluToHHTo2B2VTo2L2Nu_base_*_point_{}_13TeV-madgraph_*_histos.root'.format(X_Y)

def make_parallel(model, gpu_count):
    """
    Make a Keras model multi-GPU ready
    """

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1] // parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1] // parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0, name="TEST"))
            
    new_model = Model(input=model.inputs, output=merged)

    funcType = type(model.save)
    # Save the original model instead of multi-GPU compatible one
    def save(self_, filepath, overwrite=True):
        model.save(filepath, overwrite)

    new_model.save = funcType(save, new_model)
    return new_model

def create_resonant_model(n_inputs):
    # Define the model
    model = Sequential()
    model.add(Dense(100, input_dim=n_inputs, activation="relu", init="glorot_uniform"))

    n_hidden_layers = 4
    for i in range(n_hidden_layers):
        model.add(Dense(100, activation="relu", init='glorot_uniform'))
        # if i != (n_hidden_layers - 1):
            # model.add(Dropout(0.1))

    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax', init='glorot_uniform'))

    # optimizer = Adam(lr=0.000005)
    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    return model

def create_nonresonant_model(n_inputs, multi_gpu=False):

    # Define the model
    model = Sequential()
    model.add(Dense(128, input_dim=n_inputs, init="glorot_uniform"))
    model.add(LeakyReLU(alpha=0.2))

    n_hidden_layers = 10
    for i in range(n_hidden_layers):
        model.add(Dense(128, init='glorot_uniform'))
        model.add(LeakyReLU(alpha=0.2))
        if i != (n_hidden_layers - 1):
            model.add(Dropout(0.1))

    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax', init='glorot_uniform'))

    if HAVE_GPU and multi_gpu:
        model = make_parallel(model, 2)

    # optimizer = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    return model

def join_expression(*exprs):
    if len(exprs) == 0:
        return ""
    elif len(exprs) == 1:
        return exprs[0]
    else:
        total_expr = "("
        for expr in exprs:
            expr = expr.strip().strip("*")
            if len(expr) == 0:
                continue
            total_expr += "(" + expr + ")*" 
        total_expr = total_expr.strip("*") + ")"
        return total_expr

def skim_arrays(*arrays, **options):
    """
    Randomly select entries from arrays
    """
    rs = np.random.RandomState(42)

    fraction = options.pop('fraction', 1)

    if fraction != 1:
        n_entries = int(math.floor(fraction * len(arrays[0])))

        indices = np.arange(len(arrays[0]))
        rs.shuffle(indices)
        indices = indices[:n_entries]

        return [safe_indexing(a, indices) for a in arrays]

    return arrays

def tree_to_numpy(input_file, variables, weight, cut=None, reweight_to_cross_section=False, n=None):
    """
    Convert a ROOT TTree to a numpy array.
    """

    file_handle = TFile.Open(input_file)

    tree = file_handle.Get('t')

    cross_section = 1
    relative_weight = 1
    if reweight_to_cross_section:
        cross_section = file_handle.Get('cross_section').GetVal()
        relative_weight = cross_section / file_handle.Get("event_weight_sum").GetVal()

    # relative_weight = cross_section / file_handle.Get("event_weight_sum").GetVal()

    if isinstance(weight, dict):
        # Keys are regular expression and values are actual weights. Find the key matching
        # the input filename
        found = False
        weight_expr = None
        if '__base__' in weight:
            weight_expr = weight['__base__']

        for k, v in weight.items():
            if k == '__base__':
                continue

            groups = re.search(k, input_file)
            if not groups:
                continue
            else:
                if found:
                    raise Exception("The input file is matched by more than one weight regular expression. %r" % input_file)

                found = True
                weight_expr = join_expression(weight_expr, v)

        if not weight_expr:
            raise Exception("Not weight expression found for input file %r" % weight_expr)

        weight = weight_expr

    # Read the tree and convert it to a numpy structured array
    a = tree2array(tree, branches=variables + [weight], selection=cut)

    # Rename the last column to 'weight'
    a.dtype.names = variables + ['weight']

    dataset = a[variables]
    weights = a['weight'] * relative_weight

    # Convert to plain numpy arrays
    # dataset = dataset.view((dataset.dtype[0], len(variables))).copy()
    dataset = np.array(dataset.tolist(), dtype=np.float32)

    if n:
        print("Reading only {} from input tree".format(n))
        dataset = dataset[:n]
        weights = weights[:n]

    return dataset, weights


class DatasetManager:
    def __init__(self, variables, weight_expression, selection):
        """
        Create a new dataset manager

        Parameters:
            variables: list of input variables. This can be either branch names or a more
              complexe mathematical expression
            selection: a cut expression applied to each event. Only events passing this selection are
              kept
            weight_expression: either:
              - a string representing a global weight expression
              - a dict. Keys are regular expression, and values are weight expression. The weight
                  expression chosen will be the one where the keys matches the input file name.
        """

        self.variables = variables
        self.selection = selection
        self.weight_expression = weight_expression

        self.resonant_mass_probabilities = None
        self.nonresonant_parameters_probabilities = None

        self.resonant_masses = None
        self.nonresonant_parameters_list = None

    def load_resonant_signal(self, masses=resonant_signal_masses, add_mass_column=False, fraction=1):
        """
        Load resonant signal

        Parameters:
          masses: list of masses to load
          add_mass_column: if True, a column is added at the end of the dataset with the mass of the sample
          fraction: the fraction of signal events to keep. Default to 1, ie keeping all the events
        """

        self.resonant_masses = masses
        self.n_extra_columns = 1 if add_mass_column else 0

        datasets = []
        weights = []
        p = [[], []]

        print("Loading resonant signal...")

        for m in masses:
            f = get_file_from_glob(os.path.join(INPUT_FOLDER, resonant_signals[m]))
            dataset, weight = tree_to_numpy(f, self.variables, self.weight_expression, self.selection, reweight_to_cross_section=False)

            if fraction != 1:
                dataset, weight = skim_arrays(dataset, weight, fraction=fraction)

            p[0].append(m)
            p[1].append(len(dataset))

            if add_mass_column:
                mass_col = np.empty(len(dataset)) # dtype=[('resonant_mass', '<f4')])
                mass_col.fill(m)
                dataset = np.c_[dataset, mass_col]

            datasets.append(dataset)
            weights.append(weight)

        # Normalize probabilities in order that sum(p) = 1
        p[1] = np.array(p[1], dtype='float')
        p[1] /= np.sum(p[1])

        self.signal_dataset = np.concatenate(datasets)
        self.signal_weights = np.concatenate(weights)
        self.resonant_mass_probabilities = p

        print("Done. Number of signal events: %d ; Sum of weights: %.4f" % (len(self.signal_dataset), np.sum(self.signal_weights)))

    def load_nonresonant_signal(self, parameters_list=nonresonant_parameters, add_parameters_columns=False, fraction=1):

        self.nonresonant_parameters_list = parameters_list
        self.n_extra_columns = 2 if add_parameters_columns else 0

        datasets = []
        weights = []
        p = [[], []]

        print("Loading nonresonant signal...")

        for parameters in parameters_list:
            print("For parameters: {}".format(parameters))
            files = get_files_from_glob(os.path.join(INPUT_FOLDER, nonresonant_signals[parameters]))

            dataset = []
            weight = []
            for f in files:
                dataset_, weight_ = tree_to_numpy(f, self.variables, self.weight_expression, self.selection, reweight_to_cross_section=False)
                weight.append(weight_)
                dataset.append(dataset_)

            dataset = np.concatenate(dataset)
            weight = np.concatenate(weight)

            if fraction != 1:
                dataset, weight = skim_arrays(dataset, weight, fraction=fraction)

            p[0].append(parameters)
            p[1].append(len(dataset))

            if add_parameters_columns:
                for parameter in parameters:
                    col = np.empty(len(dataset))
                    col.fill(parameter)
                    dataset = np.c_[dataset, col]

            datasets.append(dataset)
            weights.append(weight)

        # Normalize probabilities in order that sum(p) = 1
        p[1] = np.array(p[1], dtype='float')
        p[1] /= np.sum(p[1])

        self.signal_dataset = np.concatenate(datasets)
        self.signal_weights = np.concatenate(weights)
        self.nonresonant_parameters_probabilities = p

        print("Done. Number of signal events: %d ; Sum of weights: %.4f" % (len(self.signal_dataset), np.sum(self.signal_weights)))

    def load_backgrounds(self, add_mass_column=False, add_parameters_columns=False):

        if add_mass_column and add_parameters_columns:
            raise Exception("add_mass_column and add_parameters_columns are mutually exclusive")

        if add_mass_column and not self.resonant_mass_probabilities:
            raise Exception("You need to first load the resonant signal before the background")

        if add_parameters_columns and not self.nonresonant_parameters_probabilities:
            raise Exception("You need to first load the nonresonant signals before the background")

        datasets = []
        weights = []

        print("Loading backgrounds...")

        for background in backgrounds:
            f = get_file_from_glob(os.path.join(INPUT_FOLDER, background['input']))
            dataset, weight = tree_to_numpy(f, self.variables, self.weight_expression, self.selection, reweight_to_cross_section=True)
            weights.append(weight)

            probabilities = None
            if add_mass_column:
                probabilities = self.resonant_mass_probabilities
            elif add_parameters_columns:
                probabilities = self.nonresonant_parameters_probabilities

            if probabilities:
                rs = np.random.RandomState(42)

                indices = rs.choice(len(probabilities[0]), len(dataset), p=probabilities[1])
                cols = np.array(np.take(probabilities[0], indices, axis=0), dtype='float')
                dataset = np.c_[dataset, cols]

            datasets.append(dataset)

        self.background_dataset = np.concatenate(datasets)
        self.background_weights = np.concatenate(weights)

        print("Done. Number of background events: %d ; Sum of weights: %.4f" % (len(self.background_dataset), np.sum(self.background_weights)))

    def update_background_mass_column(self):
        rs = np.random.RandomState(42)
        mass_col = np.array(rs.choice(self.resonant_mass_probabilities[0], len(self.background_dataset), p=self.resonant_mass_probabilities[1]), dtype='float')

        self.background_dataset[:, len(self.variables)] = mass_col

    def update_background_parameters_column(self):
        rs = np.random.RandomState(42)
        probabilities = self.nonresonant_parameters_probabilities

        indices = rs.choice(len(probabilities[0]), len(self.background_dataset), p=probabilities[1])
        cols = np.array(np.take(probabilities[0], indices, axis=0), dtype='float')

        for i in range(len(probabilities[0])):
            self.background_dataset[:, len(self.variables) + i] = cols[:, i]

    def split(self, reweight_background_training_sample=True):
        """
        Split datasets into a training and testing samples

        Parameter:
            reweight_background_training_sample: If true, the background training sample is reweighted so that the sum of weights of signal and background are the same
        """

        self.train_signal_dataset, self.test_signal_dataset, self.train_signal_weights, self.test_signal_weights = train_test_split(self.signal_dataset, self.signal_weights, test_size=0.5, random_state=42)
        self.train_background_dataset, self.test_background_dataset, self.train_background_weights, self.test_background_weights = train_test_split(self.background_dataset, self.background_weights, test_size=0.5, random_state=42)

        if reweight_background_training_sample:
            sumw_train_signal = np.sum(self.train_signal_weights)
            sumw_train_background = np.sum(self.train_background_weights)

            ratio = sumw_train_signal / sumw_train_background
            self.train_background_weights *= ratio
            self.test_background_weights *= ratio
            print("Background training sample reweighted so that sum of event weights for signal and background match. Sum of event weight = %.4f" % (np.sum(self.train_signal_weights)))

        # Create merged training and testing dataset, with targets
        self.training_dataset = np.concatenate([self.train_signal_dataset, self.train_background_dataset])
        self.training_weights = np.concatenate([self.train_signal_weights, self.train_background_weights])
        self.testing_dataset = np.concatenate([self.test_signal_dataset, self.test_background_dataset])
        self.testing_weights = np.concatenate([self.test_signal_weights, self.test_background_weights])

        # Create one-hot vector, the target of the training
        # A hot-vector is a N dimensional vector, where N is the number of classes
        # Here we assume that class 0 is signal, and class 1 is background
        # So we have [1 0] for signal and [0 1] for background
        self.training_targets = np.array([[1, 0]] * len(self.train_signal_dataset) + [[0, 1]] * len(self.train_background_dataset))
        self.testing_targets = np.array([[1, 0]] * len(self.test_signal_dataset) + [[0, 1]] * len(self.test_background_dataset))

        # Shuffle everything
        self.training_dataset, self.training_weights, self.training_targets = shuffle(self.training_dataset, self.training_weights, self.training_targets, random_state=42)
        self.testing_dataset, self.testing_weights, self.testing_targets = shuffle(self.testing_dataset, self.testing_weights, self.testing_targets, random_state=42)

    def get_training_datasets(self):
        return self.train_signal_dataset, self.train_background_dataset

    def get_testing_datasets(self):
        return self.test_signal_dataset, self.test_background_dataset

    def get_training_weights(self):
        return self.train_signal_weights, self.train_background_weights

    def get_testing_weights(self):
        return self.test_signal_weights, self.test_background_weights

    def get_training_combined_dataset_and_targets(self):
        return self.training_dataset, self.training_targets

    def get_testing_combined_dataset_and_targets(self):
        return self.testing_dataset, self.testing_targets

    def get_training_combined_weights(self):
        return self.training_weights

    def get_testing_combined_weights(self):
        return self.testing_weights

    def get_training_testing_signal_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.train_signal_dataset, **kwargs), self._get_predictions(model, self.test_signal_dataset, **kwargs)

    def get_signal_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.signal_dataset, **kwargs)

    def get_signal_weights(self):
        return self.signal_weights

    def get_training_testing_background_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.train_background_dataset, **kwargs), self._get_predictions(model, self.test_background_dataset, **kwargs)

    def get_background_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.background_dataset, **kwargs)

    def get_background_weights(self):
        return self.background_weights

    def _get_predictions(self, model, values, **kwargs):
        ignore_n_last_columns = kwargs.get('ignore_last_columns', 0)
        if ignore_n_last_columns > 0:
            values = values[:, :-ignore_n_last_columns]
        predictions = model.predict(values, batch_size=5000, verbose=1)
        return np.delete(predictions, 1, axis=1).flatten()

    def draw_inputs(self, output_dir):

        print("Plotting input variables...")
        variables = self.variables[:]

        if self.n_extra_columns > 0:
            if self.resonant_masses:
                variables += ['mass_hypothesis']
            elif self.nonresonant_parameters_list:
                variables += ['k_l', 'k_t']

        for index, variable in enumerate(variables):
            output_file = os.path.join(output_dir, variable + ".pdf") 
            plotTools.drawTrainingTestingComparison(
                    training_background_data=self.train_background_dataset[:, index],
                    training_signal_data=self.train_signal_dataset[:, index],
                    testing_background_data=self.test_background_dataset[:, index],
                    testing_signal_data=self.test_signal_dataset[:, index],

                    training_background_weights=self.train_background_weights,
                    training_signal_weights=self.train_signal_weights,
                    testing_background_weights=self.test_background_weights,
                    testing_signal_weights=self.test_signal_weights,

                    x_label=variable,
                    output=output_file
                    )

        print("Done.")


def get_file_from_glob(f):
    files = glob.glob(f)
    if len(files) != 1:
        raise Exception('Only one input file is supported per glob pattern: %s' % files)

    return files[0]

def get_files_from_glob(f):
    files = glob.glob(f)
    if len(files) == 0:
        raise Exception('No file matching glob pattern: %s' % f)

    return files

def draw_non_resonant_training_plots(model, dataset, output_folder, split_by_parameters=False):

    # plot(model, to_file=os.path.join(output_folder, "model.pdf"))

    # Draw inputs
    output_input_plots = os.path.join(output_folder, 'inputs')
    if not os.path.exists(output_input_plots):
        os.makedirs(output_input_plots)

    dataset.draw_inputs(output_input_plots)

    training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
    training_weights = dataset.get_training_combined_weights()

    testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
    testing_weights = dataset.get_testing_combined_weights()

    print("Evaluating model performances...")

    training_signal_weights, training_background_weights = dataset.get_training_weights()
    testing_signal_weights, testing_background_weights = dataset.get_testing_weights()

    training_signal_predictions, testing_signal_predictions = dataset.get_training_testing_signal_predictions(model)
    training_background_predictions, testing_background_predictions = dataset.get_training_testing_background_predictions(model)

    print("Done.")

    print("Plotting time...")
    # NN output
    plotTools.drawNNOutput(training_background_predictions, testing_background_predictions,
                 training_signal_predictions, testing_signal_predictions,
                 training_background_weights, testing_background_weights,
                 training_signal_weights, testing_signal_weights,
                 output_dir=output_folder, output_name="nn_output.pdf")

    # ROC curve
    binned_training_background_predictions, _, bins = plotTools.binDataset(training_background_predictions, training_background_weights, bins=50, range=[0, 1])
    binned_training_signal_predictions, _, _ = plotTools.binDataset(training_signal_predictions, training_signal_weights, bins=bins)
    plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve.pdf")

    if split_by_parameters:
        output_folder = os.path.join(output_folder, 'splitted_by_parameters')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        training_signal_dataset, training_background_dataset = dataset.get_training_datasets()
        testing_signal_dataset, testing_background_dataset = dataset.get_testing_datasets()
        for parameters in dataset.nonresonant_parameters_list:
            print("  Plotting NN output and ROC curve for %s" % str(parameters))

            training_signal_mask = (training_signal_dataset[:,-1] == parameters[1]) & (training_signal_dataset[:,-2] == parameters[0])
            training_background_mask = (training_background_dataset[:,-1] == parameters[1]) & (training_background_dataset[:,-2] == parameters[0])
            testing_signal_mask = (testing_signal_dataset[:,-1] == parameters[1]) & (testing_signal_dataset[:,-2] == parameters[0])
            testing_background_mask = (testing_background_dataset[:,-1] == parameters[1]) & (testing_background_dataset[:,-2] == parameters[0])

            p_training_background_predictions = training_background_predictions[training_background_mask]
            p_testing_background_predictions = testing_background_predictions[testing_background_mask]
            p_training_signal_predictions = training_signal_predictions[training_signal_mask]
            p_testing_signal_predictions = testing_signal_predictions[testing_signal_mask]

            p_training_background_weights = training_background_weights[training_background_mask]
            p_testing_background_weights = testing_background_weights[testing_background_mask]
            p_training_signal_weights = training_signal_weights[training_signal_mask]
            p_testing_signal_weights = testing_signal_weights[testing_signal_mask]

            suffix = format_nonresonant_parameters(parameters)
            plotTools.drawNNOutput(
                         p_training_background_predictions, p_testing_background_predictions,
                         p_training_signal_predictions, p_testing_signal_predictions,
                         p_training_background_weights, p_testing_background_weights,
                         p_training_signal_weights, p_testing_signal_weights,
                         output_dir=output_folder, output_name="nn_output_fixed_parameters_%s.pdf" % (suffix))

            binned_training_background_predictions, _, bins = plotTools.binDataset(p_training_background_predictions, p_training_background_weights, bins=50, range=[0, 1])
            binned_training_signal_predictions, _, _ = plotTools.binDataset(p_training_signal_predictions, p_training_signal_weights, bins=bins)
            plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve_fixed_parameters_%s.pdf" % (suffix))
    print("Done")

def draw_resonant_training_plots(model, dataset, output_folder, split_by_mass=False):

    # plot(model, to_file=os.path.join(output_folder, "model.pdf"))

    # Draw inputs
    output_input_plots = os.path.join(output_folder, 'inputs')
    if not os.path.exists(output_input_plots):
        os.makedirs(output_input_plots)

    dataset.draw_inputs(output_input_plots)

    training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
    training_weights = dataset.get_training_combined_weights()

    testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
    testing_weights = dataset.get_testing_combined_weights()

    print("Evaluating model performances...")

    training_signal_weights, training_background_weights = dataset.get_training_weights()
    testing_signal_weights, testing_background_weights = dataset.get_testing_weights()

    training_signal_predictions, testing_signal_predictions = dataset.get_training_testing_signal_predictions(model)
    training_background_predictions, testing_background_predictions = dataset.get_training_testing_background_predictions(model)

    print("Done.")

    print("Plotting time...")
    # NN output
    plotTools.drawNNOutput(training_background_predictions, testing_background_predictions,
                 training_signal_predictions, testing_signal_predictions,
                 training_background_weights, testing_background_weights,
                 training_signal_weights, testing_signal_weights,
                 output_dir=output_folder, output_name="nn_output.pdf")

    # ROC curve
    binned_training_background_predictions, _, bins = plotTools.binDataset(training_background_predictions, training_background_weights, bins=50, range=[0, 1])
    binned_training_signal_predictions, _, _ = plotTools.binDataset(training_signal_predictions, training_signal_weights, bins=bins)
    plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve.pdf")

    if split_by_mass:
        output_folder = os.path.join(output_folder, 'splitted_by_mass')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        training_signal_dataset, training_background_dataset = dataset.get_training_datasets()
        testing_signal_dataset, testing_background_dataset = dataset.get_testing_datasets()
        for m in dataset.resonant_masses:
            print("  Plotting NN output and ROC curve for M=%d" % m)

            training_signal_mask = training_signal_dataset[:,-1] == m
            training_background_mask = training_background_dataset[:,-1] == m
            testing_signal_mask = testing_signal_dataset[:,-1] == m
            testing_background_mask = testing_background_dataset[:,-1] == m

            p_training_background_predictions = training_background_predictions[training_background_mask]
            p_testing_background_predictions = testing_background_predictions[testing_background_mask]
            p_training_signal_predictions = training_signal_predictions[training_signal_mask]
            p_testing_signal_predictions = testing_signal_predictions[testing_signal_mask]

            p_training_background_weights = training_background_weights[training_background_mask]
            p_testing_background_weights = testing_background_weights[testing_background_mask]
            p_training_signal_weights = training_signal_weights[training_signal_mask]
            p_testing_signal_weights = testing_signal_weights[testing_signal_mask]

            plotTools.drawNNOutput(
                         p_training_background_predictions, p_testing_background_predictions,
                         p_training_signal_predictions, p_testing_signal_predictions,
                         p_training_background_weights, p_testing_background_weights,
                         p_training_signal_weights, p_testing_signal_weights,
                         output_dir=output_folder, output_name="nn_output_fixed_M%d.pdf" % (m))

            binned_training_background_predictions, _, bins = plotTools.binDataset(p_training_background_predictions, p_training_background_weights, bins=50, range=[0, 1])
            binned_training_signal_predictions, _, _ = plotTools.binDataset(p_training_signal_predictions, p_training_signal_weights, bins=bins)
            plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve_fixed_M_%d.pdf" % (m))
    print("Done")

def save_training_parameters(output, model, **kwargs):
    parameters = {
            'extra': kwargs
            }

    model_definition = model.to_json()
    m = json.loads(model_definition)
    parameters['model'] = m

    with open(os.path.join(output, 'parameters.json'), 'w') as f:
        json.dump(parameters, f, indent=4)

