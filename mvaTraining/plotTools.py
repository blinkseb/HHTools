import numpy as np

import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt
from sklearn import metrics

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import os

def binDataset(dataset, weights, bins, range=None):
    """
    Bin a dataset

    Parameters:
        dataset: a numpy array of data to bin
        weights: data weights
        bins: either a list of bin boundaries or the number of bin to user
        range: The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the range are ignored

    Returns:
        a tuple (hist, errors, bin_edges):
    """

    # First, bin dataset
    hist, bin_edges = np.histogram(dataset, bins=bins, range=range, weights=weights)

    # Bin weights^2 to extract the uncertainty
    squared_errors, _ = np.histogram(dataset, weights=np.square(weights), bins=bin_edges)
    errors = np.sqrt(squared_errors)

    # norm = True
    norm = False

    if norm:
        norm_factor = np.diff(bin_edges) * hist.sum()
        hist /= norm_factor
        errors /= norm_factor

    return (hist, errors, bin_edges)

def drawTrainingTestingComparison(**kwargs):
    training_background_data = kwargs.get('training_background_data')
    training_signal_data = kwargs.get('training_signal_data')
    testing_background_data = kwargs.get('testing_background_data')
    testing_signal_data = kwargs.get('testing_signal_data')

    # Weights, optional
    training_background_weights = kwargs.get('training_background_weights', None)
    training_signal_weights = kwargs.get('training_signal_weights', None)
    testing_background_weights = kwargs.get('testing_background_weights', None)
    testing_signal_weights = kwargs.get('testing_signal_weights', None)

    range = kwargs.get('range')
    bins = kwargs.get('bins', 50)
    x_label = kwargs.get('x_label', '')
    output_file = kwargs.get('output')

    def makeErrorBoxes(ax, xdata, ydata, yerror, bin_width, ec='None', alpha=0.2):
        # Create list for all the error patches
        errorboxes = []

        # Loop over data points; create box from errors at each point
        for xc, yc, ye in zip(xdata, ydata, yerror):
            rect = Rectangle((xc - bin_width / 2, yc - ye), bin_width, 2*ye)
            errorboxes.append(rect)

        # Create patch collection with specified colour/alpha
        pc = PatchCollection(errorboxes, linewidth=0, facecolor='none', edgecolor=ec, hatch='/////', alpha=alpha)

        # Add collection to axes
        ax.add_collection(pc)

    fig = plt.figure(1, figsize=(7, 7), dpi=300)

    # Create an axes instance
    ax = fig.add_subplot(111)

    background_color = '#B64926'
    signal_color = '#468966'

    # Training data
    training_background_histogram, training_background_errors, bin_edges = binDataset(training_background_data, training_background_weights, bins=bins, range=range)
    training_signal_histogram, training_signal_errors, _ = binDataset(training_signal_data, training_signal_weights, bins=bin_edges)

    # Testing data
    testing_background_histogram, testing_background_errors, _ = binDataset(testing_background_data,testing_background_weights, bins=bin_edges)
    testing_signal_histogram, testing_signal_errors, _ = binDataset(testing_signal_data,testing_signal_weights, bins=bin_edges)

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    ax.bar(bin_centers, training_background_histogram, lw=0, align='center', alpha=0.5, label='Background (training)', color=background_color, width=bin_width)
    makeErrorBoxes(ax, bin_centers, training_background_histogram, training_background_errors, bin_width, ec=background_color, alpha=0.7)

    ax.bar(bin_centers, training_signal_histogram, lw=0, align='center', alpha=0.5, label='Signal (training)', color=signal_color, width=bin_width)
    makeErrorBoxes(ax, bin_centers, training_signal_histogram, training_signal_errors, bin_width, ec=signal_color, alpha=0.7)

    ax.errorbar(bin_centers, testing_background_histogram, yerr=testing_background_errors, linestyle='', marker='o', mew=0, mfc=background_color, ecolor=background_color, label='Background (testing)')
    ax.errorbar(bin_centers, testing_signal_histogram, yerr=testing_signal_errors, linestyle='', marker='o', mew=0, mfc=signal_color, ecolor=signal_color, label='Signal (testing)')

    ax.margins(x=0.1)
    ax.set_ylim(ymin=0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, numpoints=1, loc='best', frameon=False)

    ax.set_xlabel(x_label)

    fig.set_tight_layout(True)

    fig.savefig(output_file)

    plt.close()

    # return n_background, n_signal
    
def drawNNOutput(background_training_predictions, background_testing_predictions,
                 signal_training_predictions, signal_testing_predictions,
                 background_training_weights, background_testing_weights,
                 signal_training_weights, signal_testing_weights,
                 output_dir=".", output_name="nn_output.pdf"
                 ):

    drawTrainingTestingComparison(
            training_background_data=background_training_predictions,
            training_signal_data=signal_training_predictions,
            testing_background_data=background_testing_predictions,
            testing_signal_data=signal_testing_predictions,

            training_background_weights=background_training_weights,
            training_signal_weights=signal_training_weights,
            testing_background_weights=background_testing_weights,
            testing_signal_weights=signal_testing_weights,

            range=[0, 1],
            x_label="NN output",
            output=os.path.join(output_dir, output_name)
            )

def get_roc(signal, background):
    """
    Compute ROC

    Arguments:
        signal, background: an array of discriminant, one for each event

    Return:
        x, y
    """

    n_points = len(signal)

    def get_efficiency(data, from_):
        # print data
        skimmed_data = data[from_:]
        # print np.sum(skimmed_data), np.sum(data), np.sum(skimmed_data) / np.sum(data)
        return np.sum(skimmed_data) / np.sum(data)

    x = []
    y = []
    for i in range(n_points):
        y.append(get_efficiency(signal, i))
        x.append(get_efficiency(background, i))

    x = np.asarray(x)
    y = np.asarray(y)

    order = np.lexsort((y, x))
    x, y = x[order], y[order]

    return x, y

def draw_roc(signal, background, output_dir=".", output_name="roc.pdf"):
    """
    Draw a ROC curve

    Arguments:
    signal, background: an array of discriminant, one for each event
    """

    x, y = get_roc(signal, background)
    auc = metrics.auc(x, y, reorder=True)

    fig = plt.figure(1, figsize=(7, 7), dpi=300)
    fig.clear()

    # Create an axes instance
    ax = fig.add_subplot(111)

    ax.plot(x, y, '-', color='#B64926', lw=2, label="AUC: %.4f" % auc)
    ax.margins(0.05)

    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    
    fig.set_tight_layout(True)

    ax.legend(loc='lower right', numpoints=1, frameon=False)

    print("AUC: %.4f" % auc)

    fig.savefig(os.path.join(output_dir, output_name))

    plt.close()

    def get_index(y, value):
        """
        Find the last index of the element in y
        satistying y[index] <= value
        """

        for i in range(len(y)):
            if y[i] <= value:
                continue

            return i

    print("Background efficiency for signal efficiency of 0.70: %f" % x[get_index(y, 0.70)])
    print("Background efficiency for signal efficiency of 0.80: %f" % x[get_index(y, 0.80)])
    print("Background efficiency for signal efficiency of 0.90: %f" % x[get_index(y, 0.90)])

def draw_keras_history(history, output_dir='.', output_name='loss.pdf'):
    """
    Plot loss value for training and validation samples

    Argument:
      history:  Keras training history
    """

    fig = plt.figure(1, figsize=(7, 7), dpi=300)
    fig.clear()

    # Create an axes instance
    ax = fig.add_subplot(111)

    training_losses = history.history['loss']
    validation_losses = history.history['val_loss']
    epochs = np.arange(0, len(training_losses))

    l1 = ax.plot(epochs, training_losses, '-', color='#8E2800', lw=2, label="Training loss")
    l2 = ax.plot(epochs, validation_losses, '-', color='#468966', lw=2, label="Validation loss")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    
    # training_acc = history.history['acc']
    # validation_acc = history.history['val_acc']

    # ax2 = ax.twinx()
    # l3 = ax2.plot(epochs, training_acc, '--', color='#8E2800', lw=2, label="Training accuracy")
    # l4 = ax2.plot(epochs, validation_acc, '--', color='#468966', lw=2, label="Validation accuracy")
    # ax2.set_ylabel("Accuracy")

    ax.margins(0.05)
    # ax2.margins(0.05)

    fig.set_tight_layout(True)

    # lns = l1 + l2 + l3 + l4
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best', numpoints=1, frameon=False)

    fig.savefig(os.path.join(output_dir, output_name))

    plt.close()
