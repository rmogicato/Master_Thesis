import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from matplotlib.ticker import LogLocator

from missclassification_methods import check_if_files_exist, get_state_dict


def evaluate_model_openset(modelWrapper, path_to_features):
    """
    This function creates an OSCR plot of the passed model.
    This could be done for each hierarchical level, however, the only implementation if for the target classes.

    :param modelWrapper: Model wrapper to be evaluated, should be either an instance of ShareNet or BaseNet
    :param path_to_features: Path where extracted outputs should be saved
    :param modelWrapper_subclass: If a BaseNet model was passed, then this model is used for subclass predictions
    :param modelWrapper_superclass: If a BaseNet model was passed, then this model is used for superclass predictions
    :return:
    """

    histogram_superclass = {}
    histogram_subclass = {}

    super_outputs_file = 'super_outputs.pth'
    sub_outputs_file = 'sub_outputs.pth'
    target_outputs_file = 'target_outputs.pth'
    all_superclasses_file = 'all_superclasses.pth'
    all_subclasses_file = 'all_subclasses.pth'
    all_targets_file = 'all_targets.pth'

    device = torch.device('cuda:' + str(modelWrapper.device_ids[0]))
    # Create the cache directory if it doesn't exist
    os.makedirs(path_to_features, exist_ok=True)

    for ns in set(modelWrapper.supercluster_map.values()):
        histogram_superclass[ns] = []

    for ns in set(modelWrapper.subcluster_map.values()):
        histogram_subclass[ns] = []


    target_outputs = []

    all_targets = []

    use_saved_features = check_if_files_exist(path_to_features)

    if not use_saved_features:
        modelWrapper.model.load_state_dict(get_state_dict(modelWrapper.checkpoint_dir, device=device))
        modelWrapper.model.eval()

        if modelWrapper.model_string == "Target":
            use_single_model = False
        else:
            use_single_model = True
        print("No saved features found, extracting probabilities")

        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for inputs, targets, subclasses, superclasses in modelWrapper.testloader:
                inputs, targets, subclasses, superclasses = inputs.to(
                    'cuda:' + str(modelWrapper.device_ids[0])), targets.to(
                    'cuda:' + str(modelWrapper.device_ids[0])), subclasses.to(
                    'cuda:' + str(modelWrapper.device_ids[0])), superclasses.to(
                    'cuda:' + str(modelWrapper.device_ids[0]))

                if use_single_model:
                    t_output, _, _ = modelWrapper.model(
                        inputs)
                else:
                    t_output = modelWrapper.model(inputs)

                t_probs = softmax(t_output)

                target_outputs.append(t_probs)

                all_targets.append(targets)

            target_outputs = torch.cat(target_outputs, dim=0).cpu()  # Assuming target_outputs is a list of tensors
            all_targets = torch.cat(all_targets, dim=0).cpu()
            torch.save(target_outputs, os.path.join(path_to_features, target_outputs_file))
            torch.save(all_targets, os.path.join(path_to_features, all_targets_file))

    target_scores = torch.load(os.path.join(path_to_features, target_outputs_file)).numpy()
    all_targets = torch.load(os.path.join(path_to_features, all_targets_file)).numpy()
    ccr_targets, fpr_targets = calculate_oscr(all_targets, target_scores)

    return ccr_targets, fpr_targets


def plot_oscr_curves(ccrs, fprs, names, linestyles, colors, results_dir, unknown_level):
    fig, ax = plt.subplots()
    ax.set_title(f"Samples from Unknown {unknown_level}")
    ax.set_xlabel('FPR')
    ax.set_ylabel('CCR')
    os.makedirs(results_dir, exist_ok=True)

    for idx in range(len(ccrs)):
        fpr = fprs[idx]
        ccr = ccrs[idx]
        color = colors[idx]
        name = names[idx]
        linestyle = linestyles[idx]
        plot_single_oscr(fpr, ccr, ax, linestyle=linestyle, color=color, label=name)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False)

    plt.legend()
    plt.savefig(results_dir + f"/EOS_{unknown_level}.pdf")


def calculate_oscr(gt, scores, unk_label=-1):
    """
    This function is taken from https://github.com/AIML-IfI/openset-imagenet-comparison/blob/main/openset_imagenet/util.py

    Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    # get predicted class for known samples
    pred_class = np.argmax(scores, axis=1)[kn]
    correctly_predicted = pred_class == gt[kn]
    target_score = scores[kn][range(kn.sum()), gt[kn]]

    # get maximum scores for unknown samples
    max_score = np.max(scores, axis=1)[unk]

    # Any max score can be a threshold
    thresholds = np.unique(max_score)

    # print(target_score) #HB
    for tau in thresholds:
        # compute CCR value
        val = (correctly_predicted & (target_score >= tau)).sum() / total_kn
        ccr.append(val)

        val = (max_score >= tau).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr


def plot_single_oscr(fpr, ccr, ax, label, linestyle="solid", color="r", scale="semilog"):
    """
    This function is taken from https://github.com/AIML-IfI/openset-imagenet-comparison/blob/main/openset_imagenet/util.py
    and has been slightly adapted
    """
    linewidth = 1.1
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Manual limits
        ax.set_ylim(0.09, 1)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(direction='in')
    elif scale == 'semilog':
        ax.set_xscale('log')
        # Manual limits
        ax.set_ylim(0.0, .8)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # MaxNLocator(6))  #, prune='lower'))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.tick_params(direction='in')
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_ylim(0.0, 0.8)
        # ax.set_xlim(None, None)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # , prune='lower'))
    # Remove fpr=0 since it cause errors with different ccrs and logscale.
    #    if len(x):
    #        non_zero = x != 0
    #        x = x[non_zero]
    #        y = y[non_zero]
    ax.plot(fpr,
            ccr,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            label=label
            )  # marker='2', markersize=1

    ax.grid(True, color="#b5b7ba", linestyle="dashed")
    return ax
