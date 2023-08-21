import collections
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches
from sklearn.metrics import confusion_matrix


def find_first_incorrect_classification(logit_child, target_parents, child_parent_map, histogram, original_targets_map):
    """
    :param logit_child: logits of child (lower level) class
    :param target_parents: ground truth to which each target class belongs
    :param child_parent_map: mapping to easily check whether we have the right parent, either a supercluster_map, or subcluster_map
    :param histogram: histogram that is being created
    :param original_targets_map: get original target, necessary to find correct parent
    :return: histogram
    """
    for i in range(logit_child.shape[0]):
        target_parent = int(target_parents[i])
        # sorting the subclass logits so that we get indexes from largest to smallest
        sorted_indices = np.argsort(-logit_child[i]).numpy()
        # go through each value in the array
        for position, target_child in enumerate(sorted_indices):
            # first time the predicted subclass does not correspond to the target add it to the histogram
            # get original target mapping
            original_target = original_targets_map[target_child]
            if child_parent_map[original_target] != target_parent:
                histogram[target_parent].append(position)
                # break since we are only interested in first misclassification
                break
    return histogram



def create_confusion_matrix(target_outputs, all_targets, sub_outputs, all_subclasses, super_outputs, all_superclasses,
                            subcluster_map, subcluster_split, original_targets_map, subclasses_to_text, results_dir,
                            title):
    """
    This function creates confusion matrices.
    In my thesis only the "confusion_matrix_low_level_log.pdf" was used
    :param target_outputs: logits of targets
    :param all_targets: labels of targets
    :param sub_outputs: logits of subclasses
    :param all_subclasses: labels of subclasses
    :param super_outputs: logits of superclasses
    :param all_superclasses: labels of superclasses
    :param subcluster_map: mapping for subclasses
    :param subcluster_split: size of each subcluster
    :param original_targets_map: mapping of original (ImageNet) class ID and newly created mapping
    :param subclasses_to_text: subclass index to readable text
    :param results_dir: for saving resulting graphs
    :param title: Title of graph
    :return:
    """
    probabilities_target = torch.softmax(target_outputs, dim=1)
    predicted_target_classes = torch.argmax(probabilities_target, dim=1)

    probabilities_sub = torch.softmax(sub_outputs, dim=1)
    predicted_sub_classes = torch.argmax(probabilities_sub, dim=1)

    probabilities_super = torch.softmax(super_outputs, dim=1)
    predicted_super_classes = torch.argmax(probabilities_super, dim=1)

    cm_superclasses = confusion_matrix(all_superclasses, predicted_super_classes)
    np.fill_diagonal(cm_superclasses, 0)
    cm_subclasses = confusion_matrix(all_subclasses, predicted_sub_classes)
    np.fill_diagonal(cm_subclasses, 0)

    # mapping the learned classes to the original classes
    new_fusion = {}
    for new_target, old_target in original_targets_map.items():
        new_fusion[new_target] = subcluster_map[old_target]
    sorted_dict = dict(sorted(new_fusion.items(), key=lambda x: x[1]))
    sorted_targets = list(sorted_dict.keys())

    # getting the ranges of each subclass
    subclass_ranges = []
    curr_subclass = list(sorted_dict.values())[0]
    curr_beginning = 0
    for idx, subclass in enumerate(sorted_dict.values()):
        if subclass != curr_subclass:
            subclass_ranges.append((curr_beginning, idx - 1))
            curr_beginning = idx
            curr_subclass = subclass
    subclass_ranges.append((curr_beginning, len(sorted_dict.values()) - 1))

    # getting the ranges of each superclass
    superclass_ranges = []
    prev_idx = 0
    for split in subcluster_split:
        superclass_ranges.append((subclass_ranges[prev_idx][0], subclass_ranges[split + prev_idx - 1][1]))
        prev_idx = prev_idx + split

    cm_target = confusion_matrix(all_targets, predicted_target_classes)
    cm_target = cm_target[sorted_targets]  # Sort rows
    cm_target = cm_target[:, sorted_targets]  # Sort columns

    accuracy_within_subclass = count_values_within_ranges(cm_target, subclass_ranges)
    accuracy_within_superclass = count_values_within_ranges(cm_target, superclass_ranges)
    total = count_values_within_ranges(cm_target, [(superclass_ranges[0][0], superclass_ranges[-1][-1])])

    np.fill_diagonal(cm_target, 0)
    cm_target_log = np.vectorize(np.log1p)(cm_target)

    # Plotting the hierarchical confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Level 1 superclass confusion matrix
    axes[0].imshow(cm_superclasses, interpolation='bilinear', cmap="BuGn")
    axes[0].set_title('Level 1 - Superclasses')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Plot Level 2 subclass confusion matrix
    axes[1].imshow(cm_subclasses, interpolation='bilinear', cmap="BuGn")
    axes[1].set_title('Level 2 - Subclasses')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    fig_target = draw_target_cm(cm_target, superclass_ranges, subclass_ranges, subclasses_to_text)
    fig_target_log = draw_target_cm(cm_target_log, superclass_ranges, subclass_ranges, subclasses_to_text)

    # counting how many misclassifications happen inside the subclass and superclass range
    misclassified_within_subclass = count_values_within_ranges(cm_target, subclass_ranges)
    misclassified_within_superclass = count_values_within_ranges(cm_target, superclass_ranges)
    total_misclassified = count_values_within_ranges(cm_target, [(superclass_ranges[0][0], superclass_ranges[-1][-1])])
    print("misclassified subclasses:", misclassified_within_subclass / total_misclassified)
    print("misclassified superclasses:", misclassified_within_superclass / total_misclassified)

    record_number_of_misclassifications(misclassified_within_superclass / total_misclassified,
                                        misclassified_within_subclass / total_misclassified,
                                        accuracy_within_superclass / total, accuracy_within_subclass / total, title)

    # Adjust plot margin if necessary
    plt.subplots_adjust(left=0.15)

    plt.tight_layout()
    # Save the figures
    fig.savefig(os.path.join(results_dir, 'confusion_matrix_high_level.pdf'))
    fig_target.savefig(os.path.join(results_dir, 'confusion_matrix_low_level.pdf'))
    fig_target_log.savefig(os.path.join(results_dir, 'confusion_matrix_low_level_log.pdf'))


def draw_target_cm(cm_target, superclass_ranges, subclass_ranges, subclasses_to_text):
    """
    This plots the target confusion matrix
    """
    # Define the figure and axis for the last plot
    fig2, ax2 = plt.subplots(figsize=(10, 10))

    # Plot Level 3 target class confusion matrix
    ax2.imshow(cm_target, cmap="Greys")
    ax2.set_title('Confusion Matrix - Target Classes', fontsize=20)
    ax2.set_xlabel('Predicted Label', fontsize=16)
    ax2.set_ylabel('Actual Class', fontsize=16)

    # Draw rectangles
    for boundary in superclass_ranges:
        rect = patches.Rectangle((boundary[0], boundary[0]), boundary[1] - boundary[0] + 1,
                                 boundary[1] - boundary[0] + 1,
                                 linewidth=0.5, edgecolor='b', alpha=0.7, facecolor='none')
        ax2.add_patch(rect)

    for boundary in subclass_ranges:
        rect = patches.Rectangle((boundary[0], boundary[0]), boundary[1] - boundary[0] + 1,
                                 boundary[1] - boundary[0] + 1,
                                 linewidth=0.5, edgecolor='r', alpha=0.7, facecolor='none')
        ax2.add_patch(rect)

    tick_pos = [sublist[0] for sublist in subclass_ranges]

    # ticks = [item for sublist in subclass_ranges for item in sublist]
    ax2.set_yticks(tick_pos)
    ax2.set_yticklabels([])  # Hide y-axis tick labels
    ax2.set_xticklabels([])
    ax2.yaxis.tick_right()
    # Add subclass names
    for idx, boundary in enumerate(subclass_ranges):
        midpoint = (boundary[0] + boundary[1]) / 2
        sub_class_name = subclasses_to_text[idx]
        ax2.text(cm_target.shape[1] + 0.5, midpoint, sub_class_name, ha='left', va='center', fontsize=16)
    plt.tight_layout()
    return fig2


def count_values_within_ranges(matrix, ranges):
    total_count = 0
    for r in ranges:
        total_count += np.sum(matrix[r[0]:r[1]])
    return total_count


def check_if_files_exist(dir_path):
    dir_path = dir_path + "/"
    # Define the file names
    super_outputs_file = 'super_outputs.pth'
    sub_outputs_file = 'sub_outputs.pth'
    target_outputs_file = 'target_outputs.pth'
    all_superclasses_file = 'all_superclasses.pth'
    all_subclasses_file = 'all_subclasses.pth'
    all_targets_file = 'all_targets.pth'

    # Check if the files exist
    super_outputs_exist = os.path.exists(os.path.join(dir_path, super_outputs_file))
    sub_outputs_exist = os.path.exists(os.path.join(dir_path, sub_outputs_file))
    target_outputs_exist = os.path.exists(os.path.join(dir_path, target_outputs_file))
    all_superclasses_exist = os.path.exists(os.path.join(dir_path, all_superclasses_file))
    all_subclasses_exist = os.path.exists(os.path.join(dir_path, all_subclasses_file))
    all_targets_exist = os.path.exists(os.path.join(dir_path, all_targets_file))
    return super_outputs_exist and sub_outputs_exist and target_outputs_exist and all_superclasses_exist and all_subclasses_exist and all_targets_exist


def evaluate_model(modelWrapper, subcluster_split, path_to_features, results_dir, modelWrapper_subclass=None,
                   modelWrapper_superclass=None):
    """
    This function evaluates a model, getting the balanced accuracy scores, creating confusion matrices and returns
    histograms of the first misclassification indices
    :param modelWrapper: modelWrapper to be analyzed. If it's the baseline model, make sure to also pass modelWrapper_subclass and modelWrapper_superclass
    :param subcluster_split: subclass split
    :param path_to_features: where the extracted scores should be saved, this way the predictions don't have to be made multiple times
    :param results_dir: where the plots should be saved
    :param modelWrapper_subclass: subclass model wrapper if baseline model
    :param modelWrapper_superclass: superclass model wrapper if baseline model
    :return: dictionaries of histograms and result dictionary
    """

    assert (modelWrapper_subclass is not None and modelWrapper_superclass is not None) or (
            modelWrapper_subclass is None and modelWrapper_superclass is None), "Either no subclass and superclass models must be passed or both are"

    test_total = 0

    histogram_superclass = {}
    histogram_subclass = {}

    super_outputs_file = 'super_outputs.pth'
    sub_outputs_file = 'sub_outputs.pth'
    target_outputs_file = 'target_outputs.pth'
    all_superclasses_file = 'all_superclasses.pth'
    all_subclasses_file = 'all_subclasses.pth'
    all_targets_file = 'all_targets.pth'

    # Create the cache directory if it doesn't exist
    os.makedirs(path_to_features, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for ns in set(modelWrapper.supercluster_map.values()):
        histogram_superclass[ns] = []

    for ns in set(modelWrapper.subcluster_map.values()):
        histogram_subclass[ns] = []

    super_outputs = []
    sub_outputs = []
    target_outputs = []

    all_superclasses = []
    all_subclasses = []
    all_targets = []

    test_samples_per_target_class = np.zeros(modelWrapper.num_targets, dtype=int)
    test_samples_per_subclass_class = np.zeros(modelWrapper.num_subclasses, dtype=int)
    test_samples_per_superclass_class = np.zeros(modelWrapper.num_superclasses, dtype=int)

    test_correct_per_target_class = np.zeros(modelWrapper.num_targets, dtype=int)
    test_correct_per_subclass_class = np.zeros(modelWrapper.num_subclasses, dtype=int)
    test_correct_per_superclass_class = np.zeros(modelWrapper.num_superclasses, dtype=int)

    super_acc_balanced = []
    sub_acc_balanced = []
    t_acc_balanced = []

    use_saved_features = check_if_files_exist(path_to_features)

    if not use_saved_features:
        modelWrapper.model.eval()
        modelWrapper.model.load_state_dict(get_state_dict(modelWrapper.checkpoint_dir))
        use_single_model = True
        if modelWrapper_subclass is not None and modelWrapper_superclass is not None:
            use_single_model = False
            modelWrapper_subclass.model.load_state_dict(get_state_dict(modelWrapper_subclass.checkpoint_dir))
            modelWrapper_subclass.model.eval()
            modelWrapper_superclass.model.load_state_dict(get_state_dict(modelWrapper_superclass.checkpoint_dir))
            modelWrapper_superclass.model.eval()

        print("No saved features found, extracting probabilities")
        with torch.no_grad():
            for inputs, targets, subclasses, superclasses in modelWrapper.testloader:
                inputs, targets, subclasses, superclasses = inputs.to(
                    'cuda:' + str(modelWrapper.device_ids[0])), targets.to(
                    'cuda:' + str(modelWrapper.device_ids[0])), subclasses.to(
                    'cuda:' + str(modelWrapper.device_ids[0])), superclasses.to(
                    'cuda:' + str(modelWrapper.device_ids[0]))

                if use_single_model:
                    t_output, suboutput, superoutput = modelWrapper.model(
                        inputs)  # In all cases make sure no ground truth labels are provided.
                else:
                    t_output = modelWrapper.model(inputs)
                    suboutput = modelWrapper_subclass.model(inputs)
                    superoutput = modelWrapper_superclass.model(inputs)

                super_outputs.append(superoutput)
                sub_outputs.append(suboutput)
                target_outputs.append(t_output)

                all_superclasses.append(superclasses)
                all_subclasses.append(subclasses)
                all_targets.append(targets)

            super_outputs = torch.cat(super_outputs, dim=0).cpu()  # Assuming super_outputs is a list of tensors
            sub_outputs = torch.cat(sub_outputs, dim=0).cpu()  # Assuming sub_outputs is a list of tensors
            target_outputs = torch.cat(target_outputs, dim=0).cpu()  # Assuming target_outputs is a list of tensors

            # Convert the lists of labels to tensors
            all_superclasses = torch.cat(all_superclasses, dim=0).cpu()
            all_subclasses = torch.cat(all_subclasses, dim=0).cpu()
            all_targets = torch.cat(all_targets, dim=0).cpu()

            super_predictions = torch.softmax(super_outputs, dim=1)
            _, sup_predicted = super_predictions.topk(1, 1) # we only look at top-1 predictions, but this could easily be changed
            test_total += all_superclasses.size(0)

            sub_predictions = torch.softmax(sub_outputs, dim=1)
            _, sub_predicted = sub_predictions.topk(1, 1)

            t_predictions = torch.softmax(target_outputs, dim=1)
            _, t_predicted = t_predictions.topk(1, 1)

            torch.save(super_outputs, os.path.join(path_to_features, super_outputs_file))
            torch.save(sub_outputs, os.path.join(path_to_features, sub_outputs_file))
            torch.save(target_outputs, os.path.join(path_to_features, target_outputs_file))

            torch.save(all_superclasses, os.path.join(path_to_features, all_superclasses_file))
            torch.save(all_subclasses, os.path.join(path_to_features, all_subclasses_file))
            torch.save(all_targets, os.path.join(path_to_features, all_targets_file))

    super_outputs = torch.load(os.path.join(path_to_features, super_outputs_file))
    sub_outputs = torch.load(os.path.join(path_to_features, sub_outputs_file))
    target_outputs = torch.load(os.path.join(path_to_features, target_outputs_file))
    all_superclasses = torch.load(os.path.join(path_to_features, all_superclasses_file))
    all_subclasses = torch.load(os.path.join(path_to_features, all_subclasses_file))
    all_targets = torch.load(os.path.join(path_to_features, all_targets_file))

    super_predictions = torch.softmax(super_outputs, dim=1)
    _, sup_predicted = super_predictions.topk(1, 1)

    sub_predictions = torch.softmax(sub_outputs, dim=1)
    _, sub_predicted = sub_predictions.topk(1, 1)

    t_predictions = torch.softmax(target_outputs, dim=1)
    _, t_predicted = t_predictions.topk(1, 1)

    # get balanced accuracy score
    unique, counts = np.unique(all_superclasses.cpu().numpy(), return_counts=True)
    test_samples_per_superclass_class[unique] += counts
    for t, p in zip(all_superclasses.cpu().numpy(), sup_predicted.cpu().numpy()):
        if t in p:
            test_correct_per_superclass_class[t] += 1

    unique, counts = np.unique(all_subclasses.cpu().numpy(), return_counts=True)
    test_samples_per_subclass_class[unique] += counts
    for t, p in zip(all_subclasses.cpu().numpy(), sub_predicted.cpu().numpy()):
        if t in p:
            test_correct_per_subclass_class[t] += 1

    unique, counts = np.unique(all_targets.cpu().numpy(), return_counts=True)
    test_samples_per_target_class[unique] += counts
    for t, p in zip(all_targets.cpu().numpy(), t_predicted.cpu().numpy()):
        if t in p:
            test_correct_per_target_class[t] += 1

    # calculate per-class accuracy
    per_class_acc = test_correct_per_target_class / test_samples_per_target_class
    # calculate balanced accuracy
    balanced_accuracy = 100. * np.mean(per_class_acc)
    # save the balanced accuracy for this epoch
    t_acc_balanced.append(balanced_accuracy)

    # same for subclass
    per_class_acc = test_correct_per_subclass_class / test_samples_per_subclass_class
    balanced_accuracy = 100. * np.mean(per_class_acc)
    sub_acc_balanced.append(balanced_accuracy)

    # and superclass
    per_class_acc = test_correct_per_superclass_class / test_samples_per_superclass_class
    # print("val per class acc: ", per_class_acc, self.trainset.supercluster_class_weights)
    balanced_accuracy = 100. * np.mean(per_class_acc)
    super_acc_balanced.append(balanced_accuracy)

    print(
        f'Top1 Super Accuracy: {super_acc_balanced[-1]:.4f} | Top-1 Sub Accuracy: {sub_acc_balanced[-1]:.4f} | Top-1 Target Accuracy:{t_acc_balanced[-1]:.4f}')

    # get the subclass histogram
    histogram_subclass = find_first_incorrect_classification(target_outputs, all_subclasses,
                                                             modelWrapper.subcluster_map, histogram_subclass,
                                                             modelWrapper.original_targets_map)

    # get the superclass histogram
    histogram_superclass = find_first_incorrect_classification(target_outputs, all_superclasses,
                                                               modelWrapper.supercluster_map, histogram_superclass,
                                                               modelWrapper.original_targets_map)
    subclass_to_text_for_plot = {}
    for number, text in modelWrapper.subcluster_to_text.items():
        short_text = text.split(",")[0]
        subclass_to_text_for_plot[number] = short_text

    # we create teh confusion matrices
    create_confusion_matrix(target_outputs, all_targets, sub_outputs, all_subclasses, super_outputs, all_superclasses,
                            modelWrapper.subcluster_map, subcluster_split, modelWrapper.trainset.original_targets_map,
                            subclass_to_text_for_plot, results_dir, modelWrapper.model_string)

    # saving the results
    results_dict = {
        "Name": modelWrapper.model_string,
        "Super Balanced Accuracy": super_acc_balanced[-1],
        "Sub Balanced Accuracy": sub_acc_balanced[-1],
        "Target Balanced Accuracy": t_acc_balanced[-1],
    }
    return histogram_superclass, histogram_subclass, results_dict



def get_state_dict(checkpoint_dir, device=torch.device('cpu')):
    checkpoint_files = os.listdir(checkpoint_dir)
    print("loading:", checkpoint_files[-1])
    if len(checkpoint_files) > 0:
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('-')[-1].split('.')[0][5:]))
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint_file, device)
        try:
            state_dict = checkpoint['model_state_dict']
        except KeyError:
            state_dict = checkpoint
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            new_key = "module." + key
            new_state_dict[new_key] = value
        return new_state_dict


def record_number_of_misclassifications(n1, n2, n3, n4, key):
    # this file is hardcoded and provides an overview of misclassification happening inside the same super or subclass
    # the first two columns record the total number of misclassification (without correct classification) and the
    # latter two rows including correct classifications
    file_name = "misclassifications by level.csv"
    columns = ["description", "superclass misclassifications", "subclass misclassification",
               "superclass classification", "subclass classification"]

    if os.path.isfile(file_name):
        # File exists, load it and append new data
        df = pd.read_csv(file_name)
        new_data = pd.DataFrame([[key, n1, n2, n3, n4]], columns=columns)
        df = df.append(new_data, ignore_index=True)
    else:
        # File doesn't exist, create it with new data
        data = {
            columns[0]: [key],
            columns[1]: [n1],
            columns[2]: [n2],
            columns[3]: [n3],
            columns[4]: [n4]
        }
        df = pd.DataFrame(data)

    df.to_csv(file_name, index=False)


def visualize_multiple_histograms(histograms, raw_text_map, class_map, path_to_save, color_map, names,
                                  class_type="Subclass"):
    """
    This function plots multiple histograms
    :param histograms: array of histograms
    :param raw_text_map: mapping to text
    :param class_map: mapping to classes (either super- or sub)
    :param path_to_save: path to save plot
    :param color_map: color map, should correspond to ordering of histograms
    :param names: names on plot
    :param class_type: either "Subclass" or "Superclass", depending on what level is looked at. Default is "Subclass"
    """
    n_subclasses = len(raw_text_map)

    if class_type == "Subclass":
        subplot_fontsize = 28
        fontsize = 32
        num_cols = int(np.sqrt(n_subclasses))
        num_rows = int(np.ceil(n_subclasses / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    else:
        subplot_fontsize = 20
        fontsize = 18
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    text_map = {}
    for number, text in raw_text_map.items():
        short_text = text.split(",")[0]
        text_map[number] = short_text

    # get count of subcluster
    cluster_counts = {}
    for value in class_map.values():
        count = list(class_map.values()).count(value)
        cluster_counts[value] = count

    # flatten the axes array for easy iteration
    axes = axes.flatten()

    break_points = {}

    # Iterate through all subclasses
    for idx in range(n_subclasses):
        ax = axes[idx]

        # get count of subcluster
        cluster_counts = {}
        for value in class_map.values():
            count = list(class_map.values()).count(value)
            cluster_counts[value] = count

        norm_factor = cluster_counts[idx]
        break_points[text_map[idx]] = {}

        # Iterate through all histograms and plot each on its own subplot
        for i, histogram in enumerate(histograms):
            sorted_samples = np.sort(histogram[idx])
            normalized_samples = sorted_samples / norm_factor

            cumulative_samples = np.arange(1, len(histogram[idx]) + 1) / len(histogram[idx])
            # flipping cumulative samples for natural interpretability
            cumulative_samples = 1 - cumulative_samples
            # now it is a CCDF
            # break points we want to save for table (normalized by number of classes in subclass
            percentages = [0.1, .25, .5, .75]

            points = [int(norm_factor * perc) for perc in percentages]
            for point_index, p in enumerate(points):
                cdf_value = np.interp(p, sorted_samples, cumulative_samples)
                if names[i] not in break_points[text_map[idx]].keys():
                    break_points[text_map[idx]][names[i]] = {}
                break_points[text_map[idx]][names[i]][percentages[point_index]] = cdf_value

            ax.step(normalized_samples, cumulative_samples,
                    label=names[i], color=color_map[i], where="post")

            if class_type != "Subclass":
                ax.legend(loc='upper right')
            ax.set_xlim(-0.05, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.tick_params(axis='both', labelsize=16, direction='in')  # Set the tick positions
            ax.set_title('{} {} - {}'.format(class_type, text_map[idx], cluster_counts[idx]), fontsize=subplot_fontsize)
        # ax.legend(loc='upper left')

    with open(path_to_save + ".json", "w") as file:
        json.dump(break_points, file)

    # Remove unused subplots if the total number of subplots is not equal to the number of subclasses
    for idx in range(n_subclasses, len(axes)):
        fig.delaxes(axes[idx])

    fig.supxlabel('Index of first Subclass Misclassification', fontsize=fontsize)
    fig.supylabel(f'Percentage of all {class_type} samples', fontsize=fontsize, x=0.01)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.tight_layout()

    if class_type == "Subclass":
        fig.subplots_adjust(left=0.05, top=0.95)
        leg = fig.legend(handles, labels, loc="upper center", fontsize='28', ncol=len(labels),
                         mode="expand", borderaxespad=0.05)
        # Increase the lineewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

    fig.savefig(path_to_save + ".pdf")
    plt.close()


def get_trainable_parameters(model_wrapper):
    """
    Get number of parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model_wrapper.model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def extract_incorrect_images(modelWrapper, dst_folder):
    """
    This function extracts images that were misclassified at the superclass level.
    :param modelWrapper: model to be analyzed
    :param dst_folder: where the images should be saved
    """
    print("Extracing images --", modelWrapper.model_string)
    batchsize = modelWrapper.batchsizes[-1]
    modelWrapper.model.load_state_dict(get_state_dict(modelWrapper.checkpoint_dir))
    modelWrapper.model.eval()
    incorrect_winds = []
    counter = 0
    original_mapping = modelWrapper.original_targets_map

    with open('../meta_data/imagenet_class_index.json') as json_file:
        im_class_index = json.load(json_file)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    with torch.no_grad():
        for i, (inputs, targets, subclasses, superclasses) in enumerate(modelWrapper.testloader):
            inputs, targets, subclasses, superclasses = inputs.to(
                'cuda:' + str(modelWrapper.device_ids[0])), targets.to(
                'cuda:' + str(modelWrapper.device_ids[0])), subclasses.to(
                'cuda:' + str(modelWrapper.device_ids[0])), superclasses.to(
                'cuda:' + str(modelWrapper.device_ids[0]))

            t_output, suboutput, superoutput = modelWrapper.model(inputs)

            # get predictions
            _, sup_predicted = superoutput.max(1)

            incorrects = sup_predicted != superclasses
            if incorrects.any():
                incorrect_images = inputs[incorrects]
                incorrect_superclasses = superclasses[incorrects]
                incorrect_super_preds = sup_predicted[incorrects]

                incorrect_indices = torch.nonzero(incorrects, as_tuple=False).squeeze(dim=1).cpu().numpy()

                dataset_indices = incorrect_indices + i * batchsize
                incorrect_filepaths = [modelWrapper.testset.samples[idx][0] for idx in
                                       dataset_indices]  # filepaths are stored in modelWrapper.testset.samples

                for filepath, true_label, pred_label, true_subs, pred_sub, true_target, pred_target in zip(
                        incorrect_filepaths, incorrect_superclasses, incorrect_super_preds):
                    short_file_path = filepath.split('/')[-2:]

                    true_superclass = modelWrapper.supercluster_to_text[true_label.cpu().item()].split(',')[0]
                    true_subclass = modelWrapper.subcluster_to_text[true_subs.cpu().item()].split(',')[0]
                    og_index = str(original_mapping[true_target.cpu().item()])
                    true_target = im_class_index[og_index][1]
                    predicted_superclass = modelWrapper.supercluster_to_text[pred_label.cpu().item()].split(',')[0]
                    predicted_subclass = modelWrapper.subcluster_to_text[pred_sub.cpu().item()].split(',')[0]
                    og_index = str(original_mapping[pred_target.cpu().item()])
                    predicted_target = im_class_index[og_index][1]

                    # define the destination folder and filename
                    filename = os.path.basename(filepath)
                    dst_filename = f"{true_superclass}_{true_subclass}_{true_target}_vs_{predicted_superclass}_{predicted_subclass}_{predicted_target}_{filename}"
                    dst_filepath = os.path.join(dst_folder, dst_filename)

                    # skip if we already have extracted that image
                    if os.path.exists(dst_filepath):
                        # print(f"File {dst_filepath} already exists. Skipping copy operation.")
                        counter += 1
                    else:
                        print(
                            f"True label {true_superclass}_{true_subclass}_{true_target} was incorrectly classified. Predicted label: {predicted_superclass}_{predicted_subclass}_{predicted_target}.")
                        incorrect_winds.append(short_file_path[0])
                        shutil.copy2(filepath, dst_filepath)
        df_words = pd.read_csv("../meta_data/words.txt", sep="\t", names=["code", "name"])
        words = []
        for wind in set(incorrect_winds):
            words.append(df_words.loc[df_words['code'] == wind].name.values[0])
        print(words)
        print("not printed", counter)
