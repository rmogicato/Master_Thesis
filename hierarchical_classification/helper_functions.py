import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn import utils

from create_class_map import get_imagenet_map, get_openset_map


class Imagenet_Hierarchy(torchvision.datasets.ImageNet):
    """
    This class creates a hierarchical data-set for image classification
    """

    def __init__(self, supercluster_map, subcluster_map, root, split, allowed_values, transform=None,
                 target_transform=None, training_master_map=None):
        """
        Note that the terms super- and subcluster are used interchangably for super- and subclusters
        :param supercluster_map: mapping of target classes to superclasses
        :param subcluster_map: mapping of target classes to subclasses
        :param root: directory to imagenet map
        :param split: Either "train" or "val"
        :param allowed_values: array of allowed target classes
        :param transform: transform function that should be applied to samples
        :param target_transform:
        :param training_master_map:
        """
        super().__init__(root, split=split, transform=transform, target_transform=target_transform)

        self.training_master_map = training_master_map
        # final targets
        self.supercluster_labels = self._get_superclass_labels(supercluster_map)
        self.original_targets_map = {}  # keeps track of new target mapping
        self.subcluster_labels = self._get_superclass_labels(subcluster_map)
        self.master_map = {}  # target to new target & superclasses

        self.samples, self.subcluster_labels, self.supercluster_labels = self._make_dataset(self.samples, self.targets,
                                                                                            self.subcluster_labels,
                                                                                            self.supercluster_labels,
                                                                                            allowed_values)
        assert len(self.samples) == len(self.subcluster_labels) == len(self.supercluster_labels)
        self.subcluster_class_weights = torch.tensor(
            utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.subcluster_labels),
                                                    y=np.array(self.subcluster_labels)), dtype=torch.float)
        self.supercluster_class_weights = torch.tensor(utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                               classes=np.unique(
                                                                                                   self.supercluster_labels),
                                                                                               y=np.array(
                                                                                                   self.supercluster_labels)),
                                                       dtype=torch.float)
        target_labels = [tup[1] for tup in self.samples]
        self.target_class_weights = torch.tensor(
            utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(target_labels),
                                                    y=np.array(target_labels)), dtype=torch.float)

    def _make_dataset(self, samples, targets, subcluster_labels, supercluster_labels, allowed_values):
        new_samples = []
        new_subcluster_labels = []
        new_supercluster_labels = []

        sorted_targets = np.unique(np.array(allowed_values))
        for idx, t in enumerate(targets):
            if t in allowed_values:
                new_target = int(np.where(sorted_targets == t)[0][0])
                if t not in self.original_targets_map.keys():
                    self.original_targets_map[new_target] = t
                new_samples.append((samples[idx][0], new_target))
                new_subcluster_labels.append(subcluster_labels[idx])
                new_supercluster_labels.append(supercluster_labels[idx])
                if t not in self.master_map.keys():
                    self.master_map[t] = {
                        "new_target": new_target,
                        "new_subcluster_label": subcluster_labels[idx],
                        "new_supercluster_label": supercluster_labels[idx]
                    }
                if self.training_master_map:
                    # asserting that mapping is correct
                    assert self.training_master_map[t]["new_target"] == new_target, print(new_target,
                                                                                          subcluster_labels[idx],
                                                                                          supercluster_labels[idx],
                                                                                          "\n",
                                                                                          self.training_master_map[t])
        return new_samples, new_subcluster_labels, new_supercluster_labels

    def _get_superclass_labels(self, class_to_superclass):
        """Returns a list of superclass labels for each image"""
        superclass_labels = []
        for target in self.targets:
            try:
                superclass_labels.append(class_to_superclass[target])
            except KeyError:
                # skip samples with missing superclass label
                superclass_labels.append(-1)
        return np.array(superclass_labels)

    def __getitem__(self, index):

        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        subcluster_label = self.subcluster_labels[index]
        superclass_label = self.supercluster_labels[index]
        return img, target, subcluster_label, superclass_label

    def __len__(self):
        return len(self.samples)


class Openset_Hierarchy(torchvision.datasets.ImageNet):
    """
    This dataset includes unknown classes -- it should not be used used as a training set
    """

    def __init__(self, supercluster_map, subcluster_map, root, split, allowed_values, unknown_classes, transform=None,
                 target_transform=None, training_master_map=None):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform)

        assert split == "val", "this open-set should only be used for evaluation"
        self.training_master_map = training_master_map
        # final targets
        self.supercluster_labels = self._get_superclass_labels(supercluster_map)
        self.original_targets_map = {}  # keeps track of new target mapping
        self.subcluster_labels = self._get_superclass_labels(subcluster_map)
        self.master_map = {}  # target to new target & superclasses
        self.unknown_classes = unknown_classes
        self.samples, self.subcluster_labels, self.supercluster_labels = self._make_dataset(self.samples, self.targets,
                                                                                            self.subcluster_labels,
                                                                                            self.supercluster_labels,
                                                                                            allowed_values,
                                                                                            self.unknown_classes)
        assert len(self.samples) == len(self.subcluster_labels) == len(self.supercluster_labels)
        self.subcluster_class_weights = torch.tensor(
            utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.subcluster_labels),
                                                    y=np.array(self.subcluster_labels)), dtype=torch.float)
        self.supercluster_class_weights = torch.tensor(utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                               classes=np.unique(
                                                                                                   self.supercluster_labels),
                                                                                               y=np.array(
                                                                                                   self.supercluster_labels)),
                                                       dtype=torch.float)
        target_labels = [tup[1] for tup in self.samples]
        self.target_class_weights = torch.tensor(
            utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(target_labels),
                                                    y=np.array(target_labels)), dtype=torch.float)

    def _make_dataset(self, samples, targets, subcluster_labels, supercluster_labels, allowed_values, unknown_classes):
        """
        This function creates a data set
        """
        new_samples = []
        new_subcluster_labels = []
        new_supercluster_labels = []

        sorted_targets = np.unique(np.array(allowed_values))
        for idx, t in enumerate(targets):
            if t in allowed_values:
                new_target = int(np.where(sorted_targets == t)[0][0])
            elif t in unknown_classes:
                new_target = -1
            else:
                continue
            if t not in self.original_targets_map.keys():
                self.original_targets_map[new_target] = t
                new_samples.append((samples[idx][0], new_target))
                new_subcluster_labels.append(subcluster_labels[idx])
                new_supercluster_labels.append(supercluster_labels[idx])
                if t not in self.master_map.keys():
                    self.master_map[t] = {
                        "new_target": new_target,
                        "new_subcluster_label": subcluster_labels[idx],
                        "new_supercluster_label": supercluster_labels[idx]
                    }
                if self.training_master_map:
                    # asserting that mapping is correct
                    assert self.training_master_map[t]["new_target"] == new_target, print(new_target,
                                                                                          subcluster_labels[idx],
                                                                                          supercluster_labels[idx],
                                                                                          "\n",
                                                                                          self.training_master_map[t])
        return new_samples, new_subcluster_labels, new_supercluster_labels

    def check_unknown_classes(self):
        unknown_class_samples = [sample for sample in self.samples if sample[1] == -1]
        print("Number of unknown class samples: ", len(unknown_class_samples))
        return unknown_class_samples

    def _get_superclass_labels(self, class_to_superclass):
        """Returns a list of superclass labels for each image"""
        superclass_labels = []
        for target in self.targets:
            try:
                superclass_labels.append(class_to_superclass[target])
            except KeyError:
                # skip samples with missing superclass label
                superclass_labels.append(-1)
        return np.array(superclass_labels)

    def __getitem__(self, index):

        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        subcluster_label = self.subcluster_labels[index]
        superclass_label = self.supercluster_labels[index]
        return img, target, subcluster_label, superclass_label

    def __len__(self):
        return len(self.samples)


class Imagenet_Backbone():
    """
    Apologize for the ambiguous class name:
    This is in fact a wrapper class for both datasets (val and train) and not a neural network backbone.
    """

    def __init__(self, num_epoch, n_superclusters, subclusters_split, batchsizes):
        assert len(subclusters_split) == n_superclusters
        self.num_epochs = num_epoch

        # Define the transforms to apply to the data
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.supercluster_map, self.subcluster_map, self.supercluster_to_text, self.subcluster_to_text, _ = get_imagenet_map(
            n_superclusters=n_superclusters, n_subclusters=subclusters_split)

        self.batchsizes = batchsizes
        # only keep values in the hierarchy
        allowed_values = np.unique(np.array(list(self.subcluster_map.keys()))).tolist()
        self.num_targets = len(allowed_values)
        print("Training with", self.num_targets, "target classes")

        print_clusters(self.supercluster_map, self.supercluster_to_text, name="Supercluster")
        print_clusters(self.subcluster_map, self.subcluster_to_text, name="Subcluster")

        self.trainset = Imagenet_Hierarchy(root='/local/scratch/datasets/ImageNet/ILSVRC2012', split="train",
                                           supercluster_map=self.supercluster_map, subcluster_map=self.subcluster_map,
                                           transform=self.transform_train, allowed_values=allowed_values)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsizes[0],
                                                       shuffle=True, num_workers=10)

        self.testset = Imagenet_Hierarchy(root='/local/scratch/datasets/ImageNet/ILSVRC2012', split="val",
                                          supercluster_map=self.supercluster_map, subcluster_map=self.subcluster_map,
                                          transform=self.transform_test, allowed_values=allowed_values,
                                          training_master_map=self.trainset.master_map)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batchsizes[1],
                                                      shuffle=False, num_workers=8)
        self.original_targets_map = self.testset.original_targets_map


class OpenSet_Backbone():
    """
    Wrapper for open-set data sets.
    """

    def __init__(self, num_epoch, n_superclusters, subclusters_split, batchsizes, unknown):
        assert len(subclusters_split) == n_superclusters
        self.num_epochs = num_epoch
        self.batchsizes = batchsizes
        # Define the transforms to apply to the data
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.supercluster_map, self.subcluster_map, self.supercluster_to_text, self.subcluster_to_text, _, unknown_subclasses, unknown_superclasses = get_openset_map(
            n_superclusters=n_superclusters, n_subclusters=subclusters_split)

        if unknown == "Superclasses":
            unknown_classes = unknown_superclasses
        elif unknown == "Subclasses":
            unknown_classes = unknown_subclasses
        else:
            raise ValueError("Unknown level provided, please select between 'Superclasses' and 'Subclasses'")
        allowed_values = np.unique(np.array(list(self.subcluster_map.keys()))).tolist()
        self.num_targets = len(allowed_values)
        print("Training with", self.num_targets, "target classes")

        print_clusters(self.supercluster_map, self.supercluster_to_text, name="Supercluster")
        print_clusters(self.subcluster_map, self.subcluster_to_text, name="Subcluster")

        self.testset = Openset_Hierarchy(root='/local/scratch/datasets/ImageNet/ILSVRC2012', split="val",
                                         supercluster_map=self.supercluster_map, subcluster_map=self.subcluster_map,
                                         transform=self.transform_test, allowed_values=allowed_values,
                                         training_master_map=None, unknown_classes=unknown_classes)

        self.testset.check_unknown_classes()  # sanity check
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batchsizes[1],
                                                      shuffle=False, num_workers=8)

        self.trainset = None

        self.trainloader = None

        self.original_targets_map = self.testset.original_targets_map


def print_clusters(map, text_map, name="Cluster"):
    supercluster_counts = {}
    for value in map.values():
        count = list(map.values()).count(value)
        supercluster_counts[value] = count

    for key, value in supercluster_counts.items():
        print(f"{name} {text_map[key]}: {value} values")
