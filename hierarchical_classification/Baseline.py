import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from helper_functions import Imagenet_Backbone


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, checkpoint_path)




class BaseNet():
    """
    Baseline model that can be used for each hierarchical level.
    This model has no knowledge of the hierarchy.
    The parameters are as follows:
    - num_epochs: integer with number of epochs
    - model_string: either "Superclass" or "Subclass", otherwise the target classes are used from the constructed hierarchy
    - n_superclusters: integer of the number of superclasses
    - subcluster_split: tuple with the size of each subcluster, length of tuple must match n_superclusters.
    - device_ids: tuple with device ids, if empty, only the cpu is used
    - checkpoint_dir: string to directory where checkpoints are saved
    - batchsizes: optional if no im_backbone is passed
    - im_backbone: instance of Imagenet_Backbone, avoids repetition of creating data-set if multiple models should be
    trained evaluated
    """

    def __init__(self, num_epochs, model_string, n_superclusters, subcluster_split, device_ids, checkpoint_dir,
                 batchsizes=(80, 50), im_backbone=None):

        if not im_backbone:
            im_backbone = Imagenet_Backbone(num_epochs, n_superclusters, subcluster_split, batchsizes)

        self.num_epochs = num_epochs
        self.num_targets = im_backbone.num_targets
        self.trainset = im_backbone.trainset
        self.supercluster_map, self.subcluster_map, self.supercluster_to_text, self.subcluster_to_text = im_backbone.supercluster_map, im_backbone.subcluster_map, im_backbone.supercluster_to_text, im_backbone.subcluster_to_text
        self.trainset = im_backbone.trainset
        self.trainloader = im_backbone.trainloader
        self.testset = im_backbone.testset
        self.testloader = im_backbone.testloader
        self.device_ids = device_ids
        self.num_superclasses = n_superclusters
        self.num_subclasses = sum(subcluster_split)
        self.original_targets_map = self.testset.original_targets_map

        self.model_string = model_string
        if model_string == "Superclass":
            self.num_classes = self.num_superclasses
        elif model_string == "Subclass":
            self.num_classes = self.num_subclasses
        else:
            self.model_string = "Target"
            self.num_classes = self.num_targets
        model = models.resnet50(pretrained=False, num_classes=self.num_classes)
        if type(device_ids) is int:
            self.device_ids = (device_ids,)
        if torch.cuda.device_count() > 1 and len(self.device_ids) > 0:
            print("Using", len(self.device_ids), "GPUs!")
            self.model = nn.DataParallel(
                model.to('cuda:' + str(self.device_ids[0])),
                device_ids=self.device_ids)
        else:
            self.model = nn.DataParallel(model)
            print("CPU evaluation mode - nothing is loaded to the GPU")
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_dir = checkpoint_dir

    def train(self):
        if self.model_string == "Superclass":
            self.weights = self.trainset.supercluster_class_weights
        elif self.model_string == "Subclass":
            self.weights = self.trainset.subcluster_class_weights
        else:
            self.weights = self.trainset.target_class_weights
        for name, param in self.model.named_parameters():
            if param.device.type == 'cuda:' + str(self.device_ids[0]) and param.device.index == 0:
                print(f'{name}: {param.device}')

        acc = []
        train_acc = []
        train_acc_balanced = []
        acc_balanced = []
        train_loss = []

        # cross entropy loss is adapted to class weights
        criterion = nn.CrossEntropyLoss(weight=self.weights).to(
            'cuda:' + str(self.device_ids[0]))

        opt = optim.SGD(self.model.module.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

        start_epoch = 0

        checkpoint_files = os.listdir(self.checkpoint_dir)
        if len(checkpoint_files) > 0:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('-')[-1].split('.')[0][5:]))
            latest_checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_files[-1])

            checkpoint = torch.load(latest_checkpoint_file)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = collections.OrderedDict()
            for key, value in state_dict.items():
                new_key = "module." + key
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']

        for epoch in range(start_epoch, self.num_epochs):
            train_total = 0
            train_correct_total = 0
            train_correct_per_class = np.zeros(self.num_classes, dtype=int)
            train_samples_per_class = np.zeros(self.num_classes, dtype=int)
            train_recorded_loss = 0

            self.model.train()

            for batch_idx, (inputs, targets, subclasses, superclasses) in enumerate(self.trainloader):
                inputs, targets, subclasses, superclasses = inputs.to('cuda:' + str(self.device_ids[0])), targets.to(
                    'cuda:' + str(self.device_ids[0])), subclasses.to(
                    'cuda:' + str(self.device_ids[0])), superclasses.to('cuda:' + str(self.device_ids[0]))

                if self.model_string == "Superclass":
                    targets = superclasses
                elif self.model_string == "Subclass":
                    targets = subclasses

                opt.zero_grad()

                output = self.model(inputs)
                # calculate loss

                loss = criterion(output, targets)

                loss.backward()

                opt.step()

                # Record training performance
                _, predicted = output.max(1)
                train_total += targets.size(0)
                train_correct_total += predicted.eq(targets).sum().item()

                # Calculate training loss
                train_recorded_loss += loss.item() * inputs.size(0)

                # count how many samples per class exist
                unique, counts = np.unique(targets.cpu().numpy(), return_counts=True)
                train_samples_per_class[unique] += counts
                for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                    if t == p:
                        train_correct_per_class[t] += 1

            scheduler.step()
            if epoch % 5 == 0:  # saving every 5th epoch
                save_checkpoint(self.model, opt, scheduler, epoch, train_loss, self.checkpoint_dir)

            # Calculate training accuracies
            train_acc.append(100. * train_correct_total / train_total)

            train_loss.append(100. * train_recorded_loss / train_total)

            # calculate per-class accuracy
            per_class_acc = train_correct_per_class / (train_samples_per_class + 1e-10)
            # Calculate balanced accuracy
            balanced_accuracy = np.mean(per_class_acc)
            # Save the balanced accuracy for this epoch
            train_acc_balanced.append(100. * balanced_accuracy)

            self.model.eval()

            test_total = 0
            test_correct_total = 0
            test_correct_per_class = np.zeros(self.num_classes, dtype=int)
            test_samples_per_class = np.zeros(self.num_classes, dtype=int)
            test_recorded_loss = 0

            with torch.no_grad():
                for inputs, targets, subclasses, superclasses in self.testloader:
                    inputs, targets, subclasses, superclasses = inputs.to(
                        'cuda:' + str(self.device_ids[0])), targets.to(
                        'cuda:' + str(self.device_ids[0])), subclasses.to(
                        'cuda:' + str(self.device_ids[0])), superclasses.to('cuda:' + str(self.device_ids[0]))

                    if self.model_string == "Superclass":
                        targets = superclasses
                    elif self.model_string == "Subclass":
                        targets = subclasses

                    output = self.model(inputs)

                    _, predicted = output.max(1)
                    test_total += targets.size(0)
                    test_correct_total += predicted.eq(targets).sum().item()

                    # count how many samples per class exist and how many were correctly predicted for each class
                    unique, counts = np.unique(targets.cpu().numpy(), return_counts=True)
                    test_samples_per_class[unique] += counts
                    for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                        if t == p:
                            test_correct_per_class[t] += 1

                acc.append(100. * test_correct_total / test_total)

                # calculate per-class accuracy
                per_class_acc = test_correct_per_class / (test_samples_per_class + 1e-10)
                # calculate balanced accuracy
                balanced_accuracy = 100. * np.mean(per_class_acc)
                # save the balanced accuracy for this epoch
                acc_balanced.append(balanced_accuracy)

            print(
                f'Epoch {epoch + 1}/{self.num_epochs}: {self.model_string} Accuracy - Val: {acc_balanced[-1]:.4f}({acc[-1]:.4f}) Train:{train_acc_balanced[-1]:.4f}({train_acc[-1]:.4f})')

        accuracies = {
            "validation " + self.model_string + " accuracy": acc,
            "validation " + self.model_string + " balanced accuracy": acc_balanced,
            "train " + self.model_string + " balanced accuracy": train_acc,
            "train " + self.model_string + " balanced accuracy": train_acc_balanced,
            "loss " + self.model_string + " training loss": train_loss,
        }
        try:
            save_checkpoint(self.model, opt, scheduler, self.num_epochs, [], self.checkpoint_dir)
        except Exception:
            print("Error while saving")

        print("Training finished!")

        return accuracies, self.model.module
