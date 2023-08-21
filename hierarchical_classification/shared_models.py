import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from helper_functions import Imagenet_Backbone


class BSN(nn.Module):
    """
    This is the BSN network.
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(BSN, self).__init__()
        self.backbone = models.resnet50(pretrained=False)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final FC layer

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features, num_subclasses)

        # Target FC layer
        self.fc_target = nn.Linear(num_features, num_targetclasses)

    def forward(self, x):
        # Compute features
        features = self.backbone(x)
        # Predict superclass
        super_logits = self.fc_super(features)

        subclass_logits = self.fc_sub(features)

        target_logits = self.fc_target(features)

        return target_logits, subclass_logits, super_logits


class ASN(nn.Module):
    """
    This is the ASN. The bug is marked with a comment.
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(ASN, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # removing last fc layer from resnet

        num_features = resnet.fc.in_features # set number of features to that of the deep features

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features, num_subclasses)

        # Target FC layer
        self.fc_t = nn.Linear(num_features, num_targetclasses)

        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute features
        features = self.backbone(x)
        features = features.view(features.size(0), -1) # keep batches flatten resulting map
        # Predict superclass
        super_logits = self.fc_super(features)
        # get superclass probabilities
        super_probs = self.softmax(super_logits)  # hardmax, ground truth?
        # multiply with weights of the superclass - this way the weight with the highest probability is the largest
        super_probability_weights = torch.matmul(super_probs, self.fc_super.weight)
        # THIS IS THE BUG -- here we sum up across a batch, which is not as intended
        W_super = torch.sum(super_probability_weights, dim=0)
        # and use it to weight the features
        fused_super_features = torch.mul(features, W_super)

        sub_logits = self.fc_sub(fused_super_features)
        sub_probs = self.softmax(sub_logits)
        sub_probability_weights = torch.matmul(sub_probs, self.fc_sub.weight)

        # THIS IS THE BUG -- here we sum up across a batch, which is not as intended
        W_sub = torch.sum(sub_probability_weights, dim=0)
        sub_fused_features = torch.mul(features, W_sub)

        target_logits = self.fc_t(sub_fused_features)

        return target_logits, sub_logits, super_logits


class ASN_upside_down(nn.Module):
    """
    This is the ASN-U. The bug is marked with a comment.
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses, ):
        super(ASN_upside_down, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # removing last fc layer from resnet

        num_features = resnet.fc.in_features

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features, num_subclasses)

        # Target FC layer
        self.fc_t = nn.Linear(num_features, num_targetclasses)

        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # predict target classes first
        target_logits = self.fc_t(features)
        # get probability distribution of target classes
        target_probs = self.softmax(target_logits)
        target_probability_weights = torch.matmul(target_probs, self.fc_t.weight)
        # BUG: this sum is incorrect
        W_t = torch.sum(target_probability_weights, dim=0)
        target_fused_features = torch.mul(features, W_t)

        # the fused features are now used for subclass prediction
        sub_logits = self.fc_sub(target_fused_features)
        sub_probs = self.softmax(sub_logits)
        sub_probability_weights = torch.matmul(sub_probs, self.fc_sub.weight)
        # BUG: again this sum is incorrect
        W_sub = torch.sum(sub_probability_weights, dim=0)
        sub_fused_features = torch.mul(features, W_sub)

        # Predict superclass
        super_logits = self.fc_super(sub_fused_features)

        return target_logits, sub_logits, super_logits


class ASN_A(nn.Module):
    """
    This is the ASN-A
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(ASN_A, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # removing last fc layer from resnet

        num_features = resnet.fc.in_features

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features, num_subclasses)

        # Target FC layer
        self.fc_t = nn.Linear(num_features, num_targetclasses)

    def forward(self, x):
        # Compute features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # Predict superclass
        super_logits = self.fc_super(features)
        # get superclass probabilities
        _, super_preds = torch.max(super_logits, dim=1)
        W_super = self.fc_super.weight[super_preds]
        # and use it to weight the features
        fused_super_features = torch.mul(features, W_super)

        sub_logits = self.fc_sub(fused_super_features)
        _, sub_preds = torch.max(sub_logits, dim=1)
        W_sub = self.fc_sub.weight[sub_preds]
        sub_fused_features = torch.mul(features, W_sub)

        target_logits = self.fc_t(sub_fused_features)

        return target_logits, sub_logits, super_logits


class ASN_T(nn.Module):
    """
    This is the ASN-T
    """

    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(ASN_T, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # removing last fc layer from resnet

        num_features = resnet.fc.in_features

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features, num_subclasses)

        # Target FC layer
        self.fc_t = nn.Linear(num_features, num_targetclasses)

    # Make sure that in validation no superclass and subclass labels are passed
    def forward(self, x, superclass=None, subclass=None):
        # Compute features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # Predict superclass
        super_logits = self.fc_super(features)

        # We we do not have labels, we use the predicted maximum
        if superclass is None:
            _, superclass = torch.max(super_logits, dim=1)  # getting predictions

        # get superclass probabilities
        W_super = self.fc_super.weight[superclass]
        # and use it to weight the features
        fused_super_features = torch.mul(features, W_super)

        sub_logits = self.fc_sub(fused_super_features)

        if subclass is None:
            _, subclass = torch.max(sub_logits, dim=1)  # getting predictions
        W_sub = self.fc_sub.weight[subclass]
        sub_fused_features = torch.mul(features, W_sub)

        target_logits = self.fc_t(sub_fused_features)

        return target_logits, sub_logits, super_logits


class ASN_TS(nn.Module):
    """
    This is the ASN-TS. The bug is marked with a comment
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(ASN_TS, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # removing last fc layer from resnet

        num_features = resnet.fc.in_features

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features, num_subclasses)

        # Target FC layer
        self.fc_t = nn.Linear(num_features, num_targetclasses)

        # softmax
        self.softmax = nn.Softmax(dim=1)

    # Make sure that in validation no superclass and subclass labels are passed
    def forward(self, x, superclass=None, subclass=None):
        # Compute features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # Predict superclass
        super_logits = self.fc_super(features)

        if superclass is None:
            super_probs = self.softmax(super_logits)  # getting predictions
            super_probability_weights = torch.matmul(super_probs, self.fc_super.weight)
            # BUG: we sum up across batches, it only affects predictions for the validation set
            W_super = torch.sum(super_probability_weights, dim=0)
        # get superclass probabilities
        else:
            W_super = self.fc_super.weight[superclass]
        # and use it to weight the features
        fused_super_features = torch.mul(features, W_super)

        sub_logits = self.fc_sub(fused_super_features)

        if subclass is None:
            sub_probs = self.softmax(sub_logits)
            sub_probability_weights = torch.matmul(sub_probs, self.fc_sub.weight)
            # BUG: we sum up across batches, it only affects predictions for the validation set
            W_sub = torch.sum(sub_probability_weights, dim=0)
        else:
            W_sub = self.fc_sub.weight[subclass]

        sub_fused_features = torch.mul(features, W_sub)
        target_logits = self.fc_t(sub_fused_features)

        return target_logits, sub_logits, super_logits


class LogitSharing(nn.Module):
    """
    This is the LSN
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(LogitSharing, self).__init__()
        self.backbone = models.resnet50(pretrained=False)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final FC layer

        # Superclass FC layer
        self.fc_super = nn.Linear(num_features, num_superclasses)

        # Subclass FC layer
        self.fc_sub = nn.Linear(num_features + num_superclasses, num_subclasses)

        # Target FC layer
        self.fc_target = nn.Linear(num_features + num_subclasses, num_targetclasses)

    def forward(self, x):
        # Compute features
        features = self.backbone(x)
        # Predict superclass
        # getting logits
        super_logits = self.fc_super(features)
        # concatinating probabilities
        sub_features = torch.cat([features, super_logits], dim=1)
        subclass_logits = self.fc_sub(sub_features)

        target_features_features = torch.cat([features, subclass_logits], dim=1)
        target_logits = self.fc_target(target_features_features)

        return target_logits, subclass_logits, super_logits


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, checkpoint_path)


class SharedNet:
    # setting a seed
    torch.manual_seed(42)

    def __init__(self, num_epochs, model_string, n_superclusters, subcluster_split, device_ids, checkpoint_dir,
                 batchsizes=(64, 50), im_backbone=None):
        """
        :param num_epochs: number of epochs for training
        :param model_string: a string of these possible answers:  LSN, BSN, ASN, ASN-A, ASN-T, ASN-TS, ASN-U
        :param n_superclusters: number of superclasses
        :param subcluster_split: split of subclasses
        :param device_ids: devices for training
        :param checkpoint_dir: directory of checkpoints
        :param batchsizes: desired batchsize, only required if no ImageNet_Backbone is passed
        :param im_backbone: wrapper of datasets, should be either a class ImageNet_Backbone or OpenSet_Backbone
        """
        # original architecture did this vida inheritance, but that means each model needs to create a new dataset.
        if not im_backbone:
            # if no imageset is provided, a manual one is constructed
            print("No dataset provided, creating new one")
            im_backbone = Imagenet_Backbone(num_epochs, n_superclusters, subcluster_split, batchsizes)

        self.num_targets = im_backbone.num_targets
        self.trainset = im_backbone.trainset
        self.supercluster_map, self.subcluster_map, self.supercluster_to_text, self.subcluster_to_text = im_backbone.supercluster_map, im_backbone.subcluster_map, im_backbone.supercluster_to_text, im_backbone.subcluster_to_text
        self.trainset = im_backbone.trainset
        self.trainloader = im_backbone.trainloader
        self.testset = im_backbone.testset
        self.testloader = im_backbone.testloader

        allowed_values = np.unique(np.array(list(self.subcluster_map.keys()))).tolist()
        self.num_targets = len(allowed_values)
        self.num_epochs = num_epochs
        print("Training with", self.num_targets, "target classes")

        # we keep track of the original classes
        self.original_targets_map = self.testset.original_targets_map

        self.device_ids = device_ids
        self.num_superclasses = n_superclusters
        self.num_subclasses = sum(subcluster_split)
        self.baseline = False
        self.model_string = model_string
        self.batchsizes = im_backbone.batchsizes

        # selecting correct architecture
        if model_string == "LSN":
            model = LogitSharing(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                                                num_targetclasses=self.num_targets)
        elif model_string == "BSN":
            model = BSN(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                        num_targetclasses=self.num_targets)
        elif model_string == "ASN":
            model = ASN(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                        num_targetclasses=self.num_targets)
        elif model_string == "ASN-A":
            model = ASN_A(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                          num_targetclasses=self.num_targets)
        elif model_string == "ASN-T":
            model = ASN_T(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                          num_targetclasses=self.num_targets)
        elif model_string == "ASN-TS":
            model = ASN_TS(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                           num_targetclasses=self.num_targets)
        elif model_string == "ASN-U":
            model = ASN_upside_down(num_superclasses=n_superclusters, num_subclasses=self.num_subclasses,
                                    num_targetclasses=self.num_targets)
        else:
            raise Exception(
                "Incorrect model string, please select from these models: LSN\nBSN\nASN\nASN-A\nASN-T\nASN-TS\nASN-U\n")

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
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        for name, param in self.model.named_parameters():
            if param.device.type == 'cuda:' + str(self.device_ids[0]) and param.device.index == 0:
                print(f'{name}: {param.device}')

        super_acc = []
        sub_acc = []
        t_acc = []
        train_super_acc = []
        train_sub_acc = []
        train_t_acc = []

        train_super_acc_balanced = []
        train_sub_acc_balanced = []
        train_t_acc_balanced = []

        super_acc_balanced = []
        sub_acc_balanced = []
        t_acc_balanced = []

        train_t_loss = []
        train_sub_loss = []
        train_super_loss = []

        # cross entropy loss is adapted to class weights
        super_criterion = nn.CrossEntropyLoss(weight=self.trainset.supercluster_class_weights).to(
            'cuda:' + str(self.device_ids[0]))
        sub_criterion = nn.CrossEntropyLoss(weight=self.trainset.subcluster_class_weights).to(
            'cuda:' + str(self.device_ids[0]))
        target_criterion = nn.CrossEntropyLoss(weight=self.trainset.target_class_weights).to(
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
            train_correct_total = np.zeros(3, dtype=int)
            train_correct_target_per_class = np.zeros(self.num_targets, dtype=int)
            train_correct_sub_per_class = np.zeros(self.num_subclasses, dtype=int)
            train_correct_superclasses_per_class = np.zeros(self.num_superclasses, dtype=int)

            train_samples_per_target_class = np.zeros(self.num_targets, dtype=int)
            train_samples_per_subclass_class = np.zeros(self.num_subclasses, dtype=int)
            train_samples_per_superclass_class = np.zeros(self.num_superclasses, dtype=int)
            train_losses = np.zeros(3, dtype=float)

            self.model.train()

            for batch_idx, (inputs, targets, subclasses, superclasses) in enumerate(self.trainloader):
                inputs, targets, subclasses, superclasses = inputs.to('cuda:' + str(self.device_ids[0])), \
                                                            targets.to('cuda:' + str(self.device_ids[0])), \
                                                            subclasses.to('cuda:' + str(self.device_ids[0])), \
                                                            superclasses.to('cuda:' + str(self.device_ids[0]))
                opt.zero_grad()

                if self.model_string == "ASN-T" or self.model_string == "ASN-TS":
                    t_output, suboutput, superoutput = self.model(inputs, superclasses, subclasses)
                else:
                    t_output, suboutput, superoutput = self.model(inputs)
                # calculate loss
                superloss = super_criterion(superoutput, superclasses)
                subloss = sub_criterion(suboutput, subclasses)
                t_loss = target_criterion(t_output, targets)

                # if it is not a baseline, i.e., three models, we propagate the loss jointly

                weighted_loss = (superloss * 0.25) + (subloss * 0.25) + (t_loss * 0.5)
                weighted_loss.backward()

                opt.step()

                # Record training performance
                _, sup_predicted = superoutput.max(1)
                train_total += superclasses.size(0)
                train_correct_total[2] += sup_predicted.eq(superclasses).sum().item()

                _, sub_predicted = suboutput.max(1)
                train_correct_total[1] += sub_predicted.eq(subclasses).sum().item()

                _, t_predicted = t_output.max(1)
                train_correct_total[0] += t_predicted.eq(targets).sum().item()

                # Calculate training loss
                train_losses[0] += t_loss.item() * 0.5 * inputs.size(0)
                train_losses[1] += subloss.item() * 0.25 * inputs.size(0)
                train_losses[2] += superloss.item() * 0.25 * inputs.size(0)

                # count how many samples per class exist
                unique, counts = np.unique(superclasses.cpu().numpy(), return_counts=True)
                train_samples_per_superclass_class[unique] += counts
                for t, p in zip(superclasses.cpu().numpy(), sup_predicted.cpu().numpy()):
                    if t == p:
                        train_correct_superclasses_per_class[t] += 1

                unique, counts = np.unique(subclasses.cpu().numpy(), return_counts=True)
                train_samples_per_subclass_class[unique] += counts
                for t, p in zip(subclasses.cpu().numpy(), sub_predicted.cpu().numpy()):
                    if t == p:
                        train_correct_sub_per_class[t] += 1

                unique, counts = np.unique(targets.cpu().numpy(), return_counts=True)
                train_samples_per_target_class[unique] += counts
                for t, p in zip(targets.cpu().numpy(), t_predicted.cpu().numpy()):
                    if t == p:
                        train_correct_target_per_class[t] += 1

            scheduler.step()
            if epoch % 5 == 0:  # saving every 5th epoch
                save_checkpoint(self.model, opt, scheduler, epoch, train_losses, self.checkpoint_dir)

            # Calculate training accuracies
            train_super_acc.append(100. * train_correct_total[2] / train_total)
            train_sub_acc.append(100. * train_correct_total[1] / train_total)
            train_t_acc.append(100. * train_correct_total[0] / train_total)

            train_t_loss.append(100. * train_losses[0] / train_total)
            train_sub_loss.append(100. * train_losses[1] / train_total)
            train_super_loss.append(100. * train_losses[2] / train_total)

            # calculate per-class accuracy
            per_class_acc = train_correct_target_per_class / train_samples_per_target_class
            # Calculate balanced accuracy
            balanced_accuracy = np.mean(per_class_acc)
            # Save the balanced accuracy for this epoch
            train_t_acc_balanced.append(100. * balanced_accuracy)

            per_class_acc = train_correct_sub_per_class / train_samples_per_subclass_class
            balanced_accuracy = np.mean(per_class_acc)
            train_sub_acc_balanced.append(100. * balanced_accuracy)

            per_class_acc = train_correct_superclasses_per_class / train_samples_per_superclass_class
            balanced_accuracy = np.mean(per_class_acc)
            train_super_acc_balanced.append(100. * balanced_accuracy)

            self.model.eval()

            test_total = 0
            test_correct = [0, 0, 0]
            test_samples_per_target_class = np.zeros(self.num_targets, dtype=int)
            test_samples_per_subclass_class = np.zeros(self.num_subclasses, dtype=int)
            test_samples_per_superclass_class = np.zeros(self.num_superclasses, dtype=int)

            test_correct_per_target_class = np.zeros(self.num_targets, dtype=int)
            test_correct_per_subclass_class = np.zeros(self.num_subclasses, dtype=int)
            test_correct_per_superclass_class = np.zeros(self.num_superclasses, dtype=int)

            with torch.no_grad():
                for inputs, targets, subclasses, superclasses in self.testloader:
                    inputs, targets, subclasses, superclasses = inputs.to(
                        'cuda:' + str(self.device_ids[0])), targets.to(
                        'cuda:' + str(self.device_ids[0])), subclasses.to(
                        'cuda:' + str(self.device_ids[0])), superclasses.to('cuda:' + str(self.device_ids[0]))

                    t_output, suboutput, superoutput = self.model(inputs)

                    _, sup_predicted = superoutput.max(1)
                    test_total += superclasses.size(0)
                    test_correct[2] += sup_predicted.eq(superclasses).sum().item()

                    _, sub_predicted = suboutput.max(1)
                    test_correct[1] += sub_predicted.eq(subclasses).sum().item()

                    _, t_predicted = t_output.max(1)
                    test_correct[0] += t_predicted.eq(targets).sum().item()

                    # count how many samples per class exist and how many were correctly predicted for each class
                    unique, counts = np.unique(superclasses.cpu().numpy(), return_counts=True)
                    test_samples_per_superclass_class[unique] += counts
                    for t, p in zip(superclasses.cpu().numpy(), sup_predicted.cpu().numpy()):
                        if t == p:
                            test_correct_per_superclass_class[t] += 1

                    unique, counts = np.unique(subclasses.cpu().numpy(), return_counts=True)
                    test_samples_per_subclass_class[unique] += counts
                    for t, p in zip(subclasses.cpu().numpy(), sub_predicted.cpu().numpy()):
                        if t == p:
                            test_correct_per_subclass_class[t] += 1

                    unique, counts = np.unique(targets.cpu().numpy(), return_counts=True)
                    test_samples_per_target_class[unique] += counts
                    for t, p in zip(targets.cpu().numpy(), t_predicted.cpu().numpy()):
                        if t == p:
                            test_correct_per_target_class[t] += 1

                super_acc.append(100. * test_correct[2] / test_total)
                sub_acc.append(100. * test_correct[1] / test_total)
                t_acc.append(100. * test_correct[0] / test_total)

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
                f'Epoch {epoch + 1}/{self.num_epochs}: Super Accuracy - Val: {super_acc_balanced[-1]:.4f}({super_acc[-1]:.4f}) Train:{train_super_acc_balanced[-1]:.4f}({train_super_acc[-1]:.4f}) | Sub Accuracy: - Val {sub_acc_balanced[-1]:.4f}({sub_acc[-1]:.4f})  Train: {train_sub_acc_balanced[-1]:.4f}({train_sub_acc[-1]:.4f}) | Target Accuracy Val:{t_acc_balanced[-1]:.4f}({t_acc[-1]:.4f}) Train: {train_t_acc_balanced[-1]:.4f}({train_t_acc[-1]:.4f})')

        val_accuracies = {
            "validation target accuracy": t_acc,
            "validation subclass accuracy": sub_acc,
            "validation superclass accuracy": super_acc,
        }

        val_balanced_accuracies = {
            "validation target balanced accuracy": t_acc_balanced,
            "validation subclass balanced accuracy": sub_acc_balanced,
            "validation superclass balanced accuracy": super_acc_balanced,
        }

        train_balanced_accuracies = {
            "training target balanced accuracy": train_t_acc_balanced,
            "training subclass balanced accuracy": train_sub_acc_balanced,
            "training superclass balanced accuracy": train_super_acc_balanced,
        }

        train_accuracies = {
            "training target balanced accuracy": train_t_acc,
            "training subclass balanced accuracy": train_sub_acc,
            "training superclass balanced accuracy": train_super_acc,
        }

        train_l = {
            "training target loss": train_t_loss,
            "training subclass loss": train_sub_loss,
            "training superclass loss": train_super_loss
        }
        try:
            save_checkpoint(self.model, opt, scheduler, self.num_epochs, [], self.checkpoint_dir)
        except Exception:
            print("Error while saving")
        print("Training finished!")
        return val_balanced_accuracies, train_balanced_accuracies, val_accuracies, train_accuracies, train_l, self.model.module


def get_state_dict(checkpoint_dir):
    checkpoint_files = os.listdir(checkpoint_dir)
    if len(checkpoint_files) > 0:
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('-')[-1].split('.')[0][5:]))
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print("loading:", checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint_file)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            new_key = "module." + key
            new_state_dict[new_key] = value
        return new_state_dict
