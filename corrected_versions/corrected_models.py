import torch
import torch.nn as nn
import torchvision.models as models
"""
This file contains the corrected versions.
The bug only exists for the ASN, ASN-TS and ASN-U
"""
class ASN_TS_correct(nn.Module):
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(ASN_TS_correct, self).__init__()
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

    def forward(self, x, superclass=None, subclass=None):
        # Compute features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # Predict superclass
        super_logits = self.fc_super(features)

        # get superclass probabilities
        if superclass is None:
            super_probs = self.softmax(super_logits)  # getting predictions
            super_probability_weights = torch.matmul(super_probs, self.fc_super.weight)

        else:
            # use ground truth labels if a superclass is given
            super_probability_weights = self.fc_super.weight[superclass]
        # and use it to weight the features
        fused_super_features = torch.mul(features, super_probability_weights)

        sub_logits = self.fc_sub(fused_super_features)

        if subclass is None:
            sub_probs = self.softmax(sub_logits)
            sub_probability_weights = torch.matmul(sub_probs, self.fc_sub.weight)
            W_sub = torch.sum(sub_probability_weights, dim=0)
        else:
            W_sub = self.fc_sub.weight[subclass]

        sub_fused_features = torch.mul(features, W_sub)
        target_logits = self.fc_t(sub_fused_features)

        return target_logits, sub_logits, super_logits



class ASN_correct(nn.Module):
    """
    This is the fixed version where I removed the sum across batches
    """
    def __init__(self, num_superclasses, num_subclasses, num_targetclasses):
        super(ASN_correct, self).__init__()
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

        # Predict superclass
        super_logits = self.fc_super(features)
        # get superclass probabilities
        super_probs = self.softmax(super_logits)  # hardmax, ground truth?

        W_super = torch.matmul(super_probs, self.fc_super.weight)
        # and use it to weight the features
        fused_super_features = torch.mul(features, W_super)

        sub_logits = self.fc_sub(fused_super_features)
        sub_probs = self.softmax(sub_logits)
        # and use it to weight the features
        W_sub = torch.matmul(sub_probs, self.fc_sub.weight)

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
        W_t = torch.matmul(target_probs, self.fc_t.weight)

        target_fused_features = torch.mul(features, W_t)

        # the fused features are now used for subclass prediction
        sub_logits = self.fc_sub(target_fused_features)
        sub_probs = self.softmax(sub_logits)
        W_sub = torch.matmul(sub_probs, self.fc_sub.weight)

        sub_fused_features = torch.mul(features, W_sub)

        # Predict superclass
        super_logits = self.fc_super(sub_fused_features)

        return target_logits, sub_logits, super_logits