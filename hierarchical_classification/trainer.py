import argparse
import os

import pandas as pd
import torch

import Baseline
from shared_models import SharedNet


def main(args):
    filename = args.filename

    num_epochs = args.num_epochs
    num_superclusters = args.num_superclasses
    # not the best way to get a tuple... but I suppose that most of the time this will be a default argument
    subclass_split = tuple(map(int, args.subclass_split.split(',')))
    device_ids = tuple(map(int, args.device_ids.split(',')))
    checkpoint = args.checkpoint
    model_name = args.model_name
    batch_sizes = tuple(map(int, args.batch_sizes.split(',')))

    df = pd.DataFrame(
        columns=['target accuracy', 'subclass accuracy', 'superclass accuracy', 'hierarchical accuracy'])
    df.to_csv(filename, index=False)

    print("Training: " + model_name)
    if model_name == "Baseline":
        os.makedirs(checkpoint + "_target", exist_ok=True)
        target_model = Baseline.BaseNet(num_epochs=num_epochs, model_string="Target",
                                        n_superclusters=num_superclusters, subcluster_split=subclass_split,
                                        device_ids=device_ids, checkpoint_dir=checkpoint + "_target",
                                        batchsizes=batch_sizes)

        target_results, _ = target_model.train()

        os.makedirs(checkpoint + "_subclass", exist_ok=True)
        subclass_model = Baseline.BaseNet(num_epochs=num_epochs, model_string="Subclass",
                                          n_superclusters=num_superclusters, subcluster_split=subclass_split,
                                          device_ids=device_ids, checkpoint_dir=checkpoint + "_subclass",
                                          batchsizes=batch_sizes)

        subclass_results, _ = subclass_model.train()

        os.makedirs(checkpoint + "_superclass", exist_ok=True)
        superclass_model = Baseline.BaseNet(num_epochs=num_epochs, model_string="Superclass",
                                            n_superclusters=num_superclusters, subcluster_split=subclass_split,
                                            device_ids=device_ids, checkpoint_dir=checkpoint + "_subclass",
                                            batchsizes=batch_sizes)
        superclass_results, _ = superclass_model.train()

        data = {**target_results, **subclass_results, **superclass_results}
        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename)

    else:
        os.makedirs(checkpoint, exist_ok=True)
        model_wrapper = SharedNet(num_epochs=num_epochs, model_string=model_name,
                                               n_superclusters=num_superclusters, subcluster_split=subclass_split,
                                               device_ids=device_ids, checkpoint_dir=checkpoint,
                                               batchsizes=batch_sizes)

        val_balanced_accuracies, train_balanced_accuracies, val_accuracies, train_accuracies, train_l, model = model_wrapper.train()
        torch.save(model.state_dict(), model_name + '.pth')
        all_acc = {}
        all_acc.update(val_balanced_accuracies)
        all_acc.update(train_balanced_accuracies)

        data = {**val_balanced_accuracies, **train_balanced_accuracies, **val_accuracies, **train_accuracies, **train_l}
        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a hierarchical classification model.')
    parser.add_argument('--model_name', type=str,
                        help='Model name, choose between Baseline, BSN, ASN, ASN-A, ASN-T, ASN-TS, and ASN-U',
                        required=True)
    parser.add_argument('--filename', type=str, default="results.csv",
                        help='Filename for storing results, must be .csv')
    parser.add_argument('--num_epochs', type=int, default=66, help='Number of epochs the network should be trained')
    parser.add_argument('--num_superclasses', type=int, default=2, help='Number of superclasses')
    parser.add_argument('--subclass_split', type=str, default="7,5",
                        help='Subclasses splits (i.e., size of each subclass from the superclasses). Please provide as comma-separated values')
    parser.add_argument('--device_ids', type=str, default="0",
                        help='Device IDs, provide as comma-separated values to train on multiple, e.g. "0,1"')
    parser.add_argument('--checkpoint', type=str, default="checkpoint", help='Checkpoint directory')
    parser.add_argument('--batch_sizes', type=str, default="80,50", help='Batch sizes for training and validation')

    args = parser.parse_args()
    main(args)
