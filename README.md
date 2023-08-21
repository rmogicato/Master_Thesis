# Master_Thesis
## Learning Semantics of Classes in Image Classification: Attention-Sharing between Hierarchies

### Installation instructions
All relevant packages can be found in `spec-file.txt`, install it using `conda create --name <env> --file spec-file.txt`.
Before executing any scripts, make sure the `.env` file correctly points to your data set,
e.g.:

`DATASET_PATH="/local/scratch/datasets/ImageNet/ILSVRC2012"`

Python 3.10 and PyTorch 1.11 were used for development.

### Structure
The directory `hierarchical_classification` contains all relevant files implemented in this thesis.
Generally, I would recommend reading the thesis provided for the background information that is done here.

In `hierarchical_classification` there are three files required to construct the WordNet hierarchy.
These files are available on the ImageNet website: [words.txt](https://image-net.org/data/words.txt), [wordnet.is_a.txt](https://image-net.org/data/wordnet.is_a.txt), and the third is contained in the imagenet dev_kit.

### Usage
There are several scripts that support a few use cases.
First, and most importantly, the training script `trainer.py`.
This script can be started with following arguments:
* --model_name: Model name, choose between "Baseline", "BSN", "ASN", "ASN-A", "ASN-T", "ASN-TS", and "ASN-U"
* --filename: Filename for storing results. _Default="results.csv"_
* --num_epoch Number of epochs the network should be trained. _Default=66_
* --num_superclasses Number of superclasses. _Default: 2_
* --subclasses_split: Subclasses splits (i.e., size of each subclass from the superclasses). Please provide as comma-separated value. _Default="7,5"_
* --device_ids: Device IDs, provide as comma-separated values, e.g. "0,". _Default="0,"_
* --checkpoint: Checkpoint directory. _Default="checkpoint"_

The default parameters match the parameters used in the thesis, so only the model_name needs to be specified.
(I highly recommend also setting a custom checkpoint and filename to save the results)

This allows for training of a single model.
As all analysis compares different models, the other scripts can't be executed with arguments (as a comparison of multiple models would require a large number of arguments).
Instead, the three scripts should serve as template on how one can use the provided functions.
* `evaluate_missclassifications.py` shows how the semantic conditioning, balanced accuracy, confusion matrices and misclassified images can be created.
* `evaluate_open_set.py` shows how the open-set performance using methods from `open_class_methods.py`
* `visualize_plots.py` is a simple visualization file which uses methods from `visualization_helper.py`

### Python issues
If you use Python 3.10 and PyTorch 1.11, you may encounter an error message related to the `collections` package.
This is a Python 3.10 issue with a relatively easy fix found [here](https://discuss.pytorch.org/t/issues-on-using-nn-dataparallel-with-python-3-10-and-pytorch-1-11/146745/16).
