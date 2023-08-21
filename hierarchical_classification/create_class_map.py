from robustness.tools.imagenet_helpers import ImageNetHierarchy
from dotenv import dotenv_values


def get_map(in_hier, n_classes, ancestor='n00001740'):
    """
    This function gets the superclass_wnid, class ranges, label maps.
    - in_hier: ImageNetHierarchy object
    - n_classes: number of desired superclasses
    """
    superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(n_classes,
                                                                        ancestor_wnid=ancestor,
                                                                        balanced=False)

    return superclass_wnid, class_ranges, label_map

def get_imagenet_map(n_superclusters, n_subclusters):
    """
    :param n_superclusters: int, number of superclasses
    :param n_subclusters: either tuple of size of each subclass or int if both subclasses should be the same
    :return:
    """

    env = dotenv_values("../.env")
    in_path = env['DATASET_PATH']
    in_info_path = "../meta_data"

    if isinstance(n_subclusters, int):
        n_subclusters = n_superclusters * [n_subclusters]

    assert len(
        n_subclusters) == n_superclusters, "please make sure n_subclusters is either an integer or a tuple of size n_superclusters"
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid, class_ranges, label_map = get_map(in_hier, n_superclusters)

    supercluster_map = {}
    subcluster_map = {}
    supercluster_to_text = {}
    subcluster_to_text = {}
    subclass_winds = []
    for idx, super in enumerate(superclass_wnid):
        supercluster_to_text[idx] = in_hier.wnid_to_name[super]
        subclass_wind, class_ranges, label_map = get_map(in_hier, n_subclusters[idx], super)
        subclass_winds += subclass_wind
        for sub_idx, class_range in enumerate(class_ranges):
            if len(n_subclusters) > 1:
                prev_idx = n_subclusters[idx - 1]
            else:
                prev_idx = 0
            class_nr = idx * prev_idx + sub_idx
            subcluster_to_text[class_nr] = label_map[sub_idx]
            for c in class_range:
                subcluster_map[c] = class_nr
                supercluster_map[c] = idx

    return supercluster_map, subcluster_map, supercluster_to_text, subcluster_to_text, subclass_winds


def get_openset_map(n_superclusters, n_subclusters):
    """
    This function creates a class map for open-set classification.
    It works the same as the get_imagenet_map function, but unknown classes are also returned
    :param n_superclusters: number of superclusters for openset classification
    :param n_subclusters:
    :return:
    """
    in_path = "/local/scratch/datasets/ImageNet/ILSVRC2012"
    in_info_path = "../meta_data"

    all_classes = set(range(1000))
    if isinstance(n_subclusters, int):
        n_subclusters = n_superclusters * [n_subclusters]

    assert len(
        n_subclusters) == n_superclusters, "please make sure n_subclusters is either an integer or a tuple of size n_superclusters"
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid, class_ranges, label_map = get_map(in_hier, n_superclusters)

    superclass_class_ranges = set().union(*class_ranges)
    unknown_superclasses = superclass_class_ranges ^ all_classes
    print("Number of classes that do not belong to an unknown superclass: ", len(unknown_superclasses))
    supercluster_map = {}
    subcluster_map = {}
    supercluster_to_text = {}
    subcluster_to_text = {}
    subclass_winds = []
    for idx, super in enumerate(superclass_wnid):
        supercluster_to_text[idx] = in_hier.wnid_to_name[super]
        subclass_wind, class_ranges, label_map = get_map(in_hier, n_subclusters[idx], super)
        subclass_winds += subclass_wind
        for sub_idx, class_range in enumerate(class_ranges):
            if len(n_subclusters) > 1:
                prev_idx = n_subclusters[idx - 1]
            else:
                prev_idx = 0
            class_nr = idx * prev_idx + sub_idx
            subcluster_to_text[class_nr] = label_map[sub_idx]
            for c in class_range:
                subcluster_map[c] = class_nr
                supercluster_map[c] = idx

    know_subclasses = set(subcluster_map.keys())
    unknown_subclasses = (know_subclasses ^ all_classes) ^ unknown_superclasses
    print("Number of classes that do not belong to an unknown subclass: ", len(unknown_subclasses))
    return supercluster_map, subcluster_map, supercluster_to_text, subcluster_to_text, subclass_winds, unknown_subclasses, unknown_superclasses
