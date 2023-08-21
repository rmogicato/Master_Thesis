from shared_models import SharedNet
from Baseline import BaseNet
from open_class_methods import evaluate_model_openset, plot_oscr_curves
from helper_functions import OpenSet_Backbone

subcluster_split = (7,5)
unknown_levels = ["Superclasses", "Subclasses"]
for unknown_level in unknown_levels:

    os_b = OpenSet_Backbone(num_epoch=1, n_superclusters=2, subclusters_split=(7, 5), batchsizes=(20, 50), unknown=unknown_level)
    model_wrapper = BaseNet(num_epochs=1, model_string="Target", n_superclusters=2, subcluster_split=subcluster_split,
                              device_ids=(7,0), checkpoint_dir="checkpoint baseline target",
                              batchsizes=(80, 50), im_backbone=os_b)

    ccr_targets_baseline, fpr_targets_baseline = evaluate_model_openset(model_wrapper, f"openset_extracted_{unknown_level}_features_baseline")

    model_name = "BSN"

    model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                              device_ids=(4,), checkpoint_dir="checkpoint BSN_01",
                              batchsizes=(10, 50), im_backbone=os_b)


    ccr_targets_BSN, fpr_targets_BSN = evaluate_model_openset(model_wrapper, f"openset_extracted_{unknown_level}_features_BSN_new")

    model_name = "ASN-T"

    model_wrapper = SharedNet(num_epochs=70, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                              device_ids=(7,), checkpoint_dir="checkpoint ASN_02",
                              batchsizes=(10, 50), im_backbone=os_b)

    ccr_targets_ASN_T, fpr_targets_ASN_T = evaluate_model_openset(model_wrapper, f"openset_extracted_{unknown_level}_features_ASN-T")

    model_name = "LSN"

    model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                              device_ids=(4,), checkpoint_dir="checkpoint LSN_02",
                              batchsizes=(10, 50), im_backbone=os_b)
    ccr_targets_LSN, fpr_targets_LSN = evaluate_model_openset(model_wrapper, f"openset_extracted_{unknown_level}_features_LSN")

    model_name = "ASN"

    model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                              device_ids=(7,), checkpoint_dir="checkpoint ASN",
                              batchsizes=(10, 50), im_backbone=os_b)

    ccr_targets_ASN, fpr_targets_ASN = evaluate_model_openset(model_wrapper, f"openset_extracted_{unknown_level}_features_ASN")


    model_name = "ASN-TS"

    model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                              device_ids=(7,), checkpoint_dir="checkpoint ASN-TS_01",
                              batchsizes=(10, 50), im_backbone=os_b)

    ccr_targets_ASN_TS, fpr_targets_ASN_TS = evaluate_model_openset(model_wrapper, f"openset_extracted_{unknown_level}_features_ASN_TS_02")

    ccrs = [ccr_targets_baseline, ccr_targets_BSN, ccr_targets_ASN, ccr_targets_LSN, ccr_targets_ASN_TS]
    fprs = [fpr_targets_baseline, fpr_targets_BSN, fpr_targets_ASN, fpr_targets_LSN, fpr_targets_ASN_TS]

    color_map = ["#c71038", "#3fd7eb", "#4ad493", "#949494", "#af19d4"]
    names = ["Baseline", "BSN", "ASN",  "LSN", "ASN-TS"]

    plot_oscr_curves(ccrs, fprs, names, 5*["solid"], color_map, "02_results/", unknown_level)
