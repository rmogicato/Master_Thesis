from shared_models import SharedNet
from Baseline import BaseNet
import missclassification_methods as missclassification_methods
from helper_functions import Imagenet_Backbone

subcluster_split = (7,5)
"""
Generally it's a good idea to initialize the data-set wrapper for all models to be compared,
as constructing it multiple times slows the process down.
"""
im_backbone = Imagenet_Backbone(num_epoch=1, n_superclusters=2, subclusters_split=(7, 5), batchsizes=(20, 20))

model_name = "BSN"

model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint BSN_01",
                          batchsizes=(10, 50), im_backbone=im_backbone)


# print("trainable params BSN", missclassification_methods.get_trainable_parameters(model_wrapper))

BSN_histogram_superclass, BSN_histogram_subclass, BSN_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features BSN_01", "00_results/BSN01 results/")


model_wrapper = BaseNet(num_epochs=1, model_string="Target", n_superclusters=2, subcluster_split=subcluster_split,
                          device_ids=(4), checkpoint_dir="checkpoint baseline target",
                          batchsizes=(80, 50), im_backbone=im_backbone)
model_super = BaseNet(num_epochs=1, model_string="Superclass", n_superclusters=2, subcluster_split=subcluster_split,
                          device_ids=(4), checkpoint_dir="checkpoint baseline superclass",
                          batchsizes=(80, 50), im_backbone=im_backbone)
model_sub = BaseNet(num_epochs=1, model_string="Subclass", n_superclusters=2, subcluster_split=subcluster_split,
                          device_ids=(4), checkpoint_dir="checkpoint baseline subclass",
                    batchsizes=(80, 50), im_backbone=im_backbone)

baseline_params = 0
baseline_params += missclassification_methods.get_trainable_parameters(model_wrapper)
baseline_params += missclassification_methods.get_trainable_parameters(model_super)
baseline_params += missclassification_methods.get_trainable_parameters(model_sub)
print("trainable params baseline", baseline_params)


color_map = ["#c71038", "#3fd7eb"]
names = ["Baseline", "BSN"]
baseline_histogram_superclass, baseline_histogram_subclass, baseline_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features baseline", "00_results/baseline results/", modelWrapper_subclass=model_sub, modelWrapper_superclass=model_super)
missclassification_methods.visualize_multiple_histograms([baseline_histogram_superclass, BSN_histogram_superclass], model_wrapper.supercluster_to_text, model_wrapper.supercluster_map,
                        '00_results/superclass_histogram_lines_baseline_BSN', color_map, names, class_type="Superclass")
missclassification_methods.visualize_multiple_histograms([baseline_histogram_subclass, BSN_histogram_subclass], model_wrapper.subcluster_to_text, model_wrapper.subcluster_map,
                        '00_results/subclass_histogram_lines_baseline_BSN', color_map, names)

model_name = "LSN"

model_wrapper = SharedNet(num_epochs=70, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint LSN_02",
                          batchsizes=(80, 50), im_backbone=im_backbone)

LSN_histogram_superclass, LSN_histogram_subclass, LSN_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features-50 LSN_02", "00_results/LSN_02/")

print("trainable params LSN", missclassification_methods.get_trainable_parameters(model_wrapper))


model_name = "ASN"

model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint ASN",
                          batchsizes=(80, 50), im_backbone=im_backbone)

# extracting misclassification of ASN
dst_of_misclassified_superclasses = "00_results/ASN_misclassified_images"
missclassification_methods.extract_incorrect_images(model_wrapper, dst_of_misclassified_superclasses)


ASN_histogram_superclass, ASN_histogram_subclass, ASN_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features-50 ASN", "00_results/ASN_results/")
print("trainable params ASN", missclassification_methods.get_trainable_parameters(model_wrapper))

model_name = "ASN-A"


model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint ASN_A",
                          batchsizes=(80, 50), im_backbone=im_backbone)


ASN_A_histogram_superclass, ASN_A_histogram_subclass, ASN_A_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features-50 ASN-A", "00_results/ASN-A_results/")
print("trainable params ASN-A", missclassification_methods.get_trainable_parameters(model_wrapper))


model_name = "ASN-T"


model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint ASN_02",
                          batchsizes=(80, 50), im_backbone=im_backbone)

ASN_T_histogram_superclass, ASN_T_histogram_subclass, ASN_T_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features-50 ASN-T", "00_results/ASN-T_results/")
print("trainable params ASN-T", missclassification_methods.get_trainable_parameters(model_wrapper))

model_name = "ASN-U"

model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint ASN-U",
                          batchsizes=(80, 50), im_backbone=im_backbone)


ASN_U_histogram_superclass, ASN_U_histogram_subclass, ASN_U_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features-50 ASN-U", "00_results/ASN-U_results/")
print("trainable params ASN U", missclassification_methods.get_trainable_parameters(model_wrapper))


model_name = "ASN-TS"

model_wrapper = SharedNet(num_epochs=1, model_string=model_name, n_superclusters=2, subcluster_split=(7, 5),
                          device_ids=(4), checkpoint_dir="checkpoint ASN-TS_01",
                          batchsizes=(80, 10), im_backbone=im_backbone)

# extracting misclassification of ASN-TS
dst_of_misclassified_superclasses = "00_results/ASN_TS_misclassified_images"
missclassification_methods.extract_incorrect_images(model_wrapper, dst_of_misclassified_superclasses)

ASN_TS_histogram_superclass, ASN_TS_histogram_subclass, ASN_TS_results = missclassification_methods.evaluate_model(model_wrapper, subcluster_split, "extracted features-50 ASN-TS_01", "00_results/ASN-TS_01_results/")
print("trainable params ASN-TS", missclassification_methods.get_trainable_parameters(model_wrapper))

"""
Below we can visualize the histograms
"""
color_map = ["#3fd7eb", "#949494", "#4ad493"]
names = ["BSN", "LSN", "ASN"]
missclassification_methods.visualize_multiple_histograms([BSN_histogram_subclass, LSN_histogram_subclass, ASN_histogram_subclass],  model_wrapper.subcluster_to_text, model_wrapper.subcluster_map,
                        '00_results/subclass_histogram_lines_BSN_ASN_LSN', color_map, names)

missclassification_methods.visualize_multiple_histograms([BSN_histogram_superclass, LSN_histogram_superclass, ASN_histogram_superclass],  model_wrapper.supercluster_to_text, model_wrapper.supercluster_map,
                        '00_results/superclass_histogram_lines_BSN_ASN_LSN', color_map, names, class_type="Superclass")


color_map = ["#4ad493", "#306e0a", "#de8928", "#342299", "#af19d4"]
names = ["ASN", "ASN-U","ASN-A", "ASN-T", "ASN-TS"]
missclassification_methods.visualize_multiple_histograms([ASN_histogram_subclass, ASN_U_histogram_subclass, ASN_A_histogram_subclass, ASN_T_histogram_subclass, ASN_TS_histogram_subclass],  model_wrapper.subcluster_to_text, model_wrapper.subcluster_map,
                        '00_results/subclass_histogram_lines_ASN_UATTS', color_map, names)

missclassification_methods.visualize_multiple_histograms([ASN_histogram_superclass, ASN_T_histogram_superclass, ASN_A_histogram_superclass, ASN_T_histogram_subclass, ASN_TS_histogram_subclass],  model_wrapper.supercluster_to_text, model_wrapper.supercluster_map,
                        '00_results/superclass_histogram_lines_ASN_UATTS', color_map, names, class_type="Superclass")

color_map = ["#c71038", "#3fd7eb", "#949494", "#4ad493", "#306e0a", "#de8928", "#342299", "#af19d4"]
names = ["Baseline", "BSN", "LSN","ASN", "ASN-U","ASN-A", "ASN-T", "ASN-TS"]

missclassification_methods.visualize_multiple_histograms([baseline_histogram_superclass, BSN_histogram_superclass, LSN_histogram_superclass,ASN_histogram_superclass, ASN_T_histogram_superclass, ASN_A_histogram_superclass, ASN_T_histogram_subclass, ASN_TS_histogram_subclass],  model_wrapper.supercluster_to_text, model_wrapper.supercluster_map,
                        '00_results/superclass_histogram_lines_ALL_ASN-TS', color_map, names, class_type="Superclass")

color_map = ["#c71038", "#3fd7eb", "#949494", "#4ad493", "#af19d4"]
names = ["Baseline", "BSN", "LSN","ASN", "ASN-TS"]

missclassification_methods.visualize_multiple_histograms([baseline_histogram_superclass, BSN_histogram_superclass, LSN_histogram_superclass,ASN_histogram_superclass, ASN_TS_histogram_subclass],  model_wrapper.supercluster_to_text, model_wrapper.supercluster_map,
                                                         '00_results/superclass_histogram_lines_presentation_ASN-TS', color_map, names, class_type="Superclass")

missclassification_methods.visualize_multiple_histograms([baseline_histogram_subclass, BSN_histogram_subclass, LSN_histogram_subclass,ASN_histogram_subclass, ASN_TS_histogram_subclass],  model_wrapper.subcluster_to_text, model_wrapper.subcluster_map,
                        '00_results/subclass_histogram_lines_presentation_ASN-TS', color_map, names)

