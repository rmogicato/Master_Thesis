import pandas as pd
from hierarchical_classification.visualization_helper import plot_train_and_test_accuracies, plot_accuracy_ranges, generate_latex_table, load_base_line_files


df_BSN_01 = pd.read_csv("results/Shared Training - 7-5 split/correct_results_BSN_03.csv", index_col=0)
df_BSN_02 = pd.read_csv("results/Shared Training - 7-5 split/results_12_06_BSN_01.csv", index_col=0)
df_BSN_03 = pd.read_csv("results/Shared Training - 7-5 split/results_12_06_BSN_02.csv", index_col=0)

df_ASN_TS_01 = pd.read_csv("results/ASN-TS/correct_results_ASN_TS.csv", index_col=0)
df_ASN_TS_02 = pd.read_csv("results/ASN-TS/correct_results_ASN-TS_01.csv", index_col=0)
df_ASN_TS_03 = pd.read_csv("results/ASN-TS/correct_results_ASN_TS_02.csv", index_col=0)

df_ASN_U_01 = pd.read_csv("results/ASN-U/correct_results_ASN-U.csv", index_col=0)
df_ASN_U_02 = pd.read_csv("results/ASN-U/correct_results_ASN_U_02.csv", index_col=0)
df_ASN_U_03 = pd.read_csv("results/ASN-U/correct_results_ASN_U_03.csv", index_col=0)
names = ["BSN", "ASN-TS", "ASN-U"]
colors=["#3fd7eb", "#af19d4", "#306e0a"]

plot_accuracy_ranges([[df_BSN_01, df_BSN_02, df_BSN_03], [df_ASN_TS_01, df_ASN_TS_02, df_ASN_TS_03], [df_ASN_U_01, df_ASN_U_02, df_ASN_U_03]], names, keys_to_be_printed=["validation target balanced accuracy"], title='Ranges of Validation Scores Target Class', color_map=colors, path="results/01_results/range_target_classes.pdf")
plot_accuracy_ranges([[df_BSN_01, df_BSN_02, df_BSN_03], [df_ASN_TS_01, df_ASN_TS_02, df_ASN_TS_03], [df_ASN_U_01, df_ASN_U_02, df_ASN_U_03]], names, keys_to_be_printed=["validation subclass balanced accuracy"], title='Ranges of Validation Scores Subclass', color_map=colors, path="results/01_results/range_subclass_classes.pdf")
plot_accuracy_ranges([[df_BSN_01, df_BSN_02, df_BSN_03], [df_ASN_TS_01, df_ASN_TS_02, df_ASN_TS_03], [df_ASN_U_01, df_ASN_U_02, df_ASN_U_03]], names, keys_to_be_printed=["validation superclass balanced accuracy"], title='Ranges of Validation Scores Superclass', color_map=colors, path="results/01_results/range_superclasses_classes.pdf")


path1 = "results/Shared Training - 7-5 split/results_12_06_BSN_01.csv"
path2 = "results/Logit Sharing 12 clases/correct_results_LSN_02.csv"
path3 = "results/ASN/correct_results_ASN.csv"
path4 = "results/ASN-U/correct_results_ASN-U.csv"
path5 = "results/ASN-T/correct_results_ASN_02.csv"
path6 = "results/ASN-A/correct_results_ASN_A.csv"
path7 = "results/ASN-TS/correct_results_ASN_TS.csv"
# path3 = "results/LogitSharing No SE/results_03_05.csv"
df1 = pd.read_csv(path1, index_col=0)
df2 = pd.read_csv(path2, index_col=0)
df3 = pd.read_csv(path3, index_col=0)
df4 = pd.read_csv(path4, index_col=0)
df5 = pd.read_csv(path5, index_col=0)
df6 = pd.read_csv(path6, index_col=0)
df7 = pd.read_csv(path7, index_col=0)
df_baseline = load_base_line_files("results/baseline/target_baseline/results_17_05.csv", "results/baseline/subclass baseline/results_22_05.csv", "results/baseline/superclass baseline/results_25_05.csv")

table_columns = ["validation superclass balanced accuracy", "validation subclass balanced accuracy", "validation target balanced accuracy"]
color_map = ["#3fd7eb", "#de4ce6", "#4ad493"]
dfs = [df1, df2, df3]
names = ["BSN", "LSN", "ASN"]
plot_train_and_test_accuracies(dfs, names, ["validation target balanced accuracy", "training target balanced accuracy"], "Training and Validation Accuracy for Target Classes", color_map=color_map)
plot_train_and_test_accuracies(dfs, names, ["validation subclass balanced accuracy", "training subclass balanced accuracy"], "Training and Validation Accuracy for Subclasses", color_map=color_map)
plot_train_and_test_accuracies(dfs, names, ["validation superclass balanced accuracy", "training superclass balanced accuracy"], "Training and Validation Accuracy for Superclasses", color_map=color_map)
generate_latex_table(dfs, names, table_columns, "latex_tables/BSN_ASN_LSN.txt")

color_map = ["#4ad493", "#306e0a", "#de8928", ]
plot_train_and_test_accuracies([df3, df4, df6], ["ASN", "ASN-U","ASN-A"], ["validation target balanced accuracy", "training target balanced accuracy"], "Training and Validation Accuracy for Target Classes", color_map=color_map)
plot_train_and_test_accuracies([df3, df4,df6], ["ASN", "ASN-U","ASN-A"], ["validation subclass balanced accuracy", "training subclass balanced accuracy"], "Training and Validation Accuracy for Subclasses", color_map=color_map)
plot_train_and_test_accuracies([df3, df4, df6], ["ASN", "ASN-U","ASN-A"], ["validation superclass balanced accuracy", "training superclass balanced accuracy"], "Training and Validation Accuracy for Superclasses", color_map=color_map)

color_map = ["#4ad493", "#306e0a", "#de8928", "#342299", "#af19d4"]
names = ["ASN", "ASN-U","ASN-A", "ASN-T", "ASN-TS"]
dfs = [df3, df4, df6, df5, df7]
plot_train_and_test_accuracies(dfs, names, ["validation target balanced accuracy", "training target balanced accuracy"], "Training and Validation Accuracy for Target Classes", color_map=color_map)
plot_train_and_test_accuracies(dfs, names, ["validation subclass balanced accuracy", "training subclass balanced accuracy"], "Training and Validation Accuracy for Subclasses", color_map=color_map)
plot_train_and_test_accuracies(dfs, names, ["validation superclass balanced accuracy", "training superclass balanced accuracy"], "Training and Validation Accuracy for Superclasses", color_map=color_map)
names = ["BSN", "ASN", "ASN-U","ASN-A", "ASN-T", "ASN-TS"]
dfs = [df1, df3, df4, df6, df5, df7]
generate_latex_table(dfs, names, table_columns, "latex_tables/ASN_variations.txt")

dfs = [df1, df3, df_baseline, df4, df7]
names = ["BSN", "ASN", "Baseline", "ASN-U", "ASN-TS"]
color_map = ["#3fd7eb", "#4ad493", "#d61c25", "#306e0a", "#af19d4"]
plot_train_and_test_accuracies(dfs, names, ["validation target balanced accuracy", "training target balanced accuracy"], "Target Classes", color_map=color_map, path="baseline/Training_Val_Accuracies_Target.pdf")
plot_train_and_test_accuracies(dfs, names, ["validation subclass balanced accuracy", "training subclass balanced accuracy"], "Subclasses", color_map=color_map, path="baseline/Training_Val_Accuracies_Sub.pdf")
plot_train_and_test_accuracies(dfs, names, ["validation superclass balanced accuracy", "training superclass balanced accuracy"], "Superclasses", color_map=color_map, path="baseline/Training_Val_Accuracies_Sup.pdf")
generate_latex_table(dfs, names, table_columns, "latex_tables/baseline_BSN.txt")


color_map = ["#3fd7eb", "#d61c25"]
dfs = [df1, df_baseline]
names = ["BSN", "Baseline"]
plot_train_and_test_accuracies(dfs,names , ["validation superclass balanced accuracy", "training superclass balanced accuracy"], "Training and Validation Accuracy for Superclasses", color_map=color_map)
plot_train_and_test_accuracies(dfs, names, ["validation subclass balanced accuracy", "training subclass balanced accuracy"], "Training and Validation Accuracy for Subclasses", color_map=color_map)
plot_train_and_test_accuracies(dfs, names, ["validation target balanced accuracy", "training target balanced accuracy"], "Training and Validation Accuracy for Target Classes", color_map=color_map)
