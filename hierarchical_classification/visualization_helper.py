import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_keys_separately(dataframes, labels):
    columns_lists = []

    for df in dataframes:
        columns = df.columns.tolist()
        columns_lists.append(set(columns))
    common_values = set.intersection(*columns_lists)

    plt.figure(figsize=(10, 3))

    for key in common_values:
        for idx, df in enumerate(dataframes):
            values_for_keys = df[key].to_list()
            if isinstance(values_for_keys[0], str) and values_for_keys[0].startswith("tensor"):
                values_for_keys = [float(s.strip('tensor()')) for s in values_for_keys]
            plt.plot(values_for_keys, label=labels[idx])  # Plot values for the current key

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(key)
        plt.legend()  # Show a legend indicating the keys
        plt.show()  # Display the plot

def plot_accuracy_area(scores, color, label):
    min_scores = np.min(scores, axis=0)
    max_scores = np.max(scores, axis=0)
    avg_scores = np.mean(scores, axis=0)
    plt.fill_between(range(len(min_scores)), min_scores, max_scores, color=color, alpha=0.4,label=label)
    plt.plot(avg_scores, color=color, linestyle="dotted")

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

def plot_accuracy_ranges(list_of_lists, labels, keys_to_be_printed, title, color_map = None, path = None):
    plt.figure(figsize=(10, 4))
    for key in keys_to_be_printed:
        for idx, model in enumerate(list_of_lists):
            scores = []
            for run in model:
                scores.append(run[key].to_list())
                values_for_keys = run[key].to_list()
            plot_accuracy_area(scores, color_map[idx], label=labels[idx])

    tick_positions = range(4, len(values_for_keys), 5)  # python uses zero indexing
    tick_labels = [str(i + 1) for i in tick_positions]  # shift labels by +1
    # Set x-ticks labels
    plt.xticks(tick_positions, tick_labels)
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()

def plot_train_and_test_accuracies(dataframes, labels, keys_to_be_printed, title, color_map = None, path = None):

    fig = plt.figure(figsize=(10, 4))
    if not color_map:
        color_map = ["#3fd7eb", "#de2016", "#110c99", "#de4ce6"]
    symbol_map = ["solid", "dashed", "dotted"]
    for idx_key, key in enumerate(keys_to_be_printed):
        if "training" in key:
            mode_label = " training"
        else:
            mode_label = " validation"
        for idx, df in enumerate(dataframes):
            values_for_keys = df[key].to_list()
            if isinstance(values_for_keys[0], str) and values_for_keys[0].startswith("tensor"):
                values_for_keys = [float(s.strip('tensor()')) for s in values_for_keys]

                print(values_for_keys)
                print(key, color_map[idx])

            plt.plot(values_for_keys, label=labels[idx] + mode_label, color=color_map[idx], linestyle=symbol_map[idx_key])  # Plot values for the current key
            # Create new labels for x-axis
            tick_positions = range(4, len(values_for_keys), 5)  # python uses zero indexing
            tick_labels = [str(i + 1) for i in tick_positions]  # shift labels by +1
            # Set x-ticks labels
            plt.xticks(tick_positions, tick_labels)

    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title(title)
    plt.legend()  # Show a legend indicating the keys
    if path:
        plt.savefig(path)
    else:
        plt.show()  # Display the plot


def generate_latex_table(df_list, df_names, column_names, table_name):
    rename_dict = {
        "validation superclass balanced accuracy": "Superclass",
        "validation subclass balanced accuracy": "Subclass",
        "validation target balanced accuracy": "Target Class"
    }

    # Create a dictionary to store the last value of each column for each dataframe
    df_dict = {}
    for df, name in zip(df_list, df_names):
        df_dict[name] = [format(df[col].iloc[-1], '.2f') for col in column_names]

    # Rename columns
    result_df = pd.DataFrame(df_dict, index=[rename_dict.get(col, col) for col in column_names])

    # Highlight the max value in each row
    for idx, row in result_df.iterrows():
        max_val = max(row, key=lambda x: float(x))
        result_df.loc[idx] = ['\\textbf{{{}}}\%'.format(val) if val == max_val else '{}\%'.format(val) for val in row]

    # Convert dataframe to latex table
    latex_table = result_df.to_latex(escape=False)
    latex_table = latex_table.replace('\\\\\n', '\\\\\n\\hline\n', 1)
    latex_table = latex_table.replace('\\toprule', '')
    latex_table = latex_table.replace('\\midrule', '')
    latex_table = latex_table.replace('\\bottomrule', '')
    latex_table = latex_table.replace('{}', 'Model', 1)

    # Save the latex table as txt document
    with open(table_name, "w") as f:
        f.write(latex_table)

def load_base_line_files(target_path, subclass_path, superclass_path=None):
    target = pd.read_csv(target_path, index_col=0)
    subclass = pd.read_csv(subclass_path, index_col=0)
    superclass = pd.read_csv(superclass_path, index_col=0)

    target.columns = ["validation target accuracy","validation target balanced accuracy", "training target balanced accuracy","training target loss"]
    subclass.columns = ["validation subclass accuracy","validation subclass balanced accuracy","training subclass balanced accuracy", "training subclass loss"]
    superclass.columns = ["validation superclass accuracy","validation superclass balanced accuracy","training superclass balanced accuracy", "training superclass loss"]

    result = target.join(subclass)
    result = result.join(superclass)
    return result

