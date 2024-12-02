import numpy as np
import os
import pandas as pd
# plotting specifications
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from cycler import cycler
mpl.rcParams.update({'font.size': 14})
plt.rcParams['axes.axisbelow'] = True
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linewidth'] = 2
color_names = ['blue', 'red', 'green', 'black', 'magenta', 'y', 'slategray', 'darkorange']
mpl.rcParams['axes.prop_cycle'] = cycler(color=color_names)
pd.set_option('display.max_columns', None)
from nltk import ngrams
from collections import Counter


def uniformize_data(used_algorithm, **kwargs):

    if used_algorithm == 'ga':
            
            directory = kwargs.get('directory', None)
            n = kwargs.get('n', None)

            files = os.listdir(directory)
            ga_actions = np.empty([0,n*5])
            number_of_sequences = 0

            for file in files:
                if 'act_sequence' in file:
                    actions = np.genfromtxt(directory + file, dtype=int)
                    ga_actions = np.vstack([ga_actions,actions])

            return ga_actions

    if used_algorithm == 'zhang':
         file = kwargs.get('file',None)
         drl_actions = np.genfromtxt(file, dtype=int)
         return drl_actions
    
    if used_algorithm == 'sp':
        file = kwargs.get('file',None)
        drl_actions = []

        with open(file, 'r') as f:
            for line in f:
                sequence = line.split()
                sequence.pop()
                sequence = [int(x) for x in sequence]
                drl_actions.append(sequence)
            
            max_length = max(len(seq) for seq in drl_actions)
            padded_actions = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=-1) for seq in drl_actions])

        return padded_actions
                
def ngram(sequences, title, n, max_ngrams = 50, hex_code="#DDFFDD"):
    
    n_sequences = np.shape(sequences)[0]

    all_ngrams = []

    for i in range( np.shape(sequences)[0]):
        for j in range( np.shape(sequences)[1]):
            sequences[i,j] = int(sequences[i,j])

    
    for i in range(n_sequences):
        # Generate n-grams for each sequence
        sequence_ngrams = list(ngrams(sequences[i], n))
        sequence_ngrams = [tuple(int(x) for x in ngram) for ngram in sequence_ngrams]

        # Filter out n-grams that contain a negative value
        filtered_ngrams = [ngram for ngram in sequence_ngrams if all(x >= 0 for x in ngram)]
        all_ngrams.extend(filtered_ngrams)

    # Count n-gram frequencies
    ngram_counts = Counter(all_ngrams)
    total_count = sum(ngram_counts.values())

    # Get the "max_ngrams" most common n-grams
    most_common_ngrams = ngram_counts.most_common(max_ngrams)

    # Extract n-grams and their counts
    ngrams_list, counts = zip(*most_common_ngrams)
    
    # Plot histogram of n-gram frequencies
    figure, ax = plt.subplots(figsize=(12, 4))

    print(f'Total n-grams: {total_count}, Shown: {np.sum(counts)}')
    
    # Plot histogram
    ax.bar(range(len(ngrams_list)), counts, edgecolor="black", color= hex_code)
    
    # Configure x-ticks to show n-grams
    ax.set_xticks(range(len(ngrams_list)))
    ax.set_xticklabels([f"{'-'.join(map(str, ngram))}" for ngram in ngrams_list], rotation=90)

    # Configure y-ticks to show percentage of total n-grams
    max_value = int(np.max(counts))
    y = np.linspace(0, max_value, 10, dtype=int)
    ax.set_yticks(y)
    y_ticks = np.around(y * 100 / total_count, 2)
    ax.set_yticklabels(y_ticks)

    # Set grid, title, and labels
    plt.grid()
    plt.title(f"{title}. Total of {n_sequences} sequences")
    ax.set_xlabel("{}-gram".format(n))
    ax.set_ylabel("%")
    plt.tight_layout()
    plt.show()

def plot_contour(sequences, title="Contour Plot", xlabel="Time Step", ylabel="Action Number"):
    """
    Plots a contour plot for the given sequences.
    
    Parameters:
    - sequences: 2D numpy array where rows represent sequences and columns represent time steps.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    n_sequences, max_time_step = sequences.shape
    action_counts = np.zeros((int(np.max(sequences)), max_time_step))

    for i in range(n_sequences):
        for time_step in range(max_time_step):
            action = sequences[i, time_step]
            for action_index in range(int(np.max(sequences))):
                if action == action_index:  
                    action_counts[action_index, time_step] += 1

    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(range(max_time_step), range(action_counts.shape[0]))
    Z = action_counts

    contour = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(contour)
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()