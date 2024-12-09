import numpy as np
import os
import pandas as pd
from dgamod import *

# plotting specifications
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from cycler import cycler

mpl.rcParams.update({"font.size": 14})
plt.rcParams["axes.axisbelow"] = True
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2
color_names = [
    "blue",
    "red",
    "green",
    "black",
    "magenta",
    "y",
    "slategray",
    "darkorange",
]
mpl.rcParams["axes.prop_cycle"] = cycler(color=color_names)
pd.set_option("display.max_columns", None)
from nltk import ngrams
from collections import Counter


def uniformize_data(used_algorithm, **kwargs):

    if used_algorithm == "ga":

        directory = kwargs.get("directory", None)
        n = kwargs.get("n", None)

        files = os.listdir(directory)
        ga_actions = np.empty([0, n * 5])
        number_of_sequences = 0

        for file in files:
            if "act_sequence" in file:
                actions = np.genfromtxt(directory + file, dtype=int)
                ga_actions = np.vstack([ga_actions, actions])

        return ga_actions

    if used_algorithm == "zhang":
        file = kwargs.get("file", None)
        drl_actions = np.genfromtxt(file, dtype=int)
        return drl_actions

    if used_algorithm == "sp":
        file = kwargs.get("file", None)
        drl_actions = []

        with open(file, "r") as f:
            for line in f:
                sequence = line.split()
                sequence.pop()
                sequence = [int(x) for x in sequence]
                drl_actions.append(sequence)

            max_length = max(len(seq) for seq in drl_actions)
            padded_actions = np.array(
                [
                    np.pad(
                        seq, (0, max_length - len(seq)), "constant", constant_values=-1
                    )
                    for seq in drl_actions
                ]
            )

        return padded_actions


def ngram(sequences, title, n, max_ngrams=50, hex_code="#DDFFDD"):

    n_sequences = np.shape(sequences)[0]

    all_ngrams = []

    for i in range(np.shape(sequences)[0]):
        for j in range(np.shape(sequences)[1]):
            sequences[i, j] = int(sequences[i, j])

    for i in range(n_sequences):
        # Generate n-grams for each sequence
        sequence_ngrams = list(ngrams(sequences[i], n))
        sequence_ngrams = [tuple(int(x) for x in ngram) for ngram in sequence_ngrams]

        # Filter out n-grams that contain a negative value
        filtered_ngrams = [
            ngram for ngram in sequence_ngrams if all(x >= 0 for x in ngram)
        ]
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

    print(f"Total n-grams: {total_count}, Shown: {np.sum(counts)}")

    # Plot histogram
    ax.bar(range(len(ngrams_list)), counts, edgecolor="black", color=hex_code)

    # Configure x-ticks to show n-grams
    ax.set_xticks(range(len(ngrams_list)))
    ax.set_xticklabels(
        [f"{'-'.join(map(str, ngram))}" for ngram in ngrams_list], rotation=90
    )

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


def plot_contour(
    sequences, title="Contour Plot", xlabel="Time Step", ylabel="Action Number"
):
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

    contour = plt.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar(contour)
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# calculation of state properties 

def state_fidelity(state):
    nh = np.shape(state)[0]
    fid = np.real(state[nh - 1] * np.conjugate(state[nh - 1]))
    return fid

def calc_exp_value(state, op):
    val = np.matmul(np.conjugate(np.transpose(state)),np.matmul(op, state))
    return np.real(val)

def calc_ipr(state):
    nh = np.shape(state)[0]
    ipr = 0
    for i in range(nh):
        ipr += np.real(state[i]*np.conjugate(state[i]))**2
    return ipr

def calc_localization(state):
    nh = np.shape(state)[0]
    loc = 0
    for i in range(nh):
        loc += np.real(state[i]*np.conjugate(state[i]))**2*(i+1)
    return loc

def action_selector(actions_name,b,nh):

    if actions_name == 'original':
        actions = actions_zhang(b, nh)
    elif actions_name == 'oaps':
        actions = one_field_actions(b,nh)
    elif actions_name == 'extra':
        actions = one_field_actions_extra(b,nh)
    
    return actions

# plots for action sequences

def plot_single_sequence(action_sequence,nh,dt=0.15,b=100,label='',actions = 'original',add_natural=False):
    
    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence)+1
    
    # generar propagadores
    actions = action_selector(actions,b,nh)
    propagators = gen_props(actions, nh, b, dt)
    times = np.arange(0,t_steps,1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    free_state = initial_state
    
    if add_natural:
        natural_evol = [state_fidelity(free_state)]
    
        nat_sequence = np.zeros(int(t_steps-1),dtype=int)

        for action in nat_sequence:
                
            free_state = calculate_next_state(free_state,0,propagators)
            natural_evol.append(state_fidelity(free_state))

        max_natural = np.max(natural_evol)

        plt.plot(times,natural_evol, '-v', label = 'Natural Evolution , Máx: {}'.format(max_natural))
    

    # inicializacion de estado forzado 
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    forced_evol = [state_fidelity(forced_state)]

    for action in action_sequence:
        
        forced_state = calculate_next_state(forced_state,action,propagators)
        forced_evol.append(state_fidelity(forced_state))

    max_forced = np.max(forced_evol)

    plt.plot(times,forced_evol,'-o', label = label + '. Máx.: {}'.format(max_forced))

    plt.legend(loc='upper left')
    #plt.show()

def plot_exp_value(action_sequence,nh,dt=0.15,b=100,label='',actions = 'original',add_actions=False):
    
    if add_actions:
        add_actions = plt.subplots()

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence)
    
    # generar propagadores
    actions = action_selector(actions,b,nh)
    propagators = gen_props(actions, nh, b, dt)
    times = np.arange(0,t_steps,1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    zero_action = actions_zhang(b,nh)[0]

    # inicializacion de estado forzado 
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    exp_values = []

    for action in action_sequence:
        
        forced_state = calculate_next_state(forced_state,action,propagators)
        exp_values.append(calc_exp_value(forced_state,actions[action]))

    max_exp_val = np.max(exp_values)

    plt.plot(times,exp_values,'-o', label = label + '. Máx.: {}'.format(max_exp_val))

    plt.legend(loc='upper left')
    
    if add_actions:
        ax2 = add_actions[1].twinx()

        color = 'tab:grey'
        ax2.plot(action_sequence,'--o',label='Acciones',color=color)

        ax2.tick_params(axis='y', labelcolor=color)

        add_actions[0].tight_layout()
    
def plot_ipr(action_sequence,nh,dt=0.15,b=100,label='',actions = 'original',add_actions=False):
    
    if add_actions:
        add_actions = plt.subplots()

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence)+1
    
    # generar propagadores
    actions = action_selector(actions,b,nh)        
    propagators = gen_props(actions, nh, b, dt)
    times = np.arange(0,t_steps,1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    zero_action = actions_zhang(b,nh)[0]

    # inicializacion de estado forzado 
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    ipr_values = [calc_ipr(forced_state)]

    for action in action_sequence:
        
        forced_state = calculate_next_state(forced_state,action,propagators)
        ipr_values.append(calc_ipr(forced_state))

    max_exp_val = np.max(ipr_values)

    plt.plot(times,ipr_values,'-o', label = label + '. Máx.: {}'.format(max_exp_val))

    plt.legend(loc='upper left')
    
    if add_actions:
        ax2 = add_actions[1].twinx()

        color = 'tab:grey'
        ax2.plot(action_sequence,'--o',label='Acciones',color=color)

        ax2.tick_params(axis='y', labelcolor=color)

        add_actions[0].tight_layout()

def plot_localization(action_sequence,nh,dt=0.15,b=100,label='',actions = 'original',add_actions=False):
    
    if add_actions:
        add_actions = plt.subplots()

    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence)+1
    
    # generar propagadores
    actions = action_selector(actions,b,nh)
    propagators = gen_props(actions, nh, b, dt)
    times = np.arange(0,t_steps,1)

    # definicion del estado inicial e inicializacion de estados forzado y natural

    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0

    zero_action = actions_zhang(b,nh)[0]

    # inicializacion de estado forzado 
    forced_state = initial_state

    # almacenar evolucion natural y evolucion forzada
    loc_values = [calc_localization(forced_state)]

    for action in action_sequence:
        
        forced_state = calculate_next_state(forced_state,action,propagators)
        loc_values.append(calc_localization(forced_state))

    max_loc_values = np.max(loc_values)

    plt.plot(times,loc_values,'-o', label = label + '. Máx.: {}'.format(max_loc_values))

    plt.legend(loc='upper left')
    
    if add_actions:
        ax2 = add_actions[1].twinx()

        color = 'tab:grey'
        ax2.plot(action_sequence,'--o',label='Acciones',color=color)

        ax2.tick_params(axis='y', labelcolor=color)

        add_actions[0].tight_layout()
    
def plot_all_metrics(action_sequence, nh, dt=0.15, b=100, label='', actions='original', add_natural=False, add_actions=False):
    """
    Genera una grilla 2x2 con las siguientes gráficas:
    - Fidelidad (single_sequence)
    - IPR
    - Valor esperado (exp_value)
    - Localización
    """
    action_sequence = [int(x) for x in action_sequence]
    t_steps = len(action_sequence) + 1
    times = np.arange(0, t_steps, 1)
    
    # Generar propagadores
    actions = action_selector(actions,b,nh)
    propagators = gen_props(actions, nh, b, dt)
    
    # Inicializar estado inicial
    initial_state = np.zeros(nh, dtype=np.complex_)
    initial_state[0] = 1.0
    
    # Configurar la figura
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs = axs.ravel()  # Facilita el acceso a los subplots
    
    # Función auxiliar para agregar acciones
    def plot_actions(ax):
        if add_actions:
            ax2 = ax.twinx()
            color = 'tab:grey'
            ax2.plot(range(len(action_sequence)), action_sequence, '--o', label='Acciones', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')

    # 1. Fidelidad
    free_state = initial_state
    forced_state = initial_state
    forced_evol = [state_fidelity(forced_state)]
    
    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        forced_evol.append(state_fidelity(forced_state))
    axs[0].plot(times, forced_evol, '-o', label=f"{label}. Máx.: {max(forced_evol)}")
    axs[0].set_title("Fidelidad")
    axs[0].legend(loc='upper left')
    
        
    if 0 in add_actions:
        plot_actions(axs[0])
    
    # 2. IPR
    forced_state = initial_state
    ipr_values = [calc_ipr(forced_state)]
    
    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        ipr_values.append(calc_ipr(forced_state))
    axs[1].plot(times, ipr_values, '-o', label=f"{label}. Máx.: {max(ipr_values)}")
    axs[1].set_title("IPR")
    axs[1].legend(loc='upper left')

    if 1 in add_actions:
        plot_actions(axs[1])
    
    # 3. Valor esperado
    forced_state = initial_state
    exp_values = []
    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        exp_values.append(calc_exp_value(forced_state, actions[action]))
    axs[2].plot(times[:-1], exp_values, '-o', label=f"{label}. Máx.: {max(exp_values)}")
    axs[2].set_title("Valor Esperado")
    axs[2].legend(loc='upper left')
    if 2 in add_actions:
        plot_actions(axs[2])
    
    # 4. Localización
    forced_state = initial_state
    loc_values = [calc_localization(forced_state)]
    for action in action_sequence:
        forced_state = calculate_next_state(forced_state, action, propagators)
        loc_values.append(calc_localization(forced_state))
    axs[3].plot(times, loc_values, '-o', label=f"{label}. Máx.: {max(loc_values)}")
    axs[3].set_title("Localización")
    axs[3].legend(loc='upper left')

    if 3 in add_actions:
        plot_actions(axs[3])    

    if add_natural:
        natural_evol = [state_fidelity(free_state)]
        
        # generar propagadores
        actions = action_selector('original',b,nh)
        propagators = gen_props(actions, nh, b, dt)
        
        for action in action_sequence:
            free_state = calculate_next_state(free_state, 0, propagators)
            natural_evol.append(state_fidelity(free_state))
        axs[0].plot(times, natural_evol, '-v', label="Evolución natural")
        axs[0].legend(loc='upper left')
    # Ajustar diseño
    plt.tight_layout()
    plt.show()


def find_max(action_sequences,nh, b=100, dt = 0.15, actions = 'original'):

    max_fid = 0.
    max_index = 0.

    # generar propagadores
    actions = action_selector(actions,b,nh)
    propagators = gen_props(actions, nh, b, dt)

    for i in range(np.shape(action_sequences)[0]):
        action_sequence = action_sequences[i][:]
        action_sequence = [int(x) for x in action_sequence]

        t_steps = len(action_sequence)+1
        times = np.arange(0,t_steps,1)

        # definicion del estado inicial e inicializacion de estados forzado y natural

        initial_state = np.zeros(nh, dtype=np.complex_)
        initial_state[0] = 1.0        

        # inicializacion de estado forzado 
        forced_state = initial_state

        # almacenar evolucion natural y evolucion forzada
        forced_evol = [state_fidelity(forced_state)]

        for action in action_sequence:
            
            forced_state = calculate_next_state(forced_state,action,propagators)
            forced_evol.append(state_fidelity(forced_state))

            max_forced = np.max(forced_evol)
        
        if max_forced > max_fid:
            max_fid = max_forced
            max_index = i
    print('Max fid:',max_fid, 'Max Index:',max_index)
    return max_fid, max_index

