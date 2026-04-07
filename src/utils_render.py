import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
import matplotlib.animation as manimation
import random

from utils import get_map


def render_maze(agent, state=None, locs=None, colors=None, ax=None, save_path=None, wall=None, fontsize=10):
    """
    Renders the maze

    Args:
        agent (LinearRL class) : The agent
        state (tuple/array, Optional) : The state to draw the agent in, if None will use starting location
        locs (List of states, Optional) : Color specific locations (states)
        colors (List of color idxs, Optional) : The specific idx of colors to use from the colorblind color palette
        wall (list, Optional) : List containing two sublists for wall coordinates [[row1, row2], [col1, col2]]
    """
    if ax is None:
        fig, ax = plt.subplots()
    m = get_map(agent)

    if state is None:
        state = agent.start_loc

    color_palette = sns.color_palette("colorblind")
    
    ax.imshow(m, origin='upper', cmap='gray_r')

    ax.set_xticks(np.arange(-.5, len(m), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m), 1), minor=True)

    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)

    agent_loc = patches.Circle((state[1],state[0]), radius=0.4, fill=True, color='blue', alpha=0.7)
    ax.add_patch(agent_loc)

    for i, target_loc in enumerate(agent.target_locs):
        reward = patches.Rectangle((target_loc[1] - 0.5, target_loc[0] - 0.5), 1.0, 1.0, fill=True, color='green', alpha=0.7)
        ax.text(target_loc[1], target_loc[0], f'$\mathrm{{r}}_{i+1}$', color='white', fontsize=fontsize, ha='center', va='center', style='normal')
        ax.add_patch(reward)

    if locs is not None:
        for loc, color in zip(locs, colors):
            rect = patches.Rectangle((loc[1] - 0.5, loc[0] - 0.5), 1.0, 1.0, fill=True, color=color_palette[color])
            ax.text(loc[1], loc[0], f'$\mathrm{{s}}_{locs.index(loc) + 1}$', color='white', fontsize=fontsize, ha='center', va='center')
            ax.add_patch(rect)

    if wall is not None:
        print(f"Attempting to draw wall: {wall}")  # Debugging print
        [row, col], [direction] = wall
        if direction == 'h':  # Horizontal wall
            ax.plot([col - 0.5, col + 0.5], [row - 0.5, row - 0.5], color='red', linewidth=4, zorder=10)
        elif direction == 'v':  # Vertical wall
            ax.plot([col - 0.5, col - 0.5], [row - 0.5, row + 0.5], color='red', linewidth=4, zorder=10)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(which='both', size=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def render_maze_replan(agent, state=None, initial_goal=None, replan_goals=None, ax=None, save_path=None, fontsize=12):
    """
    Renders the maze

    Args:
        agent (LinearRL class) : The agent
        state (tuple/array, Optional) : The state to draw the agent in, if None will use starting location
        initial_gaol (tuple, Optional) : 
        replan_goals (list of tuples) : 
        colors (list of color idxs, Optional) : The specific idx of colors to use from the colorblind color palette
        wall (list, Optional) : List containing two sublists for wall coordinates [[row1, row2], [col1, col2]]
    """
    if ax is None:
        fig, ax = plt.subplots()
    m = get_map(agent)

    if state is None:
        state = agent.start_loc
    
    if initial_goal is None:
        initial_goal = agent.target_locs[0]

    ax.imshow(m, origin='upper', cmap='gray_r')

    ax.set_xticks(np.arange(-.5, len(m), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m), 1), minor=True)

    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)

    agent_loc = patches.Circle((state[1],state[0]), radius=0.4, fill=True, color='blue', alpha=0.7)
    ax.add_patch(agent_loc)

    reward = patches.Rectangle((initial_goal[1] - 0.5, initial_goal[0] - 0.5), 1.0, 1.0, fill=True, color='green', alpha=0.7)
    ax.text(initial_goal[1], initial_goal[0], f'$\mathrm{{r}}_{1}$', color='white', fontsize=fontsize, ha='center', va='center')
    ax.add_patch(reward)

    for i, goal_loc in enumerate(replan_goals, 2):
        reward = patches.Rectangle((goal_loc[1] - 0.5, goal_loc[0] - 0.5), 1.0, 1.0, fill=True, color='gray', alpha=0.7)
        ax.text(goal_loc[1], goal_loc[0], f'$\mathrm{{r}}_{i}$', color='white', fontsize=fontsize, ha='center', va='center')
        ax.add_patch(reward)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(which='both', size=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def render_maze_dc(maze, save_path=None, title=None):
    """
    Renders the mazes for the De Cothi tasks

    Args:
        maze (array) : The structure of the maze (open sapce = 0, wall = -1, starting loc = 1-10)
        save_path (string, Optional) : Where to save the image
        title (string, Optional) : The title of the plot
    """
    fig, ax = plt.subplots()
    
    maze_gray = np.zeros_like(maze, dtype=float)
    maze_gray[maze == -1] = 1 
    maze_gray[maze == 0] = 0 
    maze_gray[maze > 0] = 0
    
    ax.imshow(maze_gray, origin='upper', cmap='gray_r')

    ax.set_xticks(np.arange(-.5, len(maze), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)

    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(which='both', size=0)

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if 1 <= maze[i, j] <= 10:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor='blue', alpha=0.7)
                ax.add_patch(rect)
                ax.text(j, i, str(int(maze[i, j])), ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    rect = patches.Rectangle((3-0.5, 3-0.5), 1, 1, fill=True, facecolor='green', alpha=0.7)
    ax.add_patch(rect)

    if title is not None:
        plt.title(title, fontsize=18)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_decision_prob(probs_train, probs_test, colors, leg_loc=None, save_path=None, title=None, ylabel=None, std=None, remove_spine=False, ymax=None, start_i=0):
    plt.rcParams['font.family'] = 'serif'
    color_palette = sns.color_palette("colorblind")
    color_list = []
    for color in colors:
        color_list.append(color_palette[color])

    bar_positions_training = np.arange(len(probs_train)) * 0.4
    bar_positions_test = np.arange(len(probs_train)) * 0.4 + 1.5

    # Minimum height if small prob
    probs_train = np.array(probs_train)
    probs_test = np.array(probs_test)
    min_visible_height = 0.02
    probs_train[probs_train < 0.05] = min_visible_height
    probs_test[probs_test < 0.05] = min_visible_height

    plt.bar(bar_positions_training, probs_train, width=0.3, color=color_list, edgecolor='black')
    plt.bar(bar_positions_test, probs_test, width=0.3, color=color_list, edgecolor='black')

    # Add error bars if std is provided
    if std is not None:
        plt.errorbar(bar_positions_training, probs_train, yerr=std[0], fmt='none', ecolor='black', capsize=0)
        plt.errorbar(bar_positions_test, probs_test, yerr=std[1], fmt='none', ecolor='black', capsize=0)

    handles = [plt.Rectangle((0,0),1,1, facecolor=color_list[i], edgecolor='black') for i in range(len(probs_train))]

    if leg_loc is not None:
        plt.legend(handles, [f'$\mathrm{{s}}_{i+1+start_i}$' for i in range(len(probs_train))], title='States', loc=leg_loc, fontsize=14)
    else:
        plt.legend(handles, [f'$\mathrm{{s}}_{i+1+start_i}$' for i in range(len(probs_train))], title='States', loc='upper right', fontsize=14)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    plt.xticks([0.2, 1.7], ['Training', 'Test'], fontsize=18)

    # Set custom y-axis ticks
    if ymax is None:
        max_prob = max(max(probs_train), max(probs_test))
        y_ticks = np.arange(0, max_prob + 0.1, 0.1)
        plt.yticks(y_ticks)
    else:
        y_ticks = np.arange(0, ymax + 0.1, 0.2)
        plt.yticks(y_ticks)

    if title is not None:
        plt.title(title, fontsize=20)
    
    if remove_spine:
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Save the image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def plot_decision_prob_two_step(probs_train, probs_test, colors, leg_loc=None, save_path=None, title=None, ylabel=None, std=None, remove_spine=False, ymax=None, start_i=0):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
    color_palette = sns.color_palette("colorblind")
    color_list = []
    for color in colors:
        color_list.append(color_palette[color])

    bar_positions_training = np.arange(len(probs_train)) * 0.4
    bar_positions_test = np.arange(len(probs_train)) * 0.4 + 1.5

    # Minimum height if small prob
    probs_train = np.array(probs_train)
    probs_test = np.array(probs_test)
    min_visible_height = 0.02
    probs_train[probs_train < 0.05] = min_visible_height
    probs_test[probs_test < 0.05] = min_visible_height

    plt.bar(bar_positions_training, probs_train, width=0.3, color=color_list, edgecolor='black')
    plt.bar(bar_positions_test, probs_test, width=0.3, color=color_list, edgecolor='black')

    # Add error bars if std is provided
    if std is not None:
        plt.errorbar(bar_positions_training, probs_train, yerr=std[0], fmt='none', ecolor='black', capsize=0)
        plt.errorbar(bar_positions_test, probs_test, yerr=std[1], fmt='none', ecolor='black', capsize=0)

    handles = [plt.Rectangle((0,0),1,1, facecolor=color_list[i], edgecolor='black') for i in range(len(probs_train))]

    if leg_loc is not None:
        plt.legend(handles, [f'$\mathrm{{S}}_{i+1+start_i}$' for i in range(len(probs_train))], title='States', loc=leg_loc, fontsize=14, frameon=False)
    else:
        plt.legend(handles, [f'$\mathrm{{S}}_{i+1+start_i}$' for i in range(len(probs_train))], title='States', loc='upper right', fontsize=14, frameon=False)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    plt.xticks([0.2, 1.7], ['Before\nrevaluation', 'After\nrevaluation'], fontsize=18)

    # Set custom y-axis ticks
    if ymax is None:
        max_prob = max(max(probs_train), max(probs_test))
        y_ticks = np.arange(0, max_prob + 0.1, 0.1)
        plt.yticks(y_ticks)
    else:
        y_ticks = np.arange(0, ymax + 0.1, 0.2)
        plt.yticks(y_ticks)

    if title is not None:
        plt.title(title, fontsize=20)
    
    if remove_spine:
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Save the image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def plot_decision_prob_detour(probs_train, probs_test, colors, leg_loc=None, save_path=None, title=None):
    plt.rcParams['font.family'] = 'serif'
    color_palette = sns.color_palette("colorblind")
    color_list = []
    for color in colors:
        color_list.append(color_palette[color])
    
    probs_train_array = np.array(probs_train)
    probs_test_array = np.array(probs_test)
    
    mean_train = np.mean(probs_train_array, axis=0)
    mean_test = np.mean(probs_test_array, axis=0)
    
    mean_train = np.atleast_1d(mean_train)
    mean_test = np.atleast_1d(mean_test)
    
    std_train = np.std(probs_train_array, axis=0, ddof=1)
    std_test = np.std(probs_test_array, axis=0, ddof=1)
    
    std_train = np.atleast_1d(std_train)
    std_test = np.atleast_1d(std_test)
    
    stderr_train = std_train / np.sqrt(len(probs_train))
    stderr_test = std_test / np.sqrt(len(probs_test))
    
    min_visible_height = 0.02
    mean_train_plot = mean_train.copy()
    mean_test_plot = mean_test.copy()
    mean_train_plot[mean_train_plot < 0.1] = min_visible_height
    mean_test_plot[mean_test_plot < 0.1] = min_visible_height

    bar_positions_training = np.arange(len(mean_train)) * 0.4
    bar_positions_test = np.arange(len(mean_train)) * 0.4 + 1.5

    plt.bar(bar_positions_training, mean_train_plot, width=0.3, color=color_list, edgecolor='black')
    plt.bar(bar_positions_test, mean_test_plot, width=0.3, color=color_list, edgecolor='black')

    plt.errorbar(bar_positions_training, mean_train_plot, yerr=stderr_train, 
                fmt='none', ecolor='black', capsize=0, capthick=1)
    plt.errorbar(bar_positions_test, mean_test_plot, yerr=stderr_test, 
                fmt='none', ecolor='black', capsize=0, capthick=1)

    handles = [plt.Rectangle((0,0),1,1, facecolor=color_list[i], edgecolor='black') for i in range(len(mean_train))]

    if leg_loc is not None:
        plt.legend(handles, [f'$\mathrm{{s}}_{i+1}$' for i in range(len(mean_train))], title='States', loc=leg_loc, fontsize=14)
    else:
        plt.legend(handles, [f'$\mathrm{{s}}_{i+1}$' for i in range(len(mean_train))], title='States', loc='upper right', fontsize=14)
    
    plt.ylabel('Probabilities', fontsize=18)
    plt.xticks([0.4, 1.9], ['Training', 'Test'], fontsize=18)

    max_prob = max(max(mean_train_plot), max(mean_test_plot))
    y_ticks = np.arange(0, max_prob + 0.1, 0.1)
    plt.yticks(y_ticks)

    if title is not None:
        plt.title(title, fontsize=20)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def create_bar_plot(means, colors, ylabel, xlabels, y_lim=None, std=None, title=None, save_path=None):
    """
    Bar plot for NHB Plotting

    Args:
        means (array) : mean of each bar to plot
        colors (list) : colors to use
        ylabel (string) : label for the y-axis
        xlabels (list of strings) : labels for the x-axis
        std (array, optional) : std error of each bar
        title (string, optional) : title of the plot
        save_path (string, optional) : where to save the figure
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(means)) * 0.25
    
    plot_means = np.array(means, dtype=float).copy()
    min_visible_height = 0.02
    plot_means[plot_means < 0.1] = min_visible_height
    
    bars = ax.bar(x, plot_means, color=colors, edgecolor='black', linewidth=1, width=0.14)
    
    if std is not None:
        ax.errorbar(x, means, yerr=std, fmt='none', color='black', capsize=0)
    
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20) if title else None
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha='center', fontsize=18)
    
    if y_lim is None:
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylim(0, 1.1)
    else:
        ax.set_yticks(np.arange(y_lim[0], y_lim[1], 0.2))
        ax.set_ylim(y_lim[0], y_lim[1])
    
    for spine in ['left', 'right', 'bottom', 'top']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1)
    
    for spine in ['right', 'top']:
        # ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0)
    
    ax.grid(False)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_nhb_decisions(probs_reward, probs_policy, probs_transition, colors, leg_loc=None, save_path=None, title=None, std=None):
    """
    Plots the decision probability of going towards a terminal state for three different revaluation scenarios

    Args:
        probs_reward (array) : Probability of heading towards each terminal state after reward revaluation
        probs_policy (array) : Probability of heading towards each terminal state after policy revaluation
        probs_transition (array) : Probability of heading towards each terminal state after transition revaluation
        colors (array) : idx of color palette color to use 
        leg_loc (string, Optional) : Location to place the legend
        save_path (string, Optional) : File path to save the image to
        title (string, Optional) : Title of the figure
        std (list [std_reward, std_policy, std_trans], Optional) : Std deviations for reward, policy, and transition revaluations
    """
    color_palette = sns.color_palette("colorblind")
    color_list = [color_palette[color] for color in colors]

    num_states = len(probs_reward)
    bar_width = 0.3  
    state_spacing = 0.05
    group_spacing = 0.8
    
    indices = np.arange(3) * (num_states * (bar_width + state_spacing) + group_spacing)
    
    plt.figure(figsize=(14, 5))

    for i in range(num_states):
        plt.bar(indices[0] + i * (bar_width + state_spacing), probs_reward[i], bar_width, color=color_list[i], edgecolor='black')
        plt.bar(indices[1] + i * (bar_width + state_spacing), probs_policy[i], bar_width, color=color_list[i], edgecolor='black')
        plt.bar(indices[2] + i * (bar_width + state_spacing), probs_transition[i], bar_width, color=color_list[i], edgecolor='black')

    if std is not None:
        for i in range(num_states):
            plt.errorbar(indices[0] + i * (bar_width + state_spacing), probs_reward[i], yerr=std[0][i], fmt='none', ecolor='black', capsize=0)
            plt.errorbar(indices[1] + i * (bar_width + state_spacing), probs_policy[i], yerr=std[1][i], fmt='none', ecolor='black', capsize=0)
            plt.errorbar(indices[2] + i * (bar_width + state_spacing), probs_transition[i], yerr=std[2][i], fmt='none', ecolor='black', capsize=0)

    handles = [plt.Rectangle((0,0),1,1, facecolor=color_list[i], edgecolor='black') for i in range(num_states)]
    if leg_loc is not None:
        plt.legend(handles, [f'State {i+2}' for i in range(num_states)], title='States', loc=leg_loc, fontsize=12, title_fontsize=14, frameon=False)
    else:
        plt.legend(handles, [f'State {i+2}' for i in range(num_states)], title='States', fontsize=12, title_fontsize=14, frameon=False)

    plt.ylabel('Probabilities', fontsize=18)
    plt.title(title if title else 'Decision Probabilities Across Revaluations', fontsize=22, pad=20)
    
    plt.xticks(indices + (num_states - 1) * (bar_width + state_spacing) / 2, ['Reward Revaluation', 'Policy Revaluation', 'Transition Revaluation'], fontsize=18)

    max_prob = max(max(probs_reward), max(probs_policy), max(probs_transition))
    y_ticks = np.arange(0, min(max_prob + 0.1, 1.05), 0.1)
    plt.yticks(y_ticks, fontsize=12)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue']

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Increase bottom margin
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def render_DR(agent, state, ax=None):
    state_idx = agent.mapping[(state[0], state[1])]
    ax.imshow(agent.DR[state_idx].reshape(agent.height, agent.width), 
              origin='upper', cmap='plasma')
    ax.set_title("DR(%d, %d)" % (state[0], state[1]))
    ax.set_axis_off()

def render_V(values, agent, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    min_value = np.min(values[~np.isinf(values)])
    max_value = np.max(values)

    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.0)

    ax.imshow(values.reshape(agent.height, agent.width),
                origin='upper',
                cmap=cmap, vmin=min_value, vmax=max_value)
    ax.set_title("$Values$")
    ax.set_axis_off()

def render_V_log(values, agent, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    min_value = np.min(values[~np.isinf(values)])
    max_value = np.max(values)

    # Logarithmic transformation
    values_log = np.log(values - min_value + 1)  # Add 1 to avoid log(0)

    # Normalize the transformed values
    min_value_log = np.min(values_log)
    max_value_log = np.max(values_log)
    values_scaled = (values_log - min_value_log) / (max_value_log - min_value_log)

    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.0)

    ax.imshow(values_scaled.reshape(agent.height, agent.width),
                origin='upper',
                cmap=cmap, vmin=0, vmax=1)  # Scale between 0 and 1
    ax.set_title("$Values$")
    ax.set_axis_off()

def make_plots(agent, values, state=None):
    # Adjust DR at terminal state
    for target_loc in agent.target_locs:
        idx = agent.mapping[target_loc[0], target_loc[1]]
    agent.DR[idx, :] = 0
    agent.DR[idx, idx] = 1

    if state is None:
        state = agent.start_loc
        # state = (0,0)
        
    fig, axs = plt.subplots(1, 3, dpi=144)
    render_maze(agent, state, ax=axs[0])
    render_DR(agent, state, ax=axs[1])
    render_V(agent, ax=axs[2])
    
    plt.show()

def record_trials(agent, title="recorded_trials", n_trial_per_loc=1,
                    start_locs=None, max_steps=100):
    metadata = dict(title=title, artist='JG')
    writer = manimation.FFMpegFileWriter(fps=10, metadata=metadata)
    fig, axs = plt.subplots(1, 3, figsize=(7, 3))
    fig.tight_layout()

    with writer.saving(fig, "./out/%s.mp4" % title, 144):
        for sl in start_locs:
            for trial in range(n_trial_per_loc):
                agent.env.reset()
                done = False
                steps = 0
                state = sl
                
                # set the start and agent location
                agent.env.unwrapped.start_loc, agent.env.unwrapped.agent_loc = state, state
                # Render starting state
                render_maze(agent, state, ax=axs[0])
                render_DR(agent, state, ax=axs[1])
                render_V(agent, ax=axs[2])
                writer.grab_frame()
                for ax in axs:
                        ax.clear()

                # Act greedily and record each state as well
                while not done and steps < max_steps:
                    action = agent.select_action(state)
                    obs, _, done, _, _ = agent.env.step(action)

                    render_maze(agent, state, ax=axs[0])
                    render_DR(agent, state, ax=axs[1])
                    render_V(agent, ax=axs[2])
                    writer.grab_frame()

                    steps += 1

                    state = obs["agent"]

                    for ax in axs:
                        ax.clear()

def record_trajectory(agent, traj, save_path=None):
    fig, ax = plt.subplots()

    m = get_map(agent)
    
    # Display maze
    ax.imshow(m, origin='upper', cmap='gray_r')

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(m), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m), 1), minor=True)

    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)
    # Display agent
    agent_loc = patches.Circle((agent.start_loc[1],agent.start_loc[0]), radius=0.4, fill=True, color='blue', alpha=0.7)
    ax.add_patch(agent_loc)

    # Display Reward
    for i, target_loc in enumerate(agent.target_locs):
        # reward = patches.Circle((target_loc[1], target_loc[0]), radius=0.4, fill=True, color='green')
        reward = patches.Rectangle((target_loc[1] - 0.5, target_loc[0] - 0.5), 1.0, 1.0, fill=True, color='green', alpha=0.7)
        ax.text(target_loc[1], target_loc[0], f'r{i+1}', color='white', fontsize=10, ha='center', va='center')
        ax.add_patch(reward)
    
    # loop through trajectory and add arrows
    for i in range(1, len(traj)):  # Start from the second position
        # Calculate the direction of movement
        diff = traj[i] - traj[i-1]
        # Start the arrow slightly behind the current position
        start_point = traj[i-1] - 0.2 * diff
        arrow = patches.FancyArrow(x=start_point[1], y=start_point[0], dx=0.5*diff[1], dy=0.5*diff[0], width=0.06, length_includes_head=True, color="red")
        ax.add_patch(arrow)

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Hide tick marks
    ax.tick_params(which='both', size=0)

    # Save the image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def render_maze_with_DR(agent, state=None, ax=None, save_path=None, fontsize=10):
    """
    Renders the maze with DR values overlaid using a purple gradient.
    Uses direct DR[state_idx][agent.mapping[(i,j)]] values.
    No labels, title, or colorbar.
    
    Args:
        agent (LinearRL class) : The agent
        state (tuple/array, Optional) : The state to draw the agent in, if None will use starting location
        ax (matplotlib.axes, Optional) : The axes to draw on
        save_path (string, Optional) : Where to save the image
        fontsize (int, Optional) : Font size for text elements
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    m = get_map(agent)
    
    if state is None:
        state = agent.start_loc
    
    ax.imshow(m, origin='upper', cmap='gray_r')
    
    ax.set_xticks(np.arange(-.5, len(m), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m), 1), minor=True)
    
    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)
    
    state_idx = agent.mapping[(state[0], state[1])]
    
    dr_values = []
    valid_positions = []
    
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] == 0 and (i, j) in agent.mapping:
                target_idx = agent.mapping[(i, j)]
                dr_value = agent.DR[state_idx][target_idx]
                dr_values.append(dr_value)
                valid_positions.append((i, j))

    dr_min = min(dr_values) if dr_values else 0
    dr_max = max(dr_values) if dr_values else 1
    
    deep_purple = mcolors.to_rgba('#6A0DAD')
    mid_purple = mcolors.to_rgba('#9370DB')
    light_purple = mcolors.to_rgba('#D8BFD8')
    white = mcolors.to_rgba('white')
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'enhanced_purple', 
        [white, mid_purple, deep_purple], 
        N=256
    )
    
    power = 0.8 
    
    for i, j in valid_positions:
        target_idx = agent.mapping[(i, j)]
        dr_value = agent.DR[state_idx][target_idx]
        
        if dr_max > dr_min:
            normalized_dr = (dr_value - dr_min) / (dr_max - dr_min)
        else:
            normalized_dr = 0.5
        
        transformed_dr = normalized_dr ** power
        
        color = cmap(transformed_dr)
        
        rect = patches.Rectangle(
            (j - 0.5, i - 0.5), 
            1.0, 
            1.0, 
            fill=True, 
            color=color,
            alpha=0.8
        )
        ax.add_patch(rect)
    
    agent_loc = patches.Circle((state[1], state[0]), radius=0.4, fill=True, color='blue', alpha=0.9)
    ax.add_patch(agent_loc)
    
    for i, target_loc in enumerate(agent.target_locs):
        reward = patches.Rectangle(
            (target_loc[1] - 0.5, target_loc[0] - 0.5), 
            1.0, 
            1.0, 
            fill=True, 
            color='green', 
            alpha=0.9
        )
        ax.text(
            target_loc[1], 
            target_loc[0], 
            f'$\mathrm{{r}}_{i+1}$', 
            color='white', 
            fontsize=fontsize, 
            ha='center', 
            va='center'
        )
        ax.add_patch(reward)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.tick_params(which='both', size=0)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return ax
    