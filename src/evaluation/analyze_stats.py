import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def _plot_line(xs, ys, label, sem=None, N: int = 1):
    plt.plot(xs, np.convolve(ys, np.ones(N) / N, mode='same'), label=label)
    if sem is not None:
        plt.fill_between(xs, ys - sem, ys + sem, alpha=0.5)


def plot_stat(d: list or dict, stat_name: str or list, N: int = 1, symmetric: bool = False, save: bool = False):
    """
    Plot experimental results from a single stat.
    :param d: Either a list of dictionaries from multiple experiments or a single one.
    :param stat_name: The key in the dictionary to plot.
    :param N: The window size for smoothing.
    :param symmetric: Whether to combine symmetric agents (only applicable for multiple expts)
    :param save: Whether to save the plot.
    """
    if isinstance(d[0], dict):
        for player_id in d[0].keys():
            ys = np.array([ep_data[player_id].get(stat_name, 0) for ep_data in d])
            xs = range(ys.size)
            _plot_line(xs, ys, player_id, N=N)
    elif isinstance(d[0], list):
        if not symmetric:
            for player_id in d[0][0].keys():
                y_mean, y_sem = average_stat(d, stat_name, player_id=player_id)
                xs = range(y_mean.size)
                _plot_line(xs, y_mean, player_id, sem=y_sem, N=N)
        else:
            y_mean, y_sem = average_stat(d, stat_name, symmetric=True)
            xs = range(y_mean.size)
            _plot_line(xs, y_mean, 'Symmetric', sem=y_sem, N=N)
    else:
        raise ValueError

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel(stat_name.replace('_', ' ').title())
    # plt.title(stat_name)
    if save:
        plt.savefig(f"figures/{stat_name}.pdf")
    plt.show()


def plot_double_stat(d: list[dict], stat_names: tuple, save: bool = False):
    """
    Plot experimental results from two stats with different y-axes.
    :param d: Either a list of dictionaries from multiple experiments or a single one.
    :param stat_names: The two keys in the dictionary to plot.
    :param N: The window size for smoothing.
    :param save: Whether to save the plot.
    """
    stat_0 = []
    stat_1 = []
    for player_id in d[0].keys():
        # if stat_names[0] in d[0][player_id].keys() and stat_names[1] in d[0][player_id].keys():
        stat_0.append(np.array([ep_data[player_id].get(stat_names[0], 0) for ep_data in d]))
        stat_1.append(np.array([ep_data[player_id].get(stat_names[1], 0) for ep_data in d]))
    xs = range(stat_0[0].size)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reciprocal Reward')
    ax1.set_ylim(-0.075, 0.15)
    labels = ['Reciprocator: P(Cooperate)', 'NL-PPO: P(Cooperate)']
    line_rr, = ax1.plot(xs, stat_0[0], color='silver', label='Reciprocal Reward')
    ax2 = ax1.twinx()
    ax2.set_ylabel('P(Cooperate)')
    line_pcr, = ax2.plot(xs, stat_1[0], label=labels[0])
    line_pcnl, = ax2.plot(xs, stat_1[1], label=labels[1])
    fig.tight_layout()
    plt.legend(handles=[line_rr, line_pcr, line_pcnl], bbox_to_anchor=[0.9, 0.2])
    if save:
        plt.savefig('figures/ipd_pcoop_rr.pdf')
    plt.show()


def plot_divided_stat(d: list or dict, stat_names: tuple, N: int = 1, symmetric: bool = False, save: bool = False):
    """
    Plot experimental results from a single stat.
    :param d: Either a list of dictionaries from multiple experiments or a single one.
    :param stat_names: Two keys in the dictionary to plot, first is numerator and second is denominator
    :param N: The window size for smoothing.
    :param symmetric: Whether to combine symmetric agents (only applicable for multiple expts)
    :param save: Whether to save the plot.
    """
    if isinstance(d[0], dict):
        for player_id in d[0].keys():
            ys_numer = np.array([ep_data[player_id].get(stat_names[0], 0) for ep_data in d])
            ys_denom = np.array([ep_data[player_id].get(stat_names[1], 0) for ep_data in d])
            ys = ys_numer / ys_denom
            xs = range(ys.size)
            _plot_line(xs, ys, player_id, N=N)
    elif isinstance(d[0], list):
        if not symmetric:
            for player_id in d[0][0].keys():
                ys = get_stat_list(d, player_id, stat_names[0]) / get_stat_list(d, player_id, stat_names[1])
                y_mean = ys.mean(axis=0)
                y_sem = ys.std(axis=0) / ys.shape[0]
                xs = range(y_mean.size)
                _plot_line(xs, y_mean, player_id, sem=y_sem, N=N)
        else:
            ys = []
            for player_id in d[0][0].keys():
                ys.append(get_stat_list(d, player_id, stat_names[0]) / get_stat_list(d, player_id, stat_names[1]))
            ys = np.concatenate(ys, axis=0)
            y_mean = ys.mean(axis=0)
            y_sem = ys.std(axis=0) / ys.shape[0]
            xs = range(y_mean.size)
            _plot_line(xs, y_mean, 'Symmetric', sem=y_sem, N=N)
    else:
        raise ValueError

    plt.legend()
    plt.xlabel('Episode')
    if stat_names[0] == 'own_coin_count':
        plt.ylabel(f"P(own coin)")
    else:
        plt.ylabel(f"P({stat_names[0]})")
    plt.title(f"{stat_names[0]}/{stat_names[1]}")
    if save:
        plt.savefig(f"figures/{stat_names[0]}_div_{stat_names[1]}.pdf")
    plt.show()


def get_stat_list(ds: list[dict], player_id: str, stat_name: str):
    ys = []
    for i, d in enumerate(ds):
        ys.append(np.array([ep_data[player_id].get(stat_name, 0) for ep_data in d]))
    return np.stack(ys, axis=0)


def average_stat(ds, stat_name, player_id: int = None, symmetric: bool = False):
    ys = []
    if not symmetric:
        for i, d in enumerate(ds):
            ys.append(np.array([ep_data[player_id].get(stat_name, 0) for ep_data in d]))
    else:
        for player_id in ds[0][0].keys():
            for i, d_ in enumerate(ds):
                ys.append(np.array([ep_data[player_id].get(stat_name, 0) for ep_data in d_]))
    ys = np.stack(ys, axis=0)
    return ys.mean(axis=0), ys.std(axis=0) / ys.shape[0]


def average_divided_stat(ds: list, stat_names: tuple, player_id: int = None, symmetric: bool = False):
    if not symmetric:
        ys = get_stat_list(ds, player_id, stat_names[0]) / get_stat_list(ds, player_id, stat_names[1])
        y_mean = ys.mean(axis=0)
        y_sem = ys.std(axis=0) / ys.shape[0]
    else:
        ys = []
        for player_id in ds[0][0].keys():
            ys.append(get_stat_list(ds, player_id, stat_names[0]) / get_stat_list(ds, player_id, stat_names[1]))
        ys = np.concatenate(ys, axis=0)
        y_mean = ys.mean(axis=0)
        y_sem = ys.std(axis=0) / ys.shape[0]
    return y_mean, y_sem


def get_stat_names(d: list):
    stat_names = []
    for player_name in d[-1].keys():
        stat_names += d[-1][player_name].keys()
    stat_names = sorted(set(stat_names))
    return stat_names


def mean_episodes(d: list):
    """For meta-episodes."""
    mean_dict = defaultdict(lambda: defaultdict(float))
    for player_name in d[0].keys():
        for stat_name in d[0][player_name].keys():
            mean_dict[player_name][stat_name] += np.mean([ep_data[player_name][stat_name] for ep_data in d])
    return mean_dict


def get_last_episodes(d: list):
    """For meta-episodes."""
    # return [d_[-1] for d_ in d]
    return [mean_episodes(d_) for d_ in d]


def get_last_log_path(dir_path: str):
    if not os.path.isdir(dir_path):
        print(dir_path, "not a directory")
        return
    results = [f[4:-5] for f in os.listdir(dir_path) if f.startswith('out_')]
    if len(results) == 0:
        print("No results found for", dir_path)
        return
    last_result_num = str(sorted([int(r) for r in results])[-1])
    return os.path.join(dir_path, f'out_{last_result_num}.json')


def check_if_mfos(results_path: str):
    return any([os.path.isdir(os.path.join(results_path, replicate_dir, 'mfos_0')) for replicate_dir in
                os.listdir(results_path)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-sym', '--symmetric', action='store_true')
    parser.add_argument('-t', '--truncate', type=int, default=None)
    parser.add_argument('-m', '--multiple', action='store_true')
    parser.add_argument("-s", "--save", action='store_true')

    args = parser.parse_args()
    symmetric = args.symmetric
    results_path = args.path
    truncate_len = args.truncate
    multiple = args.multiple
    save = args.save

    # Plot single experiment
    if os.path.isfile(results_path):
        with open(results_path) as f:
            d = json.load(f)
            if isinstance(d[0], list):
                d = get_last_episodes(d)
        stat_names = get_stat_names(d)
        for stat_name in stat_names:
            plot_stat(d, stat_name, save=save)
        if 'own_coin_count' in stat_names and 'total_coin_count' in stat_names:
            plot_divided_stat(d, ('own_coin_count', 'total_coin_count'), save=save)
        if '0_mean_reciprocal_reward_across_batches' in stat_names and 'p(cooperate)' in stat_names:
            plot_double_stat(d, ('0_mean_reciprocal_reward_across_batches', 'p(cooperate)'), save=save)

    elif os.path.isdir(results_path):
        if not multiple:
            # Plot averages over multiple replications
            ds = []
            if check_if_mfos(results_path):
                replicate_dirs = [os.path.join(replicate_dir, 'mfos_0') for replicate_dir in os.listdir(results_path)
                                  if os.path.isdir(os.path.join(results_path, replicate_dir, 'mfos_0'))]
            else:
                replicate_dirs = [replicate_dir for replicate_dir in os.listdir(results_path)
                                  if os.path.isdir(os.path.join(results_path, replicate_dir))]
            print("Replicate dirs", replicate_dirs)
            for replicate_d in sorted(replicate_dirs):
                last_log_path = get_last_log_path(os.path.join(results_path, replicate_d))
                if last_log_path is None:
                    continue
                print("Logs", last_log_path)
                with open(last_log_path) as f:
                    d = json.load(f)
                    if isinstance(d[0], list):
                        # Deal with meta-episodes' nested lists
                        d = get_last_episodes(d)
                    ds.append(d)

            min_len = min([len(d_) for d_ in ds])
            if truncate_len is not None and truncate_len > min_len:
                ds_temp = ds
                ds = []
                for d_ in ds_temp:
                    if len(d_) >= truncate_len:
                        ds.append(d_[:truncate_len])
                    else:
                        print(f"Skipping {len(d_)} < {truncate_len}")
            elif truncate_len is not None:
                print("Truncating to length:", truncate_len)
                ds = [d_[:truncate_len] for d_ in ds]
            else:
                print("Truncating to min length:", min_len)
                ds = [d_[:min_len] for d_ in ds]

            stat_names = get_stat_names(ds[0])
            for stat_name in stat_names:
                plot_stat(ds, stat_name, symmetric=symmetric, save=save)

            if 'own_coin_count' in stat_names and 'total_coin_count' in stat_names:
                plot_divided_stat(ds, ('own_coin_count', 'total_coin_count'), symmetric=symmetric, save=save)
        else:
            # Plot averages over multiple replications for multiple experiments on same plot
            # Get all top level experiment directories
            experiment_dirs = [os.path.join(results_path, experiment_dir) for experiment_dir in os.listdir(results_path)
                               if os.path.isdir(os.path.join(results_path, experiment_dir))]
            print("Experiment dirs", experiment_dirs)

            # For each experiment
            ys = defaultdict(list)  # List of tuples (y_mean, y_sem, label)
            for experiment_dir in experiment_dirs:
                mfos = "MFOS" in experiment_dir
                print(experiment_dir, "is", mfos)
                if check_if_mfos(experiment_dir):
                    replicates = [os.path.join(replicate_dir, 'mfos_0') for replicate_dir in os.listdir(experiment_dir)
                                  if os.path.isdir(os.path.join(experiment_dir, replicate_dir, 'mfos_0'))]
                else:
                    replicates = os.listdir(experiment_dir)
                ds = []
                for replicate_dir in replicates:
                    print("Experiment dir, replicate_dir", experiment_dir, replicate_dir)
                    last_log_path = get_last_log_path(os.path.join(experiment_dir, replicate_dir))
                    if last_log_path is None:
                        continue
                    with open(last_log_path) as f:
                        d = json.load(f)
                        if isinstance(d[0], list):
                            # Deal with meta-episodes' nested lists
                            d = get_last_episodes(d)
                        ds.append(d)

                min_len = min([len(d_) for d_ in ds])
                if truncate_len is not None and truncate_len > min_len:
                    raise ValueError("Truncate length longer than min length")
                elif truncate_len is not None:
                    print("Truncating to length:", truncate_len)
                    ds = [d_[:truncate_len] for d_ in ds]
                else:
                    print("Truncating to min length:", min_len)
                    ds = [d_[:min_len] for d_ in ds]

                stat_names = get_stat_names(ds[0])
                print(stat_names)
                for stat_name in stat_names:
                    if symmetric:
                        y_mean, y_sem = average_stat(ds, stat_name, symmetric=symmetric)
                        ys[stat_name].append((y_mean, y_sem, os.path.basename(os.path.normpath(experiment_dir))))
                    else:
                        for player_id in ds[0][0].keys():
                            y_mean, y_sem = average_stat(ds, stat_name, player_id=player_id)
                            if mfos:
                                print("SWITCHING MFOS")
                                # Swap player order since MFOS code puts MFOS as player 1 and NL as player 0
                                ys[f"{stat_name}_{player_id[:-1]}{1 - int(player_id[-1])}"].append((y_mean, y_sem,
                                                                                                    os.path.basename(
                                                                                                        os.path.normpath(
                                                                                                            experiment_dir))))
                            else:
                                ys[f"{stat_name}_{player_id}"].append((y_mean, y_sem,
                                                                       os.path.basename(
                                                                           os.path.normpath(experiment_dir))))
                if 'own_coin_count' in stat_names and 'total_coin_count' in stat_names:
                    stat_name = "P(Own Coin)"
                    if symmetric:
                        print(len(ds))
                        y_mean, y_sem = average_divided_stat(ds, ('own_coin_count', 'total_coin_count'),
                                                             symmetric=symmetric)
                        ys[stat_name].append((y_mean, y_sem, os.path.basename(os.path.normpath(experiment_dir))))
                    else:
                        for player_id in ds[0][0].keys():
                            y_mean, y_sem = average_divided_stat(ds, ('own_coin_count', 'total_coin_count'),
                                                                 player_id=player_id)
                            if mfos:
                                print("SWITCHING MFOS")
                                ys[f"{stat_name}_{player_id[:-1]}{1 - int(player_id[-1])}"].append((y_mean, y_sem,
                                                                                                    os.path.basename(
                                                                                                        os.path.normpath(
                                                                                                            experiment_dir))))
                            else:
                                ys[f"{stat_name}_{player_id}"].append((y_mean, y_sem,
                                                                       os.path.basename(os.path.normpath(experiment_dir))))

            # Plot all
            colors = {'MFOS': 'tab:green', 'Reciprocator': 'tab:blue', 'NL-PPO': 'tab:orange'}
            for stat_name, data in ys.items():
                print(stat_name)
                lines = []
                for y_mean, y_sem, label in data:
                    xs = range(y_mean.size)
                    line, = plt.plot(xs, y_mean, label=label, color=colors[label])
                    lines.append(line)
                    plt.fill_between(xs, y_mean - y_sem, y_mean + y_sem, alpha=0.5, color=colors[label])
                plt.xlabel('Episode')
                if stat_name == "P(Own Coin)_player_1":
                    ylabel = "NL-PPO P(Own Coin)"
                elif stat_name == "total_coin_count_player_1":
                    ylabel = "NL-PPO Total Coins"
                else:
                    ylabel = stat_name.replace('_', ' ').title()
                plt.ylabel(ylabel)
                if len(lines) == 3:
                    plt.legend(handles=[lines[1], lines[0], lines[2]], bbox_to_anchor=[0.675, 0.575])
                    # bbox_to_anchor=[0.975, 0.225]
                else:
                    plt.legend()
                # for stat_name, data in ys.items():
                #     fig, ax1 = plt.subplots()
                #     ax1.set_xlabel('Episode')
                #     ax1.set_ylabel(stat_name.replace('_', ' ').title())
                #     ax1.set_xlim(0, 100)
                #     ax2 = ax1.twiny()
                #     ax2.set_xlabel('Meta-Episode')
                #     ax2.set_xlim(0, 1000)
                #
                #     handles = []
                #     for y_mean, y_sem, label in data:
                #         if label == "MFOS":
                #             xs = range(y_mean.size)
                #             line, = ax2.plot(xs, y_mean, label=label, color='tab:green')
                #             ax2.fill_between(xs, y_mean - y_sem, y_mean + y_sem, alpha=0.5, color='tab:green')
                #             handles.append(line)
                #         else:
                #             xs = range(y_mean.size)
                #             line, = ax1.plot(xs, y_mean, label=label)
                #             handles.append(line)
                #             ax1.fill_between(xs, y_mean - y_sem, y_mean + y_sem, alpha=0.5)
                #
                #     fig.tight_layout()
                #     if len(handles) == 3:
                #         handles = [handles[1], handles[0], handles[2]]
                #     plt.legend(handles=handles, bbox_to_anchor=[0.32, 1.])
                # plt.xlabel('Episode')
                # plt.ylabel(stat_name.replace('_', ' ').title())
                # plt.title(stat_name)
                if save:
                    plt.savefig(f"figures/{stat_name}.pdf")
                plt.show()
