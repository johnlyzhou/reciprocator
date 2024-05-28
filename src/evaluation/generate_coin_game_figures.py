import json
import os

import matplotlib.pyplot as plt
import numpy as np
import argparse

PLAYER_NAMES = ['Reciprocator', 'NL-PPO']


def process_results(d):
    """
    Process the results of the coin game analysis.
    """
    rews_0 = np.array([ep_data['rew_0'] for ep_data in d])
    rews_1 = np.array([ep_data['rew_1'] for ep_data in d])
    p0_own = np.array([ep_data['own_coin_count_0'] for ep_data in d])
    p0_other = np.array([ep_data['other_coin_count_0'] for ep_data in d])
    p1_own = np.array([ep_data['own_coin_count_1'] for ep_data in d])
    p1_other = np.array([ep_data['other_coin_count_1'] for ep_data in d])
    p0_total = p0_own + p0_other
    p1_total = p1_own + p1_other
    p0_prop = p0_own / (p0_own + p0_other)
    p1_prop = p1_own / (p1_own + p1_other)
    p0_petty = np.array([ep_data.get('petty_reward_0', 0.0) for ep_data in d])
    p1_petty = np.array([ep_data.get('petty_reward_1', 0.0) for ep_data in d])
    return rews_0, rews_1, p0_prop, p1_prop, p0_own, p0_other, p1_own, p1_other, p0_total, p1_total, p0_petty, p1_petty


def plot_coin_game_rewards(means, sems, name=None, save=False):
    """
    Plot the results of the coin game analysis.
    """
    for i in range(len(means)):
        plt.plot(means[i], label=PLAYER_NAMES[i])
        plt.fill_between(range(means[i].size), means[i] - sems[i], means[i] + sems[i], alpha=0.5)
    if len(means) > 1:
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # if name is not None:
    #     plt.title(name)
    if save:
        plt.savefig(f'figures/{name}_cg_rewards.pdf')
    plt.show()


def plot_coin_game_probs(means, sems, name=None, save=False):
    """
    Plot the results of the coin game analysis.
    """
    for i in range(len(means)):
        plt.plot(means[i], label=PLAYER_NAMES[i])
        plt.fill_between(range(means[i].size), means[i] - sems[i], means[i] + sems[i], alpha=0.5)
    if len(means) > 1:
        plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('P(own coin)')
    # if name is not None:
    #     plt.title(name)
    if save:
        plt.savefig(f'figures/{name}_cg_probs.pdf')
    plt.show()


def plot_coin_game_counts(own_means, other_means, total_means, own_sems, other_sems, total_sems, name=None, save=False):
    """
    Plot the results of the coin game analysis.
    """
    for i in range(len(own_means)):
        plt.plot(own_means[i], label=f'{PLAYER_NAMES[i]} Own Coins')
        plt.fill_between(range(own_means[i].size), own_means[i] - own_sems[i], own_means[i] + own_sems[i], alpha=0.5)
    for i in range(len(other_means)):
        plt.plot(other_means[i], label=f'{PLAYER_NAMES[i]} Other Coins')
        plt.fill_between(range(other_means[i].size), other_means[i] - other_sems[i], other_means[i] + other_sems[i],
                         alpha=0.5)
    for i in range(len(total_means)):
        plt.plot(total_means[i], label=f'{PLAYER_NAMES[i]} Total Coins')
        plt.fill_between(range(total_means[i].size), total_means[i] - total_sems[i], total_means[i] + total_sems[i],
                         alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Coin Count (32 Steps)')
    # if name is not None:
    #     plt.title(name)
    if save:
        plt.savefig(f'figures/{name}_cg_counts.pdf')
    plt.show()


def process_experiment(results_path):
    if os.path.isdir(results_path):

        expt_dirs = [expt_dir for expt_dir in os.listdir(results_path)
                     if os.path.isdir(os.path.join(results_path, expt_dir))]

        rews_0_list = []
        rews_1_list = []
        p0_prop_list = []
        p1_prop_list = []
        p0_own_list = []
        p0_other_list = []
        p1_own_list = []
        p1_other_list = []
        p0_total_list = []
        p1_total_list = []
        petty_0_list = []
        petty_1_list = []

        for expt_dir in sorted(expt_dirs):
            results = [int(f[4:-5]) for f in os.listdir(os.path.join(results_path, expt_dir)) if f.startswith('out_')]
            last_result_num = sorted(results)[-1]
            with open(os.path.join(results_path, expt_dir, f'out_{last_result_num}.json')) as f:
                d = json.load(f)
                (rews_0, rews_1,
                 p0_prop, p1_prop, p0_own, p0_other, p1_own, p1_other, p0_total, p1_total,
                 p0_petty, p1_petty) = process_results(d)
                rews_0_list.append(rews_0)
                rews_1_list.append(rews_1)
                p0_prop_list.append(p0_prop)
                p1_prop_list.append(p1_prop)
                p0_own_list.append(p0_own)
                p0_other_list.append(p0_other)
                p1_own_list.append(p1_own)
                p1_other_list.append(p1_other)
                p0_total_list.append(p0_total)
                p1_total_list.append(p1_total)
                petty_0_list.append(p0_petty)
                petty_1_list.append(p1_petty)

        return rews_0_list, rews_1_list, p0_prop_list, p1_prop_list, p0_own_list, p0_other_list, p1_own_list, p1_other_list, p0_total_list, p1_total_list, petty_0_list, petty_1_list


def get_means_and_sems(data_list):
    means = [np.mean(data, axis=0) for data in data_list]
    sems = [np.std(data, axis=0) / np.sqrt(len(data)) for data in data_list]
    return means, sems


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--results-path", type=str, default="")
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-s", "--save", action='store_true')
    parser.add_argument("-sm", "--smooth", type=int, default=1)
    parser.add_argument("-sym", action='store_true')
    parser.add_argument("-mfos", action='store_true')
    parser.add_argument("-m", "--multiple", action='store_true')
    args = parser.parse_args()
    name = args.name
    results_path = args.results_path
    save_figs = args.save
    n = args.smooth
    mfos = args.mfos
    sym = args.sym
    multiple = args.multiple

    rew_means = []
    rew_sems = []
    prob_means = []
    prob_sems = []
    petty_means = []
    petty_sems = []
    NAMES = []
    if multiple:
        for subdir in os.listdir(results_path):
            NAMES.append(subdir)
            if os.path.isdir(os.path.join(results_path, subdir)):
                (rews_0_list, rews_1_list,
                 p0_prop_list, p1_prop_list,
                 p0_own_list, p0_other_list, p1_own_list, p1_other_list, p0_total_list, p1_total_list,
                 petty_0_list, petty_1_list) = process_experiment(
                    os.path.join(results_path, subdir))

                if sym:
                    rews_list = rews_0_list + rews_1_list
                    props_list = p0_prop_list + p1_prop_list
                    own_list = p0_own_list + p1_own_list
                    other_list = p0_other_list + p1_other_list
                    rews_mean = np.mean(rews_list, axis=0)
                    prob_mean = np.mean(props_list, axis=0)
                    rews_sem = np.std(rews_list, axis=0) / np.sqrt(len(rews_list))
                    prob_sem = np.std(props_list, axis=0) / np.sqrt(len(props_list))

                    petty_mean = np.mean(petty_0_list + petty_1_list, axis=0)
                    petty_sem = np.std(petty_0_list + petty_1_list, axis=0) / np.sqrt(len(petty_0_list + petty_1_list))
                    own_mean = np.mean(own_list, axis=0)
                    other_mean = np.mean(other_list, axis=0)
                    own_sem = np.std(own_list, axis=0) / np.sqrt(len(own_list))
                    other_sem = np.std(other_list, axis=0) / np.sqrt(len(other_list))
                    rew_means.append(rews_mean)
                    rew_sems.append(rews_sem)
                    prob_means.append(prob_mean)
                    prob_sems.append(prob_sem)
                    petty_means.append(petty_mean)
                    petty_sems.append(petty_sem)

        plot_data = [(rew_means, rew_sems), (prob_means, prob_sems), (petty_means, petty_sems)]
        plot_labels = ['Average Reward (32 steps)', 'P(Own Coin)', 'Average Reciprocal Reward (32 steps)']
        plot_save_labels = ['reward', 'probs', 'petty_reward']
        for j, (mean_to_plot, sem_to_plot) in enumerate(plot_data):
            print(NAMES)
            shortest = min([pmeans.size for pmeans in mean_to_plot])
            # print([rew.size for rew in mean_to_plot])
            factors = [int(pmeans.size // shortest) for pmeans in mean_to_plot]
            # print(factors)

            for i in range(len(mean_to_plot)):
                plt.plot(mean_to_plot[i][::factors[i]], label=NAMES[i])
                plt.fill_between(range(mean_to_plot[i][::factors[i]].size),
                                 mean_to_plot[i][::factors[i]] - sem_to_plot[i][::factors[i]],
                                 mean_to_plot[i][::factors[i]] + sem_to_plot[i][::factors[i]], alpha=0.5)
            if len(mean_to_plot) > 1:
                plt.legend()
            plt.xlabel('Episode')
            plt.ylabel(plot_labels[j])
            # if name is not None:
            #     plt.title(name)
            if save_figs:
                plt.savefig(f'figures/sym_cg_{plot_save_labels[j]}.pdf')
            plt.show()
            plt.close()

        # plot_coin_game_rewards([rews_mean], [rews_sem], name=name, save=save_figs)
        # plot_coin_game_probs([prob_mean], [prob_sem], name=name, save=save_figs)
        # plot_coin_game_counts([own_mean], [other_mean], [own_sem], [other_sem], name=name, save=save_figs)

    else:
        rews_0_list, rews_1_list, p0_prop_list, p1_prop_list, p0_own_list, p0_other_list, p1_own_list, p1_other_list, p0_total_list, p1_total_list, petty_0_list, petty_1_list = process_experiment(
            results_path)
        rews_0_mean = np.mean(rews_0_list, axis=0)
        rews_1_mean = np.mean(rews_1_list, axis=0)
        p0_prop_mean = np.mean(p0_prop_list, axis=0)
        p1_prop_mean = np.mean(p1_prop_list, axis=0)
        p0_own_mean = np.mean(p0_own_list, axis=0)
        p0_other_mean = np.mean(p0_other_list, axis=0)
        p1_own_mean = np.mean(p1_own_list, axis=0)
        p1_other_mean = np.mean(p1_other_list, axis=0)
        p0_total_mean = np.mean(p0_total_list, axis=0)
        p1_total_mean = np.mean(p1_total_list, axis=0)

        print(rews_0_mean[-1], rews_1_mean[-1])
        print((rews_0_mean[-1] + rews_1_mean[-1]) / 2)

        rews_0_sem = np.std(rews_0_list, axis=0) / np.sqrt(len(rews_0_list))
        rews_1_sem = np.std(rews_1_list, axis=0) / np.sqrt(len(rews_1_list))
        p0_prop_sem = np.std(p0_prop_list, axis=0) / np.sqrt(len(p0_prop_list))
        p1_prop_sem = np.std(p1_prop_list, axis=0) / np.sqrt(len(p1_prop_list))
        p0_own_sem = np.std(p0_own_list, axis=0) / np.sqrt(len(p0_own_list))
        p0_other_sem = np.std(p0_other_list, axis=0) / np.sqrt(len(p0_other_list))
        p1_own_sem = np.std(p1_own_list, axis=0) / np.sqrt(len(p1_own_list))
        p1_other_sem = np.std(p1_other_list, axis=0) / np.sqrt(len(p1_other_list))
        p0_total_sem = np.std(p0_total_list, axis=0) / np.sqrt(len(p0_total_list))
        p1_total_sem = np.std(p1_total_list, axis=0) / np.sqrt(len(p1_total_list))

        # plot_coin_game_rewards([rews_0_mean, rews_1_mean], [rews_0_sem, rews_1_sem], name=name, save=save_figs)
        # plot_coin_game_probs([p0_prop_mean, p1_prop_mean], [p0_prop_sem, p1_prop_sem], name=name, save=save_figs)
        # plot_coin_game_counts([p0_own_mean, p1_own_mean], [p0_other_mean, p1_other_mean],
        #                       [p0_own_sem, p1_own_sem], [p0_other_sem, p1_other_sem], name=name, save=save_figs)

        petty_0_mean = np.mean(petty_0_list, axis=0)
        petty_1_mean = np.mean(petty_1_list, axis=0)
        petty_0_sem = np.std(petty_0_list, axis=0) / np.sqrt(len(petty_0_list))
        petty_1_sem = np.std(petty_1_list, axis=0) / np.sqrt(len(petty_1_list))
        total_mean = np.mean(petty_0_list + petty_1_list, axis=0)
        total_sem = np.std(petty_0_list + petty_1_list, axis=0) / np.sqrt(len(petty_0_list + petty_1_list))

        if sym:
            rews_list = rews_0_list + rews_1_list
            props_list = p0_prop_list + p1_prop_list
            # own_list = p0_own_list + p1_own_list
            # other_list = p0_other_list + p1_other_list
            rews_mean = np.mean(rews_list, axis=0)
            prob_mean = np.mean(props_list, axis=0)
            rews_sem = np.std(rews_list, axis=0) / np.sqrt(len(rews_list))
            prob_sem = np.std(props_list, axis=0) / np.sqrt(len(props_list))

            petty_mean = np.mean(petty_0_list + petty_1_list, axis=0)
            petty_sem = np.std(petty_0_list + petty_1_list, axis=0) / np.sqrt(len(petty_0_list + petty_1_list))
            # own_mean = np.mean(own_list, axis=0)
            # other_mean = np.mean(other_list, axis=0)
            # own_sem = np.std(own_list, axis=0) / np.sqrt(len(own_list))
            # other_sem = np.std(other_list, axis=0) / np.sqrt(len(other_list))

            plot_data = [(rews_mean, rews_sem), (prob_mean, prob_sem), (petty_mean, petty_sem)]
            plot_labels = ['Average Reward (32 steps)', 'P(Own Coin)', 'Average Reciprocal Reward (32 steps)']
            plot_save_labels = ['rewards', 'probs', 'petty_rewards']
            for j, (mean_to_plot, sem_to_plot) in enumerate(plot_data):
                plt.plot(mean_to_plot, label='Reward')
                plt.fill_between(range(mean_to_plot.size), mean_to_plot - sem_to_plot, mean_to_plot + sem_to_plot, alpha=0.5)
                plt.xlabel('Episode')
                plt.ylabel(plot_labels[j])
                # if name is not None:
                #     plt.title(name)
                if save_figs:
                    plt.savefig(f'figures/sym_{name}_cg_{plot_save_labels[j]}.pdf')
                plt.show()
                plt.close()

        else:
            plot_coin_game_rewards([rews_0_mean, rews_1_mean], [rews_0_sem, rews_1_sem], name=name, save=save_figs)
            plot_coin_game_probs([p0_prop_mean, p1_prop_mean], [p0_prop_sem, p1_prop_sem], name=name, save=save_figs)
            plot_coin_game_counts([p0_own_mean, p1_own_mean], [p0_other_mean, p1_other_mean], [p0_total_mean, p1_total_mean],
                                  [p0_own_sem, p1_own_sem], [p0_other_sem, p1_other_sem], [p0_total_sem, p1_total_sem], name=name, save=save_figs)
            # plt.plot(rews_0_sem, label='P0 Reciprocal Reward')
            # plt.plot(rews_1_sem, label='P1 Reciprocal Reward')
            # plt.plot(total_mean, label='Reciprocal Reward')
            # plt.fill_between(range(petty_0_mean.size), petty_0_mean - petty_0_sem, petty_0_mean + petty_0_sem, alpha=0.5)
            # plt.fill_between(range(petty_1_mean.size), petty_1_mean - petty_1_sem, petty_1_mean + petty_1_sem, alpha=0.5)
            # plt.fill_between(range(total_mean.size), total_mean - total_sem, total_mean + total_sem, alpha=0.5)
            # plt.legend()
            # plt.xlabel('Episode')
            # plt.ylabel('Average Reciprocal Reward (32 steps)')
            # plt.ylim(-10, 2)
            # if name is not None:
            #     plt.title(name)
            # if save_figs:
            #     plt.savefig(f'figures/{name}_coin_game_petty_rewards.pdf')
            # plt.show()

            # if 'petty_reward_0' in d[-1]:
            #     p0_petty = np.array([ep_data.get('petty_reward_0', 0.0) for ep_data in d])
            #     plt.plot(p0_petty, label='P0 Petty Reward')
            #
            # if 'petty_reward_1' in d[-1]:
            #     p1_petty = np.array([ep_data.get('petty_reward_1', 0.0) for ep_data in d])
            #     plt.plot(p1_petty, label='P1 Petty Reward')
