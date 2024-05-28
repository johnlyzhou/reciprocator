import torch
from torch import nn

from src.environments.utils import label_to_state


def get_index(p1: int, p2: int, c1: int, c2: int):
    return p1 * 9 ** 3 + p2 * 9 ** 2 + c1 * 9 + c2


def compute_transition_value(value_function: nn.Module, joint_transition_probs: torch.Tensor,
                             cf_transition_probs: torch.Tensor, num_players: int, num_transition_classes: int,
                             player_idx: int = None):
    """
    Compute the value of the transition probabilities for each state.
    :param value_function: Value function to evaluate the states.
    :param joint_transition_probs: (bsz * num_players * 2, num_transition_classes)
    :param cf_transition_probs: (bsz * num_players * 2, num_transition_classes)
    :param num_players: Number of players in the game.
    :param num_transition_classes: Number of possible transition classes.
    :param player_idx: Index of the player whose value function is being computed
    :return: Net change in expected return from counterfactual transition probabilities of shape (bsz,)
    """

    joint_transition_probs = torch.exp(joint_transition_probs.view(-1, num_players * 2, num_transition_classes))
    cf_transition_probs = torch.exp(cf_transition_probs.view(-1, num_players * 2, num_transition_classes))
    transition_reward_influence = torch.zeros(cf_transition_probs.shape[0], device=joint_transition_probs.device)
    possible_classes = torch.tensor(range(num_transition_classes)).to(joint_transition_probs.device)
    possible_labels = torch.cartesian_prod(possible_classes, possible_classes, possible_classes, possible_classes)
    possible_states = label_to_state(possible_labels, 3, 3)
    with torch.no_grad():
        state_values = value_function(possible_states.float())

    for p1 in range(num_transition_classes):
        for p2 in range(num_transition_classes):
            for coin_1 in range(num_transition_classes):
                for coin_2 in range(num_transition_classes):
                    joint_probs = (joint_transition_probs[:, 0, p1] * joint_transition_probs[:, 1, p2] *
                                   joint_transition_probs[:, 2, coin_1] * joint_transition_probs[:, 3, coin_2])
                    cf_probs = (cf_transition_probs[:, 0, p1] * cf_transition_probs[:, 1, p2] *
                                cf_transition_probs[:, 2, coin_1] * cf_transition_probs[:, 3, coin_2])
                    # batch_mi = 1 - torch.div(cf_probs, joint_probs).clamp(0, 1)
                    batch_mi = joint_probs - cf_probs
                    state_value = state_values[get_index(p1, p2, coin_1, coin_2)]
                    if player_idx is not None:
                        state_value = state_value[player_idx]
                    transition_reward_influence += batch_mi * state_value

    return transition_reward_influence
