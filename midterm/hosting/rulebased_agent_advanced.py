import numpy as np
import copy

from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent

# Score constants
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]

def calculate_point_value(card, trump_suit):
    """
    Calculate the point value of a card, considering if it is a trump card or not.
    """
    card_offset = offset_of_card[card]
    if color_of_card[card] == trump_suit:
        return trump_score[card_offset]
    else:
        return no_trump_score[card_offset]

def get_card_order(card, trump_suit):
    """
    Determine the ranking of a card, considering trump suit.
    Higher values indicate higher ranks.
    """
    card_color = color_of_card[card]
    card_offset = offset_of_card[card]
    if card_color == trump_suit:
        trump_ranking = [11, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1]
        return trump_ranking[card_offset]
    else:
        return 9 - card_offset

def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0
    for card_index in cards:
        card_offset = offset_of_card[card_index]
        if color_of_card[card_index] == trump:
            score += trump_score[card_offset]
        else:
            score += no_trump_score[card_offset]
    return score

def highest_card_in_trick(trick, obs: GameObservation):
    amount_played_cards = len([card for card in trick if card != -1])
    trump = obs.trump
    color_of_first_card = color_of_card[trick[0]]
    if color_of_first_card == trump:
        # trump mode and first card is trump: highest trump wins
        winner = 0
        highest_card = trick[0]
        for i in range(1, amount_played_cards):
            # lower_trump[i,j] checks if j is a lower trump than i
            if color_of_card[trick[i]] == trump and lower_trump[trick[i], highest_card]:
                highest_card = trick[i]
                winner = i

        return highest_card, winner
        

    else:
        # trump mode, but different color played on first move, so we have to check for higher cards until
        # a trump is played, and then for the highest trump
        winner = 0
        highest_card = trick[0]
        trump_played = False
        trump_card = None
        for i in range(1, amount_played_cards):
            if color_of_card[trick[i]] == trump:
                if trump_played:
                    # second trump, check if it is higher
                    if lower_trump[trick[i], trump_card]:
                        winner = i
                        trump_card = trick[i]
                else:
                    # first trump played
                    trump_played = True
                    trump_card = trick[i]
                    winner = i
            elif trump_played:
                # color played is not trump, but trump has been played, so ignore this card
                pass
            elif color_of_card[trick[i]] == color_of_first_card:
                # trump has not been played and this is the same color as the first card played
                # so check if it is higher
                if trick[i] < highest_card:
                    highest_card = trick[i]
                    winner = i

        return highest_card, winner

    

class AgentRuleBasedSchieberAdvanced(Agent):
    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()
    
    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the best card to play using rule-based logic only.
        """
        # play single valid card available
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]

        # check if stabbing (staeche) or not
        trump_suit = obs.trump
        current_trick_points = sum(calculate_point_value(card, trump_suit) for card in obs.current_trick if card != -1)
        
        highest_card, winner = highest_card_in_trick(obs.current_trick, obs)
        winner_id = (obs.trick_first_player[obs.nr_tricks] + winner) % 4

        for card in valid_card_indices:
            card_suit = color_of_card[card]
            card_value = calculate_point_value(card, trump_suit)

            # falls eigene karte besser als gespielte karten -> stechen
            print(f'trick: {obs.current_trick}, points: {current_trick_points}')
            print(f'card: {card}, card color: {card_suit}, trup: {trump_suit}')
            if current_trick_points >= 15 and card_suit == trump_suit:
                trick = copy.deepcopy(obs.current_trick)
                trick[len([card for card in trick if card != -1])] = card
                new_highest_card, new_winner = highest_card_in_trick(trick, obs)
                print(f'new highest card: {new_highest_card}')
                if new_highest_card == card:
                    print(f'stab with: {card}')
                    return card
        
        for card in valid_card_indices:
            card_suit = color_of_card[card]
            card_value = calculate_point_value(card, trump_suit)

            trick = copy.deepcopy(obs.current_trick)
            card_index = len([card for card in trick if card != -1])
            trick[card_index] = card
            new_highest_card, new_winner = highest_card_in_trick(trick, obs)
            if new_highest_card == card:
                return card

        return np.random.choice(valid_card_indices)

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation.
        """
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        scores = [calculate_trump_selection_score(card_list, trump) for trump in [0, 1, 2, 3]]
        highest_score_index = scores.index(max(scores))
        if scores[highest_score_index] > 68:
            return highest_score_index
        if obs.forehand == -1:
            return PUSH
        return highest_score_index
