import numpy as np
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

class AgentRuleBasedSchieberAdvanced(Agent):
    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()
    
    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the best card to play using rule-based logic only.
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]

        trump_suit = obs.trump
        current_trick_points = sum(calculate_point_value(card, trump_suit) for card in obs.current_trick if card != -1)
        
        highest_card_in_trick = max(
            (card for card in obs.current_trick if card != -1),
            key=lambda x: get_card_order(x, trump_suit),
            default=None
        )

        teammate_index = (obs.trick_first_player[obs.nr_tricks] + 2) % 4
        teammate_has_highest = (
            highest_card_in_trick is not None and 
            obs.trick_winner[obs.nr_tricks] == teammate_index and 
            color_of_card[highest_card_in_trick] == trump_suit
        )

        remaining_trumps = np.sum([1 for card in obs.current_trick if color_of_card[card] == trump_suit])

        # Fall 1: Partner hat die höchste Karte und es gibt wenige Trümpfe
        if teammate_has_highest and remaining_trumps <= 2:
            # Suche eine Karte mit möglichst hohem Wert, die aber die Karte des Partners nicht schlägt
            lower_value_cards = [c for c in valid_card_indices if get_card_order(c, trump_suit) < get_card_order(highest_card_in_trick, trump_suit)]
            if lower_value_cards:
                best_card = max(lower_value_cards, key=lambda x: calculate_point_value(x, trump_suit))
                print("Teammate push")
                return best_card

        # Fall 2: Hohe Stichpunkte und Trumpfkarte
        if current_trick_points >= 27:
            for card in valid_card_indices:
                if color_of_card[card] == trump_suit:
                    if highest_card_in_trick is None or get_card_order(card, trump_suit) > get_card_order(highest_card_in_trick, trump_suit):
                        print("Stechen")
                        return card

        # Fall 3: Trumpf-Konter
        for card in valid_card_indices:
            card_suit = color_of_card[card]
            if highest_card_in_trick and card_suit == color_of_card[highest_card_in_trick] and get_card_order(card, trump_suit) > get_card_order(highest_card_in_trick, trump_suit):
                print("Trumpf Konter")
                return card

        # Zufällige Auswahl, falls keine Regel zutrifft
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
