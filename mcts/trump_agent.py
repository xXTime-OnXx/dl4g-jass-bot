import numpy as np

from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent


# score if the color is trump
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0
    for card_index in cards:
        card_offset = offset_of_card[card_index]
        if color_of_card[card_index] == trump:
            score += trump_score[card_offset]
        else:
            score += no_trump_score[card_offset]
    return score

class AgentTrumpSchieber(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        
    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        scores = [calculate_trump_selection_score(card_list, trump) for trump in [0, 1, 2, 3]]
        highest_score_index = scores.index(max(scores))
        if scores[highest_score_index] > 68:
            return highest_score_index
        if obs.forehand == -1:
            return PUSH
        return highest_score_index

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        # we use the global random number generator here
        random_choice = np.random.choice(np.flatnonzero(valid_cards))
        return random_choice