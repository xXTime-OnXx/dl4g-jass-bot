import copy
import numpy as np

from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.game_sim import GameSim
import logging

# Configure logging to output to Jupyter Notebook
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check if there are handlers already and clear them to prevent duplicate logs
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
else:
    logger.handlers.clear()  # Clear existing handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


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


class AgentTrumpMCTSSchieber(Agent):
    def __init__(self, n_simulations=1, n_determinizations=5):
        super().__init__()
        self._rule = RuleSchieber()
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
    
    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the best card to play using a combination of rule-based logic and MCTS, with detailed logging.
        """
        logger.debug("Starting action_play_card for player %d", obs.player)

        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)
        
        logger.debug("Valid cards for player %d: %s", obs.player, valid_card_indices)

        # If only one valid card, play it directly
        if len(valid_card_indices) == 1:
            logger.debug("Only one valid card found, playing card: %d", valid_card_indices[0])
            return valid_card_indices[0]

        # Rule-based heuristic checks
        trump_suit = obs.trump
        current_trick_points = sum(self._rule.point_value(card, trump_suit) for card in obs.current_trick if card != -1)
        highest_card_in_trick = max((card for card in obs.current_trick if card != -1), key=lambda x: self._rule.card_order(x, trump_suit), default=None)
        teammate_index = (obs.trick_first_player[obs.nr_tricks] + 2) % 4
        
        # Check if teammate holds the highest card in the current trick
        teammate_has_highest = highest_card_in_trick is not None and obs.trick_winner[obs.nr_tricks] == teammate_index
        logger.debug("Current trick points: %d, highest card in trick: %s", current_trick_points, highest_card_in_trick)
        print("TEST1")
        # Apply heuristic rules first
        for card in valid_card_indices:
            card_suit = color_of_card[card]
            card_value = self._rule.point_value(card, trump_suit)
            print("TEST2")
            # Rule 1: Play a strong trump if trick points are high and trump can win
            if current_trick_points >= 25 and card_suit == trump_suit:
                if highest_card_in_trick is None or self._rule.card_order(card, trump_suit) > self._rule.card_order(highest_card_in_trick, trump_suit):
                    logger.debug("Rule 1 applied: Playing trump card %d with high trick points %d", card, current_trick_points)
                    return card

            # Rule 2: Play the highest card if it can win and no trump in trick
            if highest_card_in_trick and card_suit == color_of_card[highest_card_in_trick] and self._rule.card_order(card, trump_suit) > self._rule.card_order(highest_card_in_trick, trump_suit):
                logger.debug("Rule 2 applied: Playing highest card %d to beat current highest %d", card, highest_card_in_trick)
                return card

            # Rule 3: Play a medium-value non-trump card if teammate has the highest card in trick
            if teammate_has_highest and card_value < 10 and card_suit != trump_suit:
                logger.debug("Rule 3 applied: Playing low-value card %d as teammate has highest card %d", card, highest_card_in_trick)
                return card

        # If no decision from heuristics, use MCTS to determine the best card
        logger.debug("No heuristic rules applied. Starting MCTS simulations.")
        card_scores = np.zeros(len(valid_card_indices))
        
        for determinization_idx in range(self.n_determinizations):
            determinization_hands = self._create_determinization(obs)
            determinization_scores = self._run_mcts_for_determinization(determinization_hands, obs, valid_card_indices)
            card_scores += determinization_scores
            logger.debug("Determinization %d: Scores from MCTS simulation: %s", determinization_idx, determinization_scores)
        
        # Select the card with the best score from MCTS
        best_card_index = np.argmax(card_scores)
        best_card = valid_card_indices[best_card_index]
        logger.debug("MCTS complete. Best card chosen: %d with score %f", best_card, card_scores[best_card_index])

        return best_card


    def _create_determinization(self, obs: GameObservation) -> np.ndarray:
        """
        Create a determinized version of the game state by assigning random plausible hands to opponents.
        """
        hands = self._deal_unplayed_cards(obs)

        return hands
    
    def _deal_unplayed_cards(self, obs: GameObservation):
        played_cards_per_round = obs.tricks
        played_cards = set([card for round in played_cards_per_round for card in round if card != -1])

        rounds_started_by = obs.trick_first_player
        num_cards_per_player = np.full(4, (9 - obs.nr_tricks))

        first_player = rounds_started_by[obs.nr_tricks]
        for i in range(4):
            player = (first_player + i) % 4
            if obs.current_trick[i] != -1:
                num_cards_per_player[player] -= 1

        all_cards = set(range(36))

        unplayed_cards = list(all_cards - played_cards)
        opponents_unplayed_cards = list(set(unplayed_cards) - set(convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)))

        np.random.shuffle(opponents_unplayed_cards)

        hands = np.zeros(shape=[4, 36], dtype=np.int32)

        hands[obs.player] = obs.hand

        for player in range(4):
            if player != obs.player:
                players_random_cards = opponents_unplayed_cards[:num_cards_per_player[player]]
                hands[player, players_random_cards] = 1
                opponents_unplayed_cards = opponents_unplayed_cards[num_cards_per_player[player]:]

        return hands
        

    def _run_mcts_for_determinization(self, hands: np.ndarray, obs: GameObservation, valid_card_indices: np.ndarray) -> np.ndarray:
        """
        Run multiple MCTS simulations for a given determinization and return scores for each valid card.
        """
        card_scores = np.zeros(len(valid_card_indices))
        
        for _ in range(self.n_simulations):
            # For each valid card, simulate the outcome by reinitializing the game simulation
            for i, card in enumerate(valid_card_indices):
                sim_game = GameSim(rule=self._rule)
                sim_game.init_from_state(copy.deepcopy(obs))
                sim_game._state.hands = copy.deepcopy(hands)

                # Simulate playing the card
                sim_game.action_play_card(card)
                
                # Play out the rest of the game randomly
                while not sim_game.is_done():
                    valid_cards_sim = self._rule.get_valid_cards_from_obs(sim_game.get_observation())
                    
                    # Check if there are any valid cards left
                    if np.flatnonzero(valid_cards_sim).size == 0:
                        # No valid cards, break out of the loop or handle the situation
                        break
                    
                    # Randomly play a valid card 
                    sim_game.action_play_card(np.random.choice(np.flatnonzero(valid_cards_sim)))
                
                # Update score based on the points scored for the simulation
                points = sim_game.state.points[self._team(obs.player)]
                card_scores[i] += points

        return card_scores

    def _team(self, player: int) -> int:
        """
        Determine the team number for the given player.
        Players 0 and 2 are in team 0, and players 1 and 3 are in team 1.
        """
        return player % 2