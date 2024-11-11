import numpy as np

from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.game_sim import GameSim


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


class MCTSAgent(Agent):
    def __init__(self, n_simulations=200, n_determinizations=10):
        super().__init__()
        self._rule = RuleSchieber()
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
    
    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation.
        The trump selection will be handled using a heuristic as done in previous tasks.
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
        Perform the Monte Carlo Tree Search (MCTS) to select the best card to play
        based on multiple determinizations of the game state.
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            # Only one valid card, no need for MCTS
            return valid_card_indices[0]

        # Perform multiple determinizations and MCTS simulations
        card_scores = np.zeros(len(valid_card_indices))
        
        for _ in range(self.n_determinizations):
            determinization_hands = self._create_determinization(obs)
            card_scores += self._run_mcts_for_determinization(determinization_hands, obs, valid_card_indices)
        
        # Choose the card with the best score
        best_card_index = np.argmax(card_scores)
        return valid_card_indices[best_card_index]

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
                sim_game.init_from_state(obs)
                sim_game._state.hands = hands
                
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