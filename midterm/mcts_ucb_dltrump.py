import copy
import numpy as np
import math
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.game_sim import GameSim
import logging
from tensorflow.keras.models import load_model
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
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

def calculate_point_value(card, trump_suit):
    card_offset = offset_of_card[card]
    if color_of_card[card] == trump_suit:
        return trump_score[card_offset]
    else:
        return no_trump_score[card_offset]
    
def highest_card_in_trick(trick, obs: GameObservation):
    amount_played_cards = len([card for card in trick if card != -1])
    trump = obs.trump
    color_of_first_card = color_of_card[trick[0]]
    if color_of_first_card == trump:
        winner = 0
        highest_card = trick[0]
        for i in range(1, amount_played_cards):
            if color_of_card[trick[i]] == trump and lower_trump[trick[i], highest_card]:
                highest_card = trick[i]
                winner = i
        return highest_card, winner
    else:
        # trump mode, but different color played on first move
        winner = 0
        highest_card = trick[0]
        trump_played = False
        trump_card = None
        for i in range(1, amount_played_cards):
            if color_of_card[trick[i]] == trump:
                if trump_played:
                    if lower_trump[trick[i], trump_card]:
                        winner = i
                        trump_card = trick[i]
                else:
                    trump_played = True
                    trump_card = trick[i]
                    winner = i
            elif trump_played:
                pass
            elif color_of_card[trick[i]] == color_of_first_card:
                if trick[i] < highest_card:
                    highest_card = trick[i]
                    winner = i
        return highest_card, winner


class AgentDLTrumpUCBMCTSSchieber(Agent):
    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()
        self.model = load_model('trump_model_592.h5')
        self.TIME_LIMIT = 9.5  

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the best card to play.
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]

        trump_suit = obs.trump
        current_trick_points = sum(calculate_point_value(card, trump_suit) for card in obs.current_trick if card != -1)

        # Heuristic: If you can stab (stechen) when it's worthwhile
        for card in valid_card_indices:
            card_suit = color_of_card[card]
            if current_trick_points >= 15 and card_suit == trump_suit:
                trick = copy.deepcopy(obs.current_trick)
                trick[len([c for c in trick if c != -1])] = card
                new_highest_card, new_winner = highest_card_in_trick(trick, obs)
                if new_highest_card == card:
                    return card

        # If no trump stab, try to win the trick if possible
        for card in valid_card_indices:
            trick = copy.deepcopy(obs.current_trick)
            card_index = len([c for c in trick if c != -1])
            trick[card_index] = card
            new_highest_card, new_winner = highest_card_in_trick(trick, obs)
            if new_highest_card == card:
                return card

        # Otherwise, use MCTS time-based approach
        card_scores = np.zeros(len(valid_card_indices))
        card_plays = np.zeros(len(valid_card_indices), dtype=int)

        start_time = time.time()
        # Run as many determinizations and simulations as possible within the time limit
        while time.time() < start_time + self.TIME_LIMIT:
            determinization_hands = self._create_determinization(obs)
            self._run_mcts_for_determinization(obs, valid_card_indices, determinization_hands, 
                                               card_scores, card_plays, start_time)

        avg_scores = card_scores / np.maximum(card_plays, 1)
        best_card_index = np.argmax(avg_scores)
        best_card = valid_card_indices[best_card_index]

        return best_card

    def _create_determinization(self, obs: GameObservation) -> np.ndarray:
        return self._deal_unplayed_cards(obs)

    def _deal_unplayed_cards(self, obs: GameObservation):
        played_cards_per_round = obs.tricks
        played_cards = set([card for round_ in played_cards_per_round for card in round_ if card != -1])

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

    def _simulate_card_play(self, obs: GameObservation, hands: np.ndarray, card: int) -> int:
        sim_game = GameSim(rule=self._rule)
        sim_game.init_from_state(copy.deepcopy(obs))
        sim_game._state.hands = copy.deepcopy(hands)

        sim_game.action_play_card(card)

        while not sim_game.is_done():
            valid_cards_sim = self._rule.get_valid_cards_from_obs(sim_game.get_observation())
            valid_indices = np.flatnonzero(valid_cards_sim)
            if len(valid_indices) == 0:
                break
            sim_game.action_play_card(np.random.choice(valid_indices))

        points = sim_game.state.points[self._team(obs.player)]
        return points

    def _run_mcts_for_determinization(self, obs: GameObservation, valid_card_indices: np.ndarray, 
                                      hands: np.ndarray, card_scores: np.ndarray, card_plays: np.ndarray,
                                      start_time: float):
        """
        Run simulations until time runs out, using UCB1 to choose cards.
        """
        c = 2.0 
        while time.time() < start_time + self.TIME_LIMIT:
            N = np.sum(card_plays)

            ucb_values = np.zeros(len(valid_card_indices))
            for i in range(len(valid_card_indices)):
                if card_plays[i] == 0:
                    ucb_values[i] = float('inf')
                else:
                    avg_reward = card_scores[i] / card_plays[i]
                    ucb_values[i] = avg_reward + c * math.sqrt(math.log(N) / card_plays[i])

            best_i = np.argmax(ucb_values)
            chosen_card = valid_card_indices[best_i]

            points = self._simulate_card_play(obs, hands, chosen_card)

            card_scores[best_i] += points
            card_plays[best_i] += 1

            if time.time() >= start_time + self.TIME_LIMIT:
                break

    def _team(self, player: int) -> int:
        return player % 2
    
    def action_trump(self, obs: GameObservation) -> int:
        hand = obs.hand
        if obs.forehand == -1:
            hand = np.append(hand, 1)
        else:
            hand = np.append(hand, 0)
        logger.debug("Hand: " + str(hand))
        
        hand = hand.reshape(1, -1)
        
        model = self.model
        probabilities = model.predict(hand)
        logger.debug("Model probabilities: " + str(probabilities))

        trump_categories = [
            "trump_DIAMONDS",
            "trump_HEARTS",
            "trump_SPADES",
            "trump_CLUBS",
            "trump_OBE_ABE",
            "trump_UNE_UFE",
            "trump_PUSH",
        ]

        scores = probabilities[0]
        
        trump_push_index = trump_categories.index("trump_PUSH")
        ignored_indices = [
            trump_categories.index("trump_OBE_ABE"),
            trump_categories.index("trump_UNE_UFE"),
        ]

        if scores[trump_push_index] > 0.8:
            logger.debug("Decision: Push (Threshold exceeded)")
            return trump_push_index

        filtered_scores = [
            score if idx not in ignored_indices else -1 for idx, score in enumerate(scores)
        ]

        highest_score_index = filtered_scores.index(max(filtered_scores))
        logger.debug(f"Filtered scores: {filtered_scores}")
        logger.debug(f"Highest score index: {highest_score_index}")

        return highest_score_index
