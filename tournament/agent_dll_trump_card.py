import copy
import numpy as np

from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.game_sim import GameSim
import logging
from tensorflow.keras.models import load_model
import time

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
    
        

class AgentDLTrumpCardPlay(Agent):
    
    def __init__(self, n_simulations=1, n_determinizations=90):
        super().__init__()
        self._rule = RuleSchieber()
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
        self.trump_model = load_model('../model/trump_model_592.h5')
        self.play_card_model = load_model('../model/card_prediction_v02.keras')
    
    
    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the best card to play using rule-based logic only.
        """
        # play single valid card available
        valid_cards_encoded = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards_encoded)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]
        
        played_cards = obs.tricks.flatten()
        played_cards = played_cards[played_cards != -1]  # Remove unplayed card markers (-1)
        played_cards_encoded = get_cards_encoded(played_cards)
        
        trump_encoded = [np.int32(obs.trump == trump) for trump in trump_ints]

        # TODO: prepare data for model predition [played_cards, valid_cards, trump]
        model_input = np.concatenate([np.array(played_cards_encoded), np.array(valid_cards_encoded), np.array(trump_encoded)])
        prediction = self.play_card_model(model_input.reshape(1, -1))
        
        # TODO: map model output to return card value
        highest_index = np.argmax(prediction)
        logger.debug(f"Highest score index: {convert_int_encoded_cards_to_str_encoded([highest_index])}")
        return highest_index
    
    
    def action_trump(self, obs: GameObservation) -> int:
        hand = obs.hand

        if obs.forehand == -1:
            hand = np.append(hand, 1)
        else:
            hand = np.append(hand, 0)
        logger.debug("Hand: " + str(hand))
        
        hand = hand.reshape(1, -1)
        
        trump_model = self.trump_model
        probabilities = trump_model.predict(hand)
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