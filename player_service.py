from jass.service.player_service_app import PlayerServiceApp
import logging
from flask import request
from mcts_agent import AgentTrumpMCTSSchieber

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('player_service')

import os

def create_app():
    app = PlayerServiceApp('player_service')
    
    app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
    app.config['DEBUG'] = app.config['ENV'] == 'development'

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('player_service')

    @app.before_request
    def log_request_info():
        logger.info(f"Incoming request: {request.method} {request.url}")
        if request.data:
            logger.debug(f"Request Data: {request.data}")
    
    # Add the MCTS player
    app.add_player('mcts', AgentTrumpMCTSSchieber())
    
    return app

# Make sure this block is at the end of your main file
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8888)))
