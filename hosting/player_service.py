from jass.service.player_service_app import PlayerServiceApp
from mcts.mcts_agent import AgentTrumpMCTSSchieber
import logging
from flask import request

# Configure logging to include debug-level messages and specify format for better clarity
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('player_service')

def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('random', AgentTrumpMCTSSchieber())

    # Add a before_request hook to log each incoming request
    @app.before_request
    def log_request_info():
        logger.info(f"Incoming request: {request.method} {request.url}")
        logger.debug(f"Headers: {request.headers}")
        if request.data:
            logger.debug(f"Body: {request.data}")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000)  # Set the port to 8000
