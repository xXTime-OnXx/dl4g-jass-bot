from jass.service.player_service_app import PlayerServiceApp
import logging
from flask import request
from rulebased_agent import AgentRuleBasedSchieber

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('player_service')


def create_app():
    app = PlayerServiceApp('player_service')
    app.add_player('rule', AgentRuleBasedSchieber())
    @app.before_request
    def log_request_info():
        logger.info(f"Incoming request: {request.method} {request.url}")
        if request.data:
            logger.debug(f"Request Data: {request.data}")
    

    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8888)
