import logging
import time

logger = logging.getLogger(__name__)


def loggraph(func):
    def wrapper(*args, **kwargs):
        logger.info(f'{'-' * 100}')
        logger.info(f"STARTED '{func.__name__}'")
        logger.info(f'{'-' * 100}')
        
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()

        logger.info(f'{'-' * 100}')
        logger.info(f"ENDED '{func.__name__}' IN {end_time - start_time:.0f} SECONDS")
        logger.info(f'{'-' * 100}')
        return output
    return wrapper
