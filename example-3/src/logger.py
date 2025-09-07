import logging
logging.basicConfig(level=logging.INFO)

# Disabling logs for modules
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Exporting logger
logger = logging.getLogger()

# Creating annotation to wrap around langgraph methods
def loggraph(f):
    def wrap(*args, **kwargs):
        logger.info(f"{"-" * 10} {f.__name__} {"-" * 10}")
        o = f(*args, **kwargs)
        logger.info(f"{"-" * (22 + len(f.__name__))}")
        return o
    return wrap