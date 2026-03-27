import logging, sys
from config import LOG_DIR

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        fh = logging.FileHandler(LOG_DIR / "fraud.log")
        sh.setFormatter(fmt); fh.setFormatter(fmt)
        logger.addHandler(sh); logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger