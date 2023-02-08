import multiprocessing
import sys
import __main__
from pathlib import Path

from loguru import logger


# TensorRT logging is handled separately inside the Runner classes
def setup_logger(level=0, name=None, recurring=False):
    """level: set the minimum severity level to be printed out
        name: sets the name displayed in the error messages
        disable: makes all logger calls no-op
        recurring: if False, logger calls with recurring=True are not printed.
                   Used to toggle logging messages inside the main application loops
    """

    if isinstance(level, list):
        lvl_filter = lambda msg: (msg['level'].no in level) and not\
                                 (msg['extra'].get('recurring', False) and (not recurring))
    else:
        lvl_filter = lambda msg: (msg['level'].no >= level) and not\
                                 (msg['extra'].get('recurring', False) and (not recurring))

    if name is None:
        multiprocessing.current_process().name = Path(__main__.__file__).stem

    logger.remove()

    def formatter(record):
        r = "🔁" if record['extra'].get('recurring') else ""
        return ("<fg magenta>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> <yellow>|</>"
                " <lvl>{level: <8}</> "
                f"<yellow>|</> <blue>{{process.name: ^12}}</> <yellow>-</> <lvl>{{message}} {r}</>\n")

    logger.add(sys.stdout,
               format=formatter,
               diagnose=True, filter=lvl_filter)  # b28774 (magenta)

    logger.level('INFO', color='<fg white>')  # fef5ed
    logger.level('SUCCESS', color='<fg green>')  # 79d70f
    logger.level('WARNING', color='<fg yellow>')  # fd811e
    logger.level('ERROR', color='<fg red>')  # ed254e

    return
