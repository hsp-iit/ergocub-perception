import sys
import threading
from collections import defaultdict
from multiprocessing import process
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Queue

from loguru import logger

sys.path.insert(0,  Path(__file__).parent.parent.as_posix())

from utils.logging import setup_logger
from configs.manager_config import Logging, Network


setup_logger(Logging.level)

# {node: Queue(1) for node in Config.nodes}
queues = defaultdict(lambda: Queue(1))


def get_queue(name):
    return queues[name]


def main():

    logger.info('Starting communication server')

    BaseManager.register('get_queue', callable=get_queue)
    m = BaseManager(address=(Network.ip, Network.port), authkey=b'abracadabra')
    server = m.get_server()

    server.stop_event = threading.Event()
    process.current_process()._manager_server = server
    try:
        accepter = threading.Thread(target=server.accepter)
        accepter.daemon = True
        accepter.start()
        logger.success('Communication server online')
        try:
            while not server.stop_event.is_set():
                server.stop_event.wait(1)
        except (KeyboardInterrupt, SystemExit):
            pass
    finally:
        if sys.stdout != sys.__stdout__:
            logger.debug('resetting stdout, stderr')
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        sys.exit(0)


if __name__ == '__main__':
    main()
