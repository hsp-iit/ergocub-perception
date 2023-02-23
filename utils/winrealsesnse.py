import subprocess
import time

from loguru import logger
from multiprocessing.managers import BaseManager

from utils.input import RealSense


# This can run on a WSL or in a docker and will allow to read the rgb and depth
# frames streamed by a script running on windows.
# This is necessary since wsl is not able to easily mount usb devices

# The ip is in /etc/resolv.conf
# If you are running a docker mount the directory and set network=host

# Make sure that the name of the input queue matches the one of the output queue used
# by the windows script
def connect(manager):
    logger.info('Connecting to manager...')
    start = time.time()

    while True:
        try:
            manager.connect()
            break
        except ConnectionRefusedError as e:
            if time.time() - start > 120:
                logger.error('Connection refused.')
                raise e
            time.sleep(1)
    logger.success('Connected to manager.')


def get_ip():
    command = "cat /etc/resolv.conf | grep nameserver | awk '{print $2; exit;}'"

    task = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    ip = task.stdout.read().decode('utf-8').strip()

    return ip


class WinRealSense(RealSense):

    def __init__(self, *args, **kwargs):
        ip = get_ip()

        BaseManager.register('get_queue')
        manager = BaseManager(address=(ip, 4000), authkey=b'abracadabra')
        connect(manager)

        self.in_queue = manager.get_queue('windows_out')

    def read(self):
        data = self.in_queue.get()

        return data.values()

