from utils.input import RealSense

import time
from multiprocessing.managers import BaseManager


def connect(manager):
    print('Connecting to manager...')
    start = time.time()

    while True:
        try:
            manager.connect()
            break
        except ConnectionRefusedError as e:
            if time.time() - start > 120:
                print('Connection refused.')
                raise e
            time.sleep(1)
    print('Connected to manager.')


def main():
    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    connect(manager)

    send_out = manager.get_queue('windows_out')
    # get_in = manager.get_queue('docker_out')

    camera = RealSense() #postprocessing=True)

    while True:
        rgb, depth = camera.read()

        send_out.put({'rgb': rgb, 'depth': depth})
        # res = get_in.get()
        #
        # cv2.imshow('', res)
        # cv2.waitKey(1)


if __name__ == '__main__':
    main()


