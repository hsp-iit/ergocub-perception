from multiprocessing.managers import BaseManager


def main():
    give = True
    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    out_queue = manager.get_queue('source_grasping')

    while True:
        input()
        out_queue.put({'action': 'give' if give else 'other'})
        give = not give


if __name__ == '__main__':
    main()
