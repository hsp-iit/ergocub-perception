from logging import INFO

from utils.confort import BaseConfig


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    ip = 'localhost'
    port = 50000

