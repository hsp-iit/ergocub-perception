import os
from pathlib import Path

from utils.confort import BaseConfig


class Config(BaseConfig):
    class Manager:
        run_process = True
        docker = False

        file = 'scripts/manager.py'

    class Source:
        run_process = True
        docker = False

        file = 'scripts/source.py'

    class Grasping:
        run_process = True
        docker = True
        file = 'scripts/grasping_pipeline.py'

        class Docker:
            image = 'ecub'
            name = 'ecub-grasping'
            options = ['-it', '--rm', '--gpus=all']
            volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']

    class ActionRec:
        run_process = False
        docker = True
        file = 'scripts/action_rec_pipeline.py'

        class Docker:
            image = 'ecub-env'
            name = 'ecub-action_rec'
            options = ['-it', '--rm', '--gpus=all']
            volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']

    class Sink:
        run_process = True
        docker = False

        file = 'scripts/sink.py'
