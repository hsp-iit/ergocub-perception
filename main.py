from subprocess import Popen
from configs.main_config import Config
from utils.misc import Shell

if __name__ == '__main__':
    shell = Shell('cmd')

    python_procs = [p for p in Config if p.run_process and not p.docker]
    docker_procs = [p for p in Config if p.run_process and p.docker]

    for pr in python_procs:
        cmd = f'python {pr.file}'
        shell.add_pane(cmd)

    for pr in docker_procs:
        docker = pr.Docker
        cmd = f'docker run --name {docker.name}' \
              f' {" ".join(docker.options)}' \
              f' -v {" -v ".join(docker.volumes)}' \
              f' {docker.image} python {pr.file}'
        shell.add_pane(cmd)

    shell.start()

    print('Press any key to kill the containers')
    input()

    Popen(f'docker kill'.split(' ') + [pr.Docker.name for pr in docker_procs])
