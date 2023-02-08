import sys
from pathlib import Path

sys.path.insert(0,  Path(__file__).parent.parent.as_posix())
from configs.od_rpc_config import Network


if __name__ == '__main__':
    grasping = Network.node(**Network.Args.to_dict())
    grasping.run()
