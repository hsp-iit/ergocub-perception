import sys
from pathlib import Path

sys.path.insert(0,  Path(__file__).parent.parent.as_posix())
from configs.ar_rpc_config import Network


if __name__ == '__main__':
    ar = Network.node(**Network.Args.to_dict())
    ar.run()
