import os
import platform


# input_type = "skeleton"  # rgb, skeleton or hybrid
docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
ubuntu = platform.system() == "Linux"
base_dir = "ISBFSAR"
engine_dir = "engines" if not docker else os.path.join("engines", "docker")


class TRXTrainConfig(object):
    def __init__(self, input_type="skeleton"):
        # MAIN
        self.model = "DISC"  # DISC or EXP
        self.input_type = input_type  # skeleton or rgb
        self.way = 5
        self.shot = 5
        self.device = 'cuda'
        self.skeleton_type = 'smpl+head_30'

        # CHOICE DATASET
        data_name = "NTURGBD_to_YOLO_METRO" if ubuntu else "NTURGBD_to_YOLO_METRO_122"
        self.data_path = f"D:\\datasets\\{data_name}" if not ubuntu else f"../datasets/{data_name}"
        self.n_joints = 30

        # TRAINING
        self.initial_lr = 1e-2 if self.input_type == "skeleton" else 3e-4
        self.n_task = (100 if self.input_type == "skeleton" else 30) if not ubuntu else (1000 if self.input_type == "skeleton" else 500)
        self.optimize_every = 1  # Put to 1 if not used, not 0 or -1!
        self.batch_size = 1 if not ubuntu else (32 if self.input_type == "skeleton" else 4)
        self.n_epochs = 10000
        self.start_discriminator_after_epoch = 0  # self.n_epochs  # TODO CAREFUL
        self.first_mile = self.n_epochs  # 15 TODO CAREFUL
        self.second_mile = self.n_epochs  # 1500 TODO CAREFUL
        self.n_workers = 0 if not ubuntu else 16
        self.log_every = 10 if not ubuntu else 1000
        self.eval_every_n_epoch = 10

        # MODEL
        self.trans_linear_in_dim = 512 if self.input_type == "hybrid" else 256
        self.trans_linear_out_dim = 128
        self.query_per_class = 1
        self.trans_dropout = 0.
        self.num_gpus = 4
        self.temp_set = [2]
        self.checkpoints_path = "checkpoints"

        # DEPLOYMENT
        if input_type == "rgb":
            self.final_ckpt_path = os.path.join(base_dir, "modules", "ar", "modules", "raws", "rgb", "3000.pth")
        elif input_type == "skeleton":
            self.final_ckpt_path = os.path.join(base_dir, "modules", "ar", "modules", "raws", "5-w-1-s.pth")
        elif input_type == "hybrid":
            self.final_ckpt_path = os.path.join(base_dir, "modules", "ar", "modules", "raws", "hybrid",
                                                "1714_truncated_resnet.pth")
        self.trt_path = os.path.join(base_dir, "modules", "ar", engine_dir, "trx.engine")
        self.seq_len = 8 if input_type != "skeleton" else 16
