import subprocess
from datetime import datetime
from dataclasses import dataclass
import torch

device = 'cuda'


@dataclass
class DataConfig:
    dataset_path = "data/ShapeNetCore.v2"
    partial_points = 2048
    multiplier_complete_sampling = 50
    implicit_input_dimension = 8192
    dist = [0.1, 0.4, 0.5]
    noise_rate = 0.01
    tolerance = 0.0
    train_samples = 10000
    val_samples = 1024

    n_classes = 1


@dataclass
class ModelConfig:
    knn_layer = 1
    device = device
    # Transformer
    n_channels = 3
    embed_dim = 384
    encoder_depth = 6
    mlp_ratio = 2.
    qkv_bias = False
    num_heads = 6
    attn_drop_rate = 0.
    drop_rate = 0.
    qk_scale = None
    out_size = 1024
    # Implicit Function
    hidden_dim = 32
    depth = 2
    # Others
    use_object_id = False
    use_deep_weights_generator = False
    assert divmod(embed_dim, num_heads)[1] == 0


def git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


@dataclass
class TrainConfig:
    device = device
    visible_dev = '0'
    lr = 1e-4
    wd = 0.0
    mb_size = 64
    n_epoch = 2000
    clip_value = 1 # 0.5?
    seed = 1  # int(datetime.now().timestamp())
    # WARNING: Each worker load a different batches so we may end up with
    #   20 * 64 batches loaded simultaneously. Moving the batches to cuda inside the
    #   dataset can lead to OOM errors
    num_workers = 30
    git = ""  # git_hash()
    optimizer = torch.optim.Adam
    loss = torch.nn.BCEWithLogitsLoss
    loss_reduction = "mean"  # "none"
    load_ckpt = None
    save_ckpt = f"{datetime.now().strftime('%d-%m-%y_%H-%M')}"
    overfit_mode = False


@dataclass
class EvalConfig:
    grid_eval = False
    grid_res_step = 0.01
    tolerance = DataConfig.tolerance
    dist = DataConfig.dist
    noise_rate = DataConfig.noise_rate

    mb_size = 8
    log_metrics_every = 100
    val_every = 10
    wandb = True
