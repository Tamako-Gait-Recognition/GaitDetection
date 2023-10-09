import argparse
import os

import torch
import gaitRecognition.models.ResGCNv1
from gaitRecognition.datasets.graph import Graph
from einops.layers.torch import Rearrange
import torch.nn as nn
from gaitRecognition.prediction.evaluate import evaluation_function
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
class opt:

    point_noise_std = 0.05
    joint_noise_std = 0.1
    flip_probability = 0.5
    mirror_probability = 0.5
    balance_sampler = True
    train_data_path = r"E:\code file pycharm\GaitDetection\data\frames\pose_coco.csv"   #训练集
    valid_data_path = r"E:\code file pycharm\GaitDetection\data\frames\pose_coco.csv"   #测试集
    loss_func = "triplet"
    balance_sampler = True
    kernel_frame = 31
    joint_drop = "none"
    model_type = "spatialtransformer_temporalconv"
    weight_path = r"./src/gaitRecognition/GaitMixer.pt"
    sampler_num_sample = 1
    num_workers = 0
    gpus = 0
    batch_size = 128
    batch_size_validation = 256
    start_epoch = 1
    test_epoch_interval = 10
    save_model = False
    use_amp = False
    tune = False
    shuffle = False
    project = "project_name"
    name = "project_name"
    rm_conf = True
    epochs = 1
    sequence_length = 60
    embedding_layer_size = 128
    embedding_spatial_size = 32
    learning_rate = 6e-3
    debug = True
    weight_decay = 1e-5
    train_id = 3
    cuda = True
    evaluation_fn = evaluation_function
def parse_option():
    return opt
'''
def parse_option():
    parser = argparse.ArgumentParser(
        description="Training model on gait sequence")
    parser.add_argument("dataset", choices=["casia-b-query", "casia-b"])   #选择数据集
    parser.add_argument("train_data_path", help="Path to train data CSV")  #训练集的csv
    parser.add_argument("--valid_data_path",
                        help="Path to validation data CSV")   #合法数据的csv
    parser.add_argument("--train_id",
                        help="train_id")
    parser.add_argument("--test_id",
                        help="test_id")
    parser.add_argument("--loss_func", choices=["supcon", "triplet"], default="triplet")
    parser.add_argument('--balance_sampler', type=str2bool,
                        nargs='?', default=True)
    parser.add_argument("--kernel_frame", type=int, default=31)
    parser.add_argument("--rm_conf", type=str2bool,
                        nargs='?', default=True)
    parser.add_argument("--joint_drop", choices=["single", "none"], default="none") # rm drop arms
    parser.add_argument("--sampler_num_sample", type=int, default=4)
    parser.add_argument("--weight_path", help="Path to weights for model")
    parser.add_argument("--model_type", choices=["spatialtransformer_temporalconv", "spatiotemporal_transformer", "gaitgraph"], default="spatialtransformer_temporalconv")

    # Optionals
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--gpus", default="0", help="-1 for CPU, use comma for multiple gpus"
    )
    parser.add_argument("--batch_size", type=int, default=128)   #训练批次数
    parser.add_argument("--batch_size_validation", type=int, default=256)  #合法批次数
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--test_epoch_interval", type=int, default=10)
    parser.add_argument('--save_model', type=str2bool,
                        nargs='?', default=False)
    parser.add_argument("--use_amp", action="store_true")  #store_true在命令行指定了该参数，怎么use_amp就为true
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--project", default="project_name")
    parser.add_argument("--name", help="experiment_name", default="project_name")

    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--embedding_layer_size", type=int, default=128)
    parser.add_argument("--embedding_spatial_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=6e-3)
    parser.add_argument("--point_noise_std", type=float, default=0.05)
    parser.add_argument("--joint_noise_std", type=float, default=0.1)
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--mirror_probability", type=float, default=0.5)
    parser.add_argument('--debug', type=str2bool, nargs='?', default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    opt = parser.parse_args()

    # Sanitize opts
    # opt.gpus_str = opt.gpus
    # opt.gpus = [i for i in range(len(opt.gpus.split(",")))]

    return opt
'''

def get_model_resgcn():
    graph = Graph("coco")
    model_args = {
        "A": torch.tensor(graph.A, dtype=torch.float32, requires_grad=False),
        "num_class": 128,
        "num_input": 1,
        "num_channel": 3,
        "parts": graph.parts,
    }
    return nn.Sequential(
        Rearrange('b f j e -> b 1 e f j'),
        models.ResGCNv1.create('resgcn-n39-r8', **model_args))


def load_checkpoint(model, opt):
    # print(os.getcwd())
    if opt.weight_path is not None:
        checkpoint = torch.load(opt.weight_path)
        model.load_state_dict(checkpoint, strict=True)