import torch
import argparse
from math import pi

def parse_args():
    parser = argparse.ArgumentParser()

    # network type
    parser.add_argument("--project", default='TIE_physics_informed', type=str)
    parser.add_argument("--model_root", default='D:\\', type=str)
    parser.add_argument("--data_root", default='D:\\TIE_physics_informed\\data', type=str)
    parser.add_argument("--data_name_holo", default= "tie_bead_training_data", type=str)
    parser.add_argument("--save_name", default='unet', type=str)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=False, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--batch_mode", default='B', type=str)
    parser.add_argument("--initial_channel", default=32, type=int)



    # hyper-parameter
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--chk_iter", default=100, type=int)
    parser.add_argument("--visualize_chk_iter", default=1, type=int)
    parser.add_argument("--model_save_iter", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)  # 1e-4 is default
    parser.add_argument("--lr_decay_epoch", default=5, type=int)
    parser.add_argument("--lr_decay_rate", default=0.95, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--w_content", default=10.0, type=float)
    parser.add_argument("--w_identity", default=10.0, type=float)
    parser.add_argument("--w_distance", default=10.0, type=float)
    parser.add_argument("--penalty_regularizer", default=10.0, type=float)
    parser.add_argument("--w_gen", default=1.0, type=float)
    parser.add_argument("--distance_min", default=-200e-6, type=float)
    parser.add_argument("--distance_max", default=200e-6, type=float)
    parser.add_argument("--phase_normalize", default=1, type=float)

    # experiment parameter
    parser.add_argument("--pixel_size", default=1.67e-6, type=float)
    parser.add_argument("--wavelength", default=532e-9, type=float)


    return parser.parse_args()


def weights_initialize_normal(m):

    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:

        # apply a normal distribution to the weights and a bias=0
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.01)


def weights_initialize_xavier_normal(m):

    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
