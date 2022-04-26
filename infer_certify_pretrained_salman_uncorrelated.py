"""
infer and certify base models
"""
import argparse
import datetime
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms.transforms import ToTensor, Resize, Compose

from smoothadv.core import Smooth
from smoothadv.patch_model import PatchSmooth, PreprocessLayer
from smoothadv.patch_model import PatchModel
from smoothadv.architectures import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=[
                        'cifar10', 'imagenet', 'imagenette'])
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='results/')
    parser.add_argument('-mp', '--model_path',
                        help='Model path to load model', type=str, required=True)
    parser.add_argument('-mt', '--mtype', help='Model type', choices=timm.list_models(
        pretrained=True).extend(['cifar_resnet110, cifar_resnet20']))
    parser.add_argument('-dpath', help='Data path',
                        default='/data/datasets/Imagenet/val')
    parser.add_argument('--gpu', help='gpu to use', default='0', type=str)
    parser.add_argument(
        '-it', '--iter', help='No. of iterations', type=int, default=40)
    parser.add_argument('-ni', '--num_images',
                        help='Number of images to be tested', default=100, type=int)
    parser.add_argument(
        '-clip', '--clip', help='Clip perturbations to original image intensities', action='store_true')
    parser.add_argument('-ps', '--patch_size',
                        help='Patch size', default=224, type=int)
    parser.add_argument('-pstr', '--patch_stride',
                        help='Patch Stride', default=1, type=int)
    parser.add_argument('-np', '--num_patches',
                        help='Maximum number of patches to consider for patch ensemble', type=int, default=10000)
    parser.add_argument('-rp', '--random_patches', help='Flag to use random patches instead of dense grid',action='store_true')
    parser.add_argument('-si', '--start_idx',
                        help='Start index for imagenet', default=0, type=int)
    parser.add_argument("--batch", type=int, default=1000, help="batch size")
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=100000,
                        help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="failure probability")
    parser.add_argument(
        '-sigma', '--sigma', help='Sigma for smoothing noise', default=0.1, type=float)
    parser.add_argument(
        '-patch', '--patch', help='use patche-wise ensembling model (default smoothing without patches).',
        action='store_true')

    parser.add_argument('--reduction_mode', '-rm', type=str,
                        default='mean', choices=['mean', 'max', 'min'])
    parser.add_argument('--normalize', action='store_true', help='True if you want to use NormalizeLayer, False if InputCenterLayer. Note: imagenet32 / - -> NormalizeLayer \n cifar10/finetune_cifar_from_imagenetPGD2steps / - -> NormalizeLayer \n cifar10/self_training / - -> NormalizeLayer \n  imagenet/- -> InputCenterLayer \n cifar10/"everythingelse" / - -> InputCenterLayer ')
    parser.add_argument('-ns', '--new_size', type=int, default=224)
    return parser


# def build_model(args, smooth=True, patchify=True, pretrained=True):
#     base_model = create_model(args.mtype, pretrained=pretrained)
#     config = resolve_data_config({}, model=base_model)
#     # Hardocoded to ensure additional patches for now.
#     config['input_size'] = (3, 256, 256)
#     preprocess = PreprocessLayer(config)
#     if patchify:
#         print('patchify')
#         # print('args', args.patch_size, args.patch_stride)
#         base_model = PatchModel(base_model, num_patches=args.num_patches,
#                                 patch_size=args.patch_size, patch_stride=args.patch_stride)
#     if smooth:
#         model = Smooth(nn.Sequential(preprocess, base_model), num_classes=1000,
#                        sigma=args.sigma)  # num classes hardocded for imagenet
#     else:
#         model = base_model
#     return model


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'config:{str(args)}')

    outdir = Path(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load data
    if args.dataset == 'imagenet':
        indices = np.load('imagenet_indices.npy')
        imagenet_val = Subset(
            ImageNet(root=args.dpath, split='val', transform=Compose([Resize((args.new_size, args.new_size)), ToTensor()])), indices)
        test_dl = DataLoader(imagenet_val, batch_size=1)
    else:
        test_dl = DataLoader(CIFAR10(root=args.dpath, train=False, download=True, transform=Compose(
            [Resize((args.new_size, args.new_size)), ToTensor()])), shuffle=False, batch_size=1)
    # Load model

    # smooth_model = build_model(
    #    args, smooth=True, patchify=args.patch, pretrained=True)
    model_data = torch.load(args.model_path)
    base_model = get_architecture(
        model_data['arch'], dataset=args.dataset, normalize=args.normalize)
    base_model.load_state_dict(model_data['state_dict'])
    args.num_classes = 10 if args.dataset == 'cifar10' or args.dataset == 'imagenette' else 1000
    smooth_model = PatchSmooth(base_model, num_patches=args.num_patches, patch_size=args.patch_size, patch_stride=args.patch_stride,
                               reduction=args.reduction_mode, num_classes=args.num_classes, sigma=args.sigma, random_patches=args.random_patches)
    smooth_model.base_classifier.eval()
    smooth_model.base_classifier.to(device)
    outfile = open(
        outdir / f'output_{args.mtype}_{args.sigma}_{args.patch_size}_{args.patch_stride}_{args.reduction_mode}.csv', 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=outfile, flush=True)

    for idx, (x, y) in enumerate(test_dl):
        print(idx, flush=True)
        if idx < args.start_idx:
            continue
        if idx >= args.start_idx + args.num_images:
            break
        x = x.to(device)
        y = y.to(device)
       # print(x.shape)
        # Certify
        tic = perf_counter()
        prediction, radius = smooth_model.certify(
            x, args.N0, args.N, args.alpha, args.batch)
        toc = perf_counter()
        correct = int(prediction == y)
        time_elapsed = str(datetime.timedelta(seconds=(toc - tic)))
        print(f'{idx}\t{y.item()}\t{prediction}\t{radius}\t{correct}\t{time_elapsed}',
              file=outfile, flush=True)

    outfile.close()
