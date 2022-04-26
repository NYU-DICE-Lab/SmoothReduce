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
from torchvision.transforms.transforms import ToTensor

from smoothadv.core import Smooth
from smoothadv.patch_model import PreprocessLayer
from smoothadv.patch_model import PatchModel


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='results/')
    #parser.add_argument('-m', '--model', help='Model path')
    parser.add_argument('-mt', '--mtype', help='Model type', choices=timm.list_models(pretrained=True))
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
    parser.add_argument('-np', '--num_patches',help='Maximum number of patches to consider for patch ensemble', type=int, default=10000)
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
    return parser


def build_model(args, smooth=True, patchify=True, pretrained=True):
    base_model = create_model(args.mtype, pretrained=pretrained)
    config = resolve_data_config({}, model=base_model)
    config['input_size'] = (3,256 ,256) # Hardocoded to ensure additional patches for now.
    preprocess = PreprocessLayer(config)
    if patchify:
        print('patchify')
        # print('args', args.patch_size, args.patch_stride)
        base_model = PatchModel(base_model, num_patches=args.num_patches, patch_size = args.patch_size, patch_stride=args.patch_stride)
    if smooth:
        model = Smooth(nn.Sequential(preprocess, base_model), num_classes=1000,
                       sigma=args.sigma)  # num classes hardocded for imagenet
    else:
        model = base_model
    return model


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
    indices = np.load('imagenet_indices.npy')
    imagenet_val = Subset(
        ImageNet(root=args.dpath, split='val', transform=ToTensor()), indices)
    test_dl = DataLoader(imagenet_val, batch_size=1)
    # Load model

    smooth_model = build_model(
        args, smooth=True, patchify=args.patch, pretrained=True)
    smooth_model.base_classifier.eval()
    smooth_model.base_classifier.to(device)

    outfile = open(
        outdir / f'output_{args.mtype}_{args.sigma}_{args.patch_size}_{args.patch_stride}.csv', 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=outfile, flush=True)

    for idx, (x, y) in enumerate(test_dl):
        print(idx, flush=True)
        if idx < args.start_idx:
            continue
        if idx >= args.start_idx + args.num_images:
            break
        x = x.to(device)
        y = y.to(device)
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
