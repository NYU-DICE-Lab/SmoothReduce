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

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms.transforms import ToTensor, Resize, Compose

from smoothadv.core import Smooth
from smoothadv.patch_model import VideoEnsembleModel, VideoPatchSmooth, BaseVideoRandomizedSmooth
from smoothadv.videomodels.model import model_wrapper, generate_model
from smoothadv.videodataset.dataset import UCF101_test
from smoothadv.video_opts import parse_opts


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=[
                        'UCF101'])
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='results/')
    parser.add_argument('-mp', '--model_path',
                        help='Model path to load model', type=str, required=True)
    parser.add_argument('-mt', '--mtype', help='Model type')
    parser.add_argument('-dpath', help='Data path',
                        default='/data/datasets/Imagenet/val')
    parser.add_argument('--gpu', help='gpu to use', default='0', type=str)
    parser.add_argument(
        '-it', '--iter', help='No. of iterations', type=int, default=40)
    parser.add_argument('-ni', '--num_images',
                        help='Number of images to be tested', default=100, type=int)
    parser.add_argument('-cs', '--chunk_size',
                        help='Chunk size', default=16, type=int)
    parser.add_argument('-cstr', '--chunk_stride',
                        help='chunk stride', default=16, type=int)
    parser.add_argument('-ns', '--num_subvideos',
                        help='Maximum number of subvideos to consider for patch ensemble', type=int, default=10000)
    parser.add_argument('-rs', '--random_subvideos', help='Flag to use random subvideos instead of dense grid',action='store_true')
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
    parser.add_argument('--subvideo_size','-svs', help='Subvideo size', default=64, type=int)
    parser.add_argument('--subvideo_stride','-svstr', help='Subvideo stride', default=16, type=int)
    parser.add_argument('--reduction_mode', '-rm', type=str,
                        default='mean', choices=['mean', 'max', 'min'])
   # parser.add_argument('--normalize', action='store_true', help='True if you want to use NormalizeLayer, False if InputCenterLayer. Note: imagenet32 / - -> NormalizeLayer \n cifar10/finetune_cifar_from_imagenetPGD2steps / - -> NormalizeLayer \n cifar10/self_training / - -> NormalizeLayer \n  imagenet/- -> InputCenterLayer \n cifar10/"everythingelse" / - -> InputCenterLayer ')
   # parser.add_argument('-ns', '--new_size', type=int, default=224)
    parser.add_argument(
        '--frame_dir',
        default='dataset/HMDB51/',
        type=str,
        help='path of jpg files')
    parser.add_argument(
        '--annotation_path',
        default='dataset/HMDB51_labels',
        type=str,
        help='label paths')
    parser.add_argument(
        '--split',
        default=1,
        type=str,
        help='(for HMDB51 and UCF101)')
    parser.add_argument(
        '--modality',
        default='RGB',
        type=str,
        help='(RGB, Flow)')
    parser.add_argument(
        '--input_channels',
        default=3,
        type=int,
        help='(3, 2)')
    parser.add_argument(
        '--n_finetune_classes',
        default=51,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument(
        '--only_RGB', 
        action='store_true', 
        help='Extracted only RGB frames')
    parser.set_defaults(only_RGB = False)
    
    
    # Model parameters
    parser.add_argument(
        '--output_layers',
        action='append',
        help='layer to output on forward pass')
    parser.set_defaults(output_layers=[])
    parser.add_argument(
        '--model',
        default='resnext',
        type=str,
        help='Model base architecture')
    parser.add_argument(
        '--model_depth',
        default=101,
        type=int,
        help='Number of layers in model')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--freeze_BN', 
        action='store_true', 
        help='freeze_BN/testing')
    parser.set_defaults(freeze_BN=False)
    parser.add_argument(
        '--batch_size', 
        default=20, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_workers', 
        default=4, 
        type=int, 
        help='Number of workers for dataloader')
    parser.add_argument(
        '--normalize_layer',
        default=1,
        type=int,
        help='use normalize layer')

    parser.add_sargument('--basers', action='store_true', help='True if you want to use base Randomized smoothing, False if not')
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    #args_video = parse_opts()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'config:{str(args)}')

    outdir = Path(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load data
    if args.dataset == 'UCF101':
        ds = UCF101_test(train=0, opt=args, split=1 )
    else:
        print('Unsupported dataset!')
    test_dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.n_workers)

    args.num_classes = 101 
    # for the generate_model function here, we use chunk size to create the model rather than sample duration.
    # Unlike the original script, we use sample duration to mean 
    model,_ = generate_model(args)
    model = model_wrapper(model, args)
    model_dict = torch.load(args.model_path)
    sd = model_dict['state_dict']
    #print(sd.keys())
    model.load_state_dict(sd)
    #model.to(device)
    model.eval()
    video_submodel = VideoEnsembleModel(model, args.chunk_size, args.chunk_stride)
    video_submodel.eval()
    #Create smooth model
    if args.basers:
        smooth_model = BaseVideoRandomizedSmooth(video_submodel, args.num_classes, args.sigma)
    else:
        smooth_model = VideoPatchSmooth(video_submodel, args.num_subvideos, args.subvideo_size, args.subvideo_stride, args.reduction_mode, args.num_classes, args.sigma, args.random_subvideos)    
    
    smooth_model.base_classifier.eval()
    smooth_model.base_classifier.to(device)
    outfile = open(
         outdir / f'output_{model_dict["arch"]}_{args.sigma}_{args.chunk_size}_{args.chunk_stride}_{args.reduction_mode}.csv', 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=outfile, flush=True)

    for idx, (x, y) in enumerate(test_dl):
        #print(x.shape)
        #x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3]).to(device)
        #x = x.permute(0, 2, 1, 3, 4).contiguous().to(device)
        #print('input', x.shape, y.shape)
        print(idx, flush=True)
        if idx < args.start_idx:
            continue
        if idx >= args.start_idx + args.num_images:
            break
        x = x.to(device)
        y = y.to(device)
        #print('input', x[:,:,:16,...].shape)
        #print(smooth_model.base_classifier(x[:,:,:16,...]))
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
