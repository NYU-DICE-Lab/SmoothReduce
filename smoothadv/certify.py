# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

from architectures import get_architecture
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import torch
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
import numpy as np


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    print(checkpoint['arch'], args.dataset)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    print(base_classifier)
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    #dataset = get_dataset(args.dataset, args.split)
    indices = np.load('./imagenet_indices.npy')
    #dpath = '/data/datasets/ImageNet/val/'
    dpath = '/scratch/aaj458/data/ImageNet/val/'
    dataset = Subset(
         ImageNet(root=dpath, split='val', transform=transforms.Compose([transforms.Resize(256),
             transforms.CenterCrop(224),transforms.ToTensor()])), indices)
    test_dl = DataLoader(dataset, batch_size=1)

    for i, (x, label) in enumerate(test_dl):
    #for i in range(args.max):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        # print(x.min(), x.max())
        #(x, label) = dataset[i]
        print(x.min(), x.max(), label)
        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label.item(), prediction, radius, correct, time_elapsed), file=f, flush=True)
        #print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        #    i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
