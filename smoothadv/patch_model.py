"""
Patchwise smoothing
"""

from math import ceil
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm, binom_test
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from torch.nn import functional as F
matplotlib.use('Agg')



class PatchModel(nn.Module):
    """
    Patchwise smoothing
    """

    def __init__(self, base_classifier, num_patches, patch_size, patch_stride=1, reduction='mean', num_classes=10):
        super().__init__()
        self.base_classifier = base_classifier
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.reduction = reduction
        self.num_patches = num_patches
        self.num_classes = num_classes

    def get_patches(self, x):
        # print('args2', self.patch_size, self.patch_stride)
        b, c, h, w = x.shape
        #print(b, c, h, w, self.patch_size, self.patch_stride)
        h2 = h//self.patch_stride
        w2 = w//self.patch_stride
        pad_row = (h2 - 1) * self.patch_stride + self.patch_size - h
        pad_col = (w2 - 1) * self.patch_stride + self.patch_size - w
        #print(pad_row, pad_col)
        #x = F.pad(x, (pad_row//2, pad_row - (pad_row//2), pad_col//2, pad_col - (pad_row//2)))
        #num_patches = (x.shape[2] // (self.patch_size - self.patch_stride)) * (x.shape[3] // (self.patch_size - self.patch_stride))
        #print(x.shape[2], x.shape[3])
        # get patches
        #print(x.shape, self.patch_stride, self.patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(
            3, self.patch_size, self.patch_stride)
        # print(patches.shape)
        _, _, px, py, _, _ = patches.shape
        gen_num_patches = px * py
        patches = patches.reshape(
            b, c, gen_num_patches, self.patch_size, self.patch_size)
        if gen_num_patches > self.num_patches:
            patches = patches[:, :, :self.num_patches, ...]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # t = patches[4,35,...].detach().cpu().numpy().transpose(1,2,0)
        # t = (t - t.min())/t.ptp()
        # plt.imsave('./test11.png', t)
        # print(patches.shape)
        return patches

    def forward(self, x):
        # print(x.shape)
        # t = x[4,...].detach().cpu().numpy().transpose(1,2,0)
        # t = (t - t.min())/t.ptp()
        # plt.imsave('./test.png', t)
        patches = self.get_patches(x)
        outputs = torch.zeros(
            (patches.shape[0], patches.shape[1], self.num_classes), dtype=x.dtype, device=x.device)
        # get the output of each patch
        for i in range(patches.shape[0]):
            outputs[i] = self.base_classifier(patches[i, ...])
        # print('outputs', outputs.shape)
        if self.reduction == 'mean':
            outputs = outputs.mean(dim=1)
        elif self.reduction == 'max':
            outputs = outputs.max(dim=1)[0]
        elif self.reduction == 'min':
            outputs = outputs.min(dim=1)[0]
        return outputs


class PreprocessLayer(nn.Module):
    """
    Apply transformations for base classifier.
    Supports mean, std, deviation, normalization,
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.preprocess_layer = self.transforms_imagenet_eval(
            **self.transforms)

    def transforms_imagenet_eval(
            self,
            input_size=224,
            crop_pct=None,
            interpolation='bilinear',
            use_prefetcher=False,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD):

        # Since we assume that we are using a patch model, we will disregard cropping and resizing. The arguments exist to
        # maintain compatibility with timm classifiers.
        if isinstance(input_size, (tuple, list)):
            img_size = input_size[-2:]
        else:
            img_size = input_size
        crop_pct = crop_pct or DEFAULT_CROP_PCT

        if isinstance(img_size, (tuple, list)):
            assert len(img_size) == 2
            if img_size[-1] == img_size[-2]:
                # fall-back to older behaviour so Resize scales to shortest edge if target is square
                scale_size = int(math.floor(img_size[0] / crop_pct))
            else:
                scale_size = tuple([int(x / crop_pct) for x in img_size])
        else:
            scale_size = int(math.floor(img_size / crop_pct))

        tfl = [
            transforms.Resize(scale_size, interpolation=0),
            transforms.CenterCrop(img_size),
        ]
        if use_prefetcher:
            # prefetcher and collate will handle tensor conversion and norm
            tfl += [ToNumpy()]
        else:
            tfl += [
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ]

        return nn.Sequential(*tfl)

    def forward(self, x):
        return self.preprocess_layer(x)


class PatchSmooth(nn.Module):
    """
    Smooth the output of a patch classifier. (Uncorrelated noise)
    """

    def __init__(self, base_classifier, num_patches, patch_size, patch_stride=1, reduction='mean', num_classes=10, sigma=0.12, random_patches=False):
        super().__init__()
        self.base_classifier = base_classifier
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.reduction = reduction
        self.num_classes = num_classes
        self.sigma = sigma
        self.random_patches = random_patches

    def get_patches(self, x):
        b, c, h, w = x.shape
        print(self.patch_stride)
        if not self.random_patches:
            h2 = h//self.patch_stride
            w2 = w//self.patch_stride
            # pad_row = (h2 -1) * self.patch_stride + self.patch_size - h
            # pad_col = (w2 -1) * self.patch_stride + self.patch_size - w
            patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(
                3, self.patch_size, self.patch_stride)
            _, _, px, py, _, _ = patches.shape
            gen_num_patches = px * py
            patches = patches.reshape(
                b, c, gen_num_patches, self.patch_size, self.patch_size)
            if gen_num_patches > self.num_patches:
                patches = patches[:, :, :self.num_patches, ...]
            patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        else:
            """
            Randomly sample patches from the image.
            """
            patches = torch.zeros(
                (b, self.num_patches, c, self.patch_size, self.patch_size), dtype=x.dtype, device=x.device)
            for i in range(b):
                for j in range(self.num_patches):
                    x_i = np.random.randint(0, h - self.patch_size)
                    y_i = np.random.randint(0, w - self.patch_size)
                    patches[i, j, ...] = x[i, ...][:, x_i:x_i +
                                                      self.patch_size, y_i:y_i + self.patch_size]
        return patches

    def forward(self, x):
        patches = self.get_patches(x)
        outputs = self.base_classifier(patches)
        if self.reduction == 'mean':
            outputs = outputs.mean(dim=1)
        elif self.reduction == 'max':
            outputs = outputs.max(dim=1)[0]
        elif self.reduction == 'min':
            outputs = outputs.min(dim=1)[0]
        return outputs

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """
        Certify the number of patches and the size of the patches.
        :param x: Input image
        :param n0: Initial number of patches
        :param n: Final number of patches
        :param alpha: Accuracy of the number of patches
        :param batch_size: Batch size
        :return: (n, sigma)
        """
        # Get patches
        patches = self.get_patches(x)
        #print(patches.shape)
        #import sys
        #sys.exit()
        # number of patches selected
        #counts_selection = np.zeros((patches.shape[1], self.num_classes))
        #probs = np.zeros(patches.shape[1])  # probability of each patch
        #cAhat_list = np.zeros(patches.shape[1])
        #for i in range(patches.shape[1]):
        tmp = self._sample_noise(patches[0], n0, batch_size)
        # print('tmp', tmp.shape)
        counts_selection = tmp
        cAhat = counts_selection.argmax().item()
        counts_estimation = self._sample_noise(patches[0], n, batch_size)
        nA = counts_estimation[cAhat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        #probs[i] = self._lower_confidence_bound(nA, n, alpha)
        #cAhat_list[i] = cAhat
        #print('cAhat_list', cAhat_list)
        #print('probs', probs)
        #values, counts = np.unique(cAhat_list, return_counts=True)
        #cAHat = values[np.argmax(counts)]
        #pA = []
        #for p, c in zip(probs, cAhat_list):
        #    if c == cAHat:
        #        pA.append(p)
        #pA = np.array(pA)
        #if self.reduction == 'mean':
        #    pABar = np.mean(pA)
        #elif self.reduction == 'max':
        #    pABar = np.max(pA)
        #elif self.reduction == 'min':
        #    pABar = np.min(pA)
        #print(cAHat, pABar)
        if pABar < 0.5:
            return -1, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAhat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> (int, float):
        """
        Predict using patches
        """
        raise Exception("Not implemented")
        # patches = self.get_patches(x)
        # pred = torch.zeros(patches.shape[0])
        # count1 = []
        # count2 = []

        # for i in range(patches.shape[0]):
        #     pred[i],  = self.predict(patches[i], n, alpha, batch_size)

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN, count1, count2
        else:
            return top2[0], count1, count2

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [patch x channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        # print(x.shape)
        # import sys
        # sys.exit()
        #print(x.shape)
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                #print(x.shape)
                batch = x.unsqueeze(1).repeat((1, this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                batch = batch + noise
                #print('in certify1',batch.shape)
                pn, c, ch, w, h = batch.shape

                batch = batch.view(pn*c, ch,  w,  h)
                #print('in certify2', batch.shape)
                predictions = F.softmax(self.base_classifier(batch), dim=-1)#.argmax(1)
                #print('in ceritfy3', predictions.shape)
                predictions = predictions.view(pn, c, -1)
                #print('in certify4', predictions.shape)
               
                #import sys
                #sys.exit()
                #pred_labels = predictions.argmax(1)
                if self.reduction == 'max':
                    pred_maxs = predictions.max(0)[0]
                elif self.reduction == 'mean':
                    pred_maxs = predictions.mean(0)
                #print('in certify5', pred_maxs.shape)
                predictions = pred_maxs.argmax(1)
               # print('in certify6',predictions.shape)
                #print(predictions.shape, pred_maxs.shape)
                #import sys
                #sys.exit()
                counts += self._count_arr(predictions.cpu().numpy(),
                                          self.num_classes)
            #print(counts)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


class VideoPatchSmooth(nn.Module):
    """
    Smooth the output of a video classifier where we use chunks as the input.
    Smooth classifier of the form \E_{z \sim N(0, \sigma^2 I)} [max_i f(x_i+z)]
    x_i is a chunk of the video.
    """

    def __init__(self, base_classifier, num_subvideos, subvideo_size, subvideo_stride, reduction='mean', num_classes=10, sigma=0.12, random_patches=False):
        super().__init__()
        self.base_classifier = base_classifier
        self.num_subvids = num_subvideos
        self.subvideo_size = subvideo_size
        self.subvideo_stride = subvideo_stride
        #self.chunk_size = chunk_size
        #self.chunk_stride = chunk_stride
        self.reduction = reduction
        self.num_classes = num_classes
        self.sigma = sigma
        self.random_patches = random_patches

    def get_subvideos(self, x):
        b, c, f, h, w = x.shape
        #print('x.shape', x.shape)
        #print(self.patch_stride)
        if not self.random_patches:
            subvids = x.unfold(2, self.subvideo_size, self.subvideo_stride).permute(0, 1, 5, 2, 3, 4).contiguous() 
            #b, num_chunks, chunksize, ch, frame_wd, frame_ht
            gen_num_subvideos = subvids.shape[1] #
            #patches = patches.reshape(
            #    b, c, gen_num_patches, self.patch_size, self.patch_size)
            if gen_num_subvideos > self.num_subvids:
                subvids = subvids[:, :self.num_subvids, ...]
        else:
            """
            Randomly sample patches from the image.
            """
            subvids = torch.zeros(
                (b, self.num_subvids, c, self.subvideo_size, h, w), dtype=x.dtype, device=x.device)
            #print('chunks_shape',chunks.shape)
            for i in range(b):
                for j in range(self.num_subvids):
                    f_i = np.random.randint(0, f - self.subvideo_size)
                    #print(chunks[i,j,...].shape, x[i, ...][:, f_i:f_i+self.chunk_size,...].shape)
                    subvids[i, j, ...] = x[i, ...][:, f_i:f_i + self.subvideo_size,...]
        return subvids

    def forward(self, x):
        patches = self.get_patches(x)
        outputs = self.base_classifier(patches)
        if self.reduction == 'mean':
            outputs = outputs.mean(dim=1)
        elif self.reduction == 'max':
            outputs = outputs.max(dim=1)[0]
        elif self.reduction == 'min':
            outputs = outputs.min(dim=1)[0]
        return outputs

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """
        Certify the number of patches and the size of the patches.
        :param x: Input video
        :param n0: Initial number of patches
        :param n: Final number of patches
        :param alpha: Accuracy of the number of patches
        :param batch_size: Batch size
        :return: (n, sigma)
        """
        # Get patches
        #print('input size', x.shape)
        chunks = self.get_subvideos(x)
        #print('chunk_size', chunks.shape)
        counts_selection = self._sample_noise(chunks[0], n0, batch_size)
        cAhat = counts_selection.argmax().item()
        counts_estimation = self._sample_noise(chunks[0], n, batch_size)
        nA = counts_estimation[cAhat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return -1, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAhat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> (int, float):
        """
        Predict using patches
        """
        raise Exception("Not implemented")


    # def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
    #     """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
    #     class returned by this method will equal g(x).

    #     This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
    #     for identifying the top category of a multinomial distribution.

    #     :param x: the input [channel x height x width]
    #     :param n: the number of Monte Carlo samples to use
    #     :param alpha: the failure probability
    #     :param batch_size: batch size to use when evaluating the base classifier
    #     :return: the predicted class, or ABSTAIN
    #     """
    #     self.base_classifier.eval()
    #     counts = self._sample_noise(x, n, batch_size)
    #     top2 = counts.argsort()[::-1][:2]
    #     count1 = counts[top2[0]]
    #     count2 = counts[top2[1]]
    #     if binom_test(count1, count1 + count2, p=0.5) > alpha:
    #         return Smooth.ABSTAIN, count1, count2
    #     else:
    #         return top2[0], count1, count2

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [chunks x frame x channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        #print('certify input',x.shape)
        #import sys
        #sys.exit()
        #print(x.shape)
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                #print(x.shape)
                batch = x.unsqueeze(1).repeat((1, this_batch_size, 1, 1, 1, 1))
                #print('certify batch',batch.shape)
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                batch = batch + noise
                pn, c, ch, f, w, h = batch.shape
                batch = batch.view(pn*c, ch, f,  w,  h)
                #print(batch.shape)
                predictions = F.softmax(self.base_classifier(batch), dim=-1)#.argmax(1)
                predictions = predictions.view(pn, c, -1)
                #print('in certify',predictions.shape)
               
                #import sys
                #sys.exit()
                #pred_labels = predictions.argmax(1)
                if self.reduction == 'max':
                    pred_maxs = predictions.max(0)[0]
                if self.reduction == 'mean':
                    pred_maxs = predictions.mean(0)
                #print('in certify2', pred_maxs.shape)
                predictions = pred_maxs.argmax(1)
                #print(predictions.shape)
                #print(predictions)
                #print(predictions.shape, pred_maxs.shape)
                #import sys
                #sys.exit()
                counts += self._count_arr(predictions.cpu().numpy(),
                                          self.num_classes)
            #print(counts)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


class VideoEnsembleModel(nn.Module):
    """
    Takes as input a subvideo (64 frames/128 frames)
    Outputs the average of logits predicted for each 16 frame chunk
    """

    def __init__(self, base_classifier, chunk_size=16, chunk_stride=16):
        super().__init__()
        self.base_classifier = base_classifier
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride

    def forward(self, x):
        #print('in ensemble',x.shape)
        #import sys
        #sys.exit()
        #print(x.shape)
        chunks = x.unfold(2, self.chunk_size, self.chunk_stride).permute(0, 2, 1, 5, 3, 4)
        #print('in ensembel chunk ',chunks.shape)
        bs, cnum, ch, f, w, h = chunks.shape
        chunks = chunks.reshape(bs*cnum, ch, f, w, h)
        #print('in ensembel chunk 2',chunks.shape)
        logits = self.base_classifier(chunks)
        #print('in ensembel logits',logits.shape)
        logits = logits.reshape(bs, cnum, -1)
        #print('in ensembel logits2 ',logits.shape)
        #logits = logits.view(chunks.size(0), self.chunk_size, -1)
        #print(logits.shape)
        logits = logits.mean(1)
        #print(logits.argmax(-1), logits.mean(1))
        #print(logits.shape)
        return logits


class BaseVideoRandomizedSmooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        #print('Smooth', pABar, cAHat)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        #print(x.shape)
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
