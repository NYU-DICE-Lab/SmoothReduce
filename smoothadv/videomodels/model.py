from __future__ import division
import torch
from torch import nn
from smoothadv.videomodels import resnext
import pdb
from smoothadv.videodataset.preprocess_data import *

def generate_model( opt):
    assert opt.model in ['resnext']
    assert opt.model_depth in [101]

    from smoothadv.videomodels.resnext import get_fine_tuning_parameters
    model = resnext.resnet101(
            num_classes=opt.num_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers)
    

    #model = model.cuda()
    model = nn.DataParallel(model)
    
    # if opt.pretrain_path:
    #     print('loading pretrained model {}'.format(opt.pretrain_path))
    #     pretrain = torch.load(opt.pretrain_path)
        
    #     assert opt.arch == pretrain['arch']
    #     model.load_state_dict(pretrain['state_dict'])
    #     model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
    #     model.module.fc = model.module.fc.cuda()

    #     parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    #     return model, parameters

    return model, model.parameters()

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.sds = torch.tensor(sds)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, sample_duration, height, width) = input.shape
        self.means = self.means.to(input.device)
        #print('in normalize',input.shape, self.means, self.sds)
        self.sds = self.sds.to(input.device)
        means = self.means.repeat((batch_size, sample_duration, height, width, 1)).permute(0, 4, 1, 2, 3)
        sds = self.sds.repeat((batch_size, sample_duration, height, width, 1)).permute(0, 4, 1, 2, 3)
        return (input - means) / sds

def model_wrapper(model, opt):
    if opt.normalize_layer:
        if opt.dataset == "UCF101":
            print('using ucf101 test normalization')
            means = get_mean(opt.dataset)
            sds = get_std(opt.dataset)
            model = nn.Sequential(NormalizeLayer(means=means, sds=sds), model)
            #print(model)
            #print(means, sds)
    return model