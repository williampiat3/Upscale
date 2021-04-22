

import torch
import torch.nn as nn




def laplacian(img):
    return img[...,2:,1:-1] + img[...,:-2,1:-1]+ img[...,1:-1,2:] + img[...,1:-1,:-2]-4*img[...,1:-1,1:-1]


def laplacian_loss(image1,image2,insist=1.):
    loss_fn = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='none', beta=1.0)
    return loss_fn(insist*laplacian(image2),laplacian(image1))