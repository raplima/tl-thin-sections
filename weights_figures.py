"""Script to generate figures for the weights of the trained models
Reads a trained model and creates the image of the weights for the first layer

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from torchvision import models
from torchvision.utils import make_grid


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return fig


dir_out = "../data/trained_models/weights_figures/"
dir_out = "../data/trained_models/all_weights_figures/"
dir_out = 'C:/Projects/drill-cuttings/data/trained_models'

load_weights_paths = [
    "../data/trained_models/Sycamore/resnet18_rawfoot_dbs64_lr1e-3.ckpt",
    "../data/trained_models/Sycamore/resnet18_derm_dbs64_lr5e-5.ckpt",
    "../data/trained_models/Sycamore/resnet18_histo_dbs64_lr1e-4.ckpt",
    "../data/trained_models/Sycamore/resnet18_imagenet_dbs64_lr1e-4_RMSprop.ckpt",
    "../data/trained_models/Sycamore/resnet18_randinit_dbs64_lr1e-4.ckpt",

    "../data/trained_models/Others/rawfoot_resnet18_randinit.ckpt",
    "../data/trained_models/Others/histo_resnet50_randinit_dbs1024.ckpt",
    "../data/trained_models/Others/derm_resnet50_randinit_dbs256.ckpt",

    "../data/trained_models/Sycamore/resnet18_rawfoot_dbs8_lr1e-3.ckpt",
    "../data/trained_models/Sycamore/resnet18_derm_dbs64_lr5e-5_RMSprop.ckpt",
    "../data/trained_models/Sycamore/resnet18_histo_dbs64_lr1e-4_RMSprop.ckpt"

]

load_weights_paths = [os.path.join('../data/trained_models/Sycamore/', f)
                      for f in os.listdir("../data/trained_models/Sycamore/") if f.endswith(".ckpt")]

load_weights_paths = [os.path.join('C:/Projects/drill-cuttings/data/trained_models', f)
                      for f in os.listdir('C:/Projects/drill-cuttings/data/trained_models') if f.endswith(".ckpt")]


for load_weights_path in load_weights_paths:
    print(load_weights_path)
    ckpt = torch.load(load_weights_path)
    conv1_weigths = ckpt['state_dict']['model.conv1.weight']
    conv1_grid = make_grid(conv1_weigths, normalize=True, padding=1)
    fig = show(conv1_grid)

    fig.savefig(os.path.join(dir_out,
                             f'{os.path.basename(load_weights_path)}.png'))

model = models.resnet18(pretrained=True)
conv1_weigths = model.conv1.weight
conv1_grid = make_grid(conv1_weigths, normalize=True, padding=1)
fig = show(conv1_grid)
fig.savefig(os.path.join(dir_out,
                         f'imagenet.pdf'))

conv1_weights = nn.Conv2d(3, 64, 7, bias=False).weight
conv1_grid = make_grid(conv1_weights, normalize=True, padding=1)
fig = show(conv1_grid)
fig.savefig(os.path.join(dir_out,
                         f'random.pdf'))
