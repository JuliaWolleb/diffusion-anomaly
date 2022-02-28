"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys
from torch.autograd import Variable
sys.path.append("..")
sys.path.append(".")
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from visdom import Visdom
import numpy as np
viz = Visdom(port=8850)
loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='classification loss'))
val_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='validation loss'))
acc_window= viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='acc', title='accuracy'))

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    create_classifier,
    classifier_defaults
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict




def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()


    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())

    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()



    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=1,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None
    total=0; correct=0
    for i, data in enumerate(val_data):

                         x_real = data[0].to(dist_util.dev())
                         label_org2 = (data[1]["y"]).to(dist_util.dev())
                         t = th.zeros(1, dtype=th.long, device=dist_util.dev())
                         logits= classifier(x_real, timesteps=t)
                         print('logits1', logits)

    #
                         _, predicted = th.max(logits.data, 1);
                         print('pred', predicted, label_org2)
                         total += 1
                         correct += (predicted.cpu() == label_org2.cpu()).sum().item()
                         if i==40:
                             break
                         #long_pred = torch.cat((long_pred, predicted.cpu()), dim=0)

    print('tot', total, 'corr', correct)



def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="./fake_images_classcond/",#"./fake_images_classcond/","./chexpert/validate"
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="./results/model0007000class.pt",
        classifier_path="./results/model0007000class.pt",
        log_interval=1,
        eval_interval=100,
        save_interval=1000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
