"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from skimage import exposure
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import torch.nn.functional as F
sys.path.append("..")
sys.path.append(".")
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from interpolation import load_image
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    classifier_defaults,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    cond_img, _ = load_image('./chexpert/train/g/healthy/patient00567_study1.npy')
    cond_img=th.tensor(cond_img).to(dist_util.dev())
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    data = load_data(
        data_dir=args.data_dir,
        batch_size=1,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    img = next(data)  #should return an image from the dataloader "data"
    print('img0', img[0].shape)
    viz.image(visualize(img[0][0,0, ...]), opts=dict(caption="img input"))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    classifier2 = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier2.load_state_dict(
        dist_util.load_state_dict(args.classifier_path2, map_location="cpu")
    )
    classifier2.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier2.convert_to_fp16()
    classifier2.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            print(log_probs.shape, 'log_probs')
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            print('grad',a.shape)
            return a, a * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, x_noisy, org = sample_fn(
            model_fn,
            #model,
            (args.batch_size, 3, args.image_size, args.image_size), img,
            clip_denoised=args.clip_denoised,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            model_kwargs=model_kwargs, conditioning=True, conditioner=th.tensor(img[0]).to(dist_util.dev()), classifier=classifier2)
          #  model_kwargs=model_kwargs, conditioning=True, conditioner=cond_img

        viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled output"))
        viz.image(visualize(x_noisy[0, ...]), opts=dict(caption="xnoisy t=500"))
        #viz.image(visualize(condition_img[0, ...]), opts=dict(caption="condition"))
        diff = abs(org[0, ...] - sample[0, ...])
        diff = np.array(visualize(diff.cpu())) * 255
        p2 = np.percentile(diff, 0)
        p98 = np.percentile(diff, 98)
       # diff= exposure.rescale_intensity(diff, in_range=(p2, p98))
        viz.heatmap(np.flipud(diff[0, ...]), opts=dict(caption="diff", colormap='Jet'))
       # viz.image(visualize(abs(org[0, ...]-sample[0, ...])), opts=dict(caption="diff"))
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        print('sample', sample.shape)
        s=th.tensor(sample)
        th.save(s, './tensor.pt')

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        classifier_path2="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
