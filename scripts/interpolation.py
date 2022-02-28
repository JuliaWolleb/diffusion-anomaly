"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from PIL import Image
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import scipy.fftpack
import scipy.signal as  signal
from scipy import ndimage

sys.path.append("..")
sys.path.append(".")
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data,center_crop_arr
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
blank = np.ones((256, 256))
image_window = viz.image(blank)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def load_image(path):
        numpy_img = np.load(path)
        print('img', numpy_img.shape)
        pil_image = Image.fromarray((numpy_img[...,0] * 255).astype(np.uint8)).convert('RGB')


        arr = center_crop_arr(pil_image, 256)

        arr = arr.astype(np.float32) / 127.5 - 1
        print('arr', arr.shape)
        viz.image(visualize(arr[:, :,0]), win=image_window, opts=dict(caption="input"))
        out_dict = {}
        return np.transpose(arr, [2, 0, 1])[None,...], out_dict#np.transpose(arr, [2, 0, 1]), out_dict

def butterLow(cutoff, critical, order):
    normal_cutoff = float(cutoff) / critical
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butterFilter(data, cutoff_freq, nyq_freq, order):
    b, a = butterLow(cutoff_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)
    return y

# x,_=load_image('./chexpert/train/g/healthy/patient00567_study1.npy')
#
# lowpass = ndimage.gaussian_filter(x, 3)
# gauss_highpass = x - lowpass
#
#
# viz.image(visualize(x[0, ...]), opts=dict(caption="img input1"))
# viz.image(visualize(lowpass[0, ...]), opts=dict(caption="lowpassfiltered"))
# viz.image(visualize(gauss_highpass[0, ...]), opts=dict(caption="gauss_highpass"))

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    steps=[0,0.2,0.4,0.6,0.8,1]
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
    #img = next(data)  #should return an image from the dataloader "data"
    img1,_=load_image('./chexpert/train/g/healthy/patient00567_study1.npy')
    img2,_ = load_image('./chexpert/train/k/ill_effusion/patient00136_study1.npy')
    print('img12', img1.shape, img2.shape)
    viz.image(visualize(img1[0, ...]), opts=dict(caption="img input1"))
    viz.image(visualize(img2[0, ...]), opts=dict(caption="img input2"))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    for lambdaint in steps:
     all_images = []
     all_labels = []
     print('lambda', lambdaint)
     while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop_interpolation if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, x_noisy, org1, org2 = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size), img1, img2,lambdaint,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled lambda " + str(lambdaint)))
      #  viz.image(visualize(x_noisy[0, ...]), opts=dict(caption="xnoisy lambda0.5"))
        #viz.image(visualize(org1[0, ...]), opts=dict(caption="org1"))
       # viz.image(visualize(org2[0, ...]), opts=dict(caption="org2"))

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
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
