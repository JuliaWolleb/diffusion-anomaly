"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from visdom import Visdom
viz = Visdom(port=8851)
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=1,
    #     image_size=args.image_size,
    #     class_cond=True,
    # )
    ds = BRATSDataset(args.data_dir, test_flag=False)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data=iter(datal)
    img = next(data)  #should return an image from the dataloader "data"
    while img[2] == 0:
        img = next(data)
    print('label', img[2])
    if img[2] == 0:
        viz.image(visualize(img[0][0,0, ...]), opts=dict(caption="img input gesund"))
    elif img[2] == 1:
        viz.image(visualize(img[0][0, 0,...]), opts=dict(caption="img input krank0"))
        viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input krank1"))
        viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input krank2"))
        viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input krank3"))
        viz.image(visualize(img[0][0, 4, ...]), opts=dict(caption="img input krank4"))
        viz.image(visualize(img[0][0, 5, ...]), opts=dict(caption="img input krank5"))
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
        dist_util.load_state_dict(args.classifier_path)
    )
 #   classifier.load_state_dict(th.load(args.classifier_path))
    print('loaded classifier')



    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


    def cond_fn(x, t, org=None, y=None):
        assert y is not None
        with th.enable_grad():
            x=x[:,:4,...]
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale


    def cond_fn_mse(x, t, org, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            Loss_cls=selected.sum()
         #   z=th.zeros(x_in.shape).to(dist_util.dev())
            #loss=th.nn.functional.mse_loss(x_in, org, size_average=None, reduce=None, reduction='mean')

            d=-x_in+org

          #  print('L', L.max(), L.min())
            #loss_diff=L.sum()

           # loss=th.nn.functional.l1_loss(x_in, org, size_average=None, reduce=None, reduction='mean')


            grad=th.autograd.grad(Loss_cls, x_in)[0]
            grad=5000*grad


            mask = abs(grad) > 0.01 * abs(d)
            mask=mask*1
           # viz.image(visualize(mask[0, 0, ...]))


            diffd=(d.max()-d.min()).cpu()
            diffgrad=(grad.max()-grad.min()).cpu()
            viz.line(X=th.ones((1, 1)).cpu() * t.cpu(), Y=th.Tensor([grad.min()]).unsqueeze(0).cpu(),
                     win=grad_window, name='amplitude gradmin',
                     update='append')
            viz.line(X=th.ones((1, 1)).cpu() * t.cpu(), Y=th.Tensor([grad.max()]).unsqueeze(0).cpu(),
                     win=grad_window, name='amplitude gradmax',
                     update='append')
            viz.line(X=th.ones((1, 1)).cpu() * t.cpu(), Y=th.Tensor([d.max()]).unsqueeze(0).cpu(),
                     win=grad_window, name='dmax',
                     update='append')
            viz.line(X=th.ones((1, 1)).cpu() * t.cpu(), Y=th.Tensor([d.min()]).unsqueeze(0).cpu(),
                     win=grad_window, name='dmin',
                     update='append')
            if t>100:
                #a=grad+10*d*(1-mask)

               a=grad
            else:
                a=100*d+grad
           # viz.image(visualize(a[0,0,...]))

            return a,a * args.classifier_scale

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
            print('y', model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        print('samplefn', sample_fn)
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=True,
            device=dist_util.dev(),
            cyclic=True,
        )


        viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled output"))
        difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
        viz.heatmap(visualize(abs(org[0,0, ...]-sample[0, 0,...])), opts=dict(caption="diff 0"))
        viz.heatmap(visualize(abs(org[0, 1, ...] - sample[0, 1, ...])), opts=dict(caption="diff 1"))
        viz.heatmap(visualize(abs(org[0, 2, ...] - sample[0, 2, ...])), opts=dict(caption="diff 2"))
        viz.heatmap(visualize(abs(org[0, 3, ...] - sample[0, 3, ...])), opts=dict(caption="diff 3"))
        viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        print('sample', sample.shape, classes, 'classtarget')
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
        classifier_scale=1000,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
