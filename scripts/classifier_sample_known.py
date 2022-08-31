"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850)
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

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.dataset=='brats':
      ds = BRATSDataset(args.data_dir, test_flag=True)
      datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    elif args.dataset=='chexpert':
     data = load_data(
         data_dir=args.data_dir,
         batch_size=args.batch_size,
         image_size=args.image_size,
         class_cond=True,
     )
     datal = iter(data)
   
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

    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)


    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale



    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    for img in datal:

        model_kwargs = {}
     #   img = next(data)  # should return an image from the dataloader "data"
        print('img', img[0].shape, img[1])
        if args.dataset=='brats':
          Labelmask = th.where(img[3] > 0, 1, 0)
          number=img[4][0]
          if img[2]==0:
              continue    #take only diseased images as input
              
          viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input 0"))
          viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input 1"))
          viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input 2"))
          viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input 3"))
          viz.image(visualize(img[3][0, ...]), opts=dict(caption="ground truth"))
        else:
          viz.image(visualize(img[0][0, ...]), opts=dict(caption="img input"))
          print('img1', img[1])
          number=img[1]["path"]
          print('number', number)

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
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level
        )
        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()


        print('time for 1000', start.elapsed_time(end))

        if args.dataset=='brats':
          viz.image(visualize(sample[0,0, ...]), opts=dict(caption="sampled output0"))
          viz.image(visualize(sample[0,1, ...]), opts=dict(caption="sampled output1"))
          viz.image(visualize(sample[0,2, ...]), opts=dict(caption="sampled output2"))
          viz.image(visualize(sample[0,3, ...]), opts=dict(caption="sampled output3"))
          difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
          viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))
          
        elif args.dataset=='chexpert':
          viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled output"+str(name)))
          diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
          diff=np.array(diff.cpu())
          viz.heatmap(np.flipud(diff), opts=dict(caption="diff"))


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

