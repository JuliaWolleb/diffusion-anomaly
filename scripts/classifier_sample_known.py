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
    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
   # data = iter(datal)
      #  viz.image(visualize(img[0][0, 5, ...]), opts=dict(caption="img input krank5"))
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


    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x=x[:,:4,...]
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
    # print(img.shape)
    #         print(label.shape)
    #         break
  #  while len(all_images) * args.batch_size < args.num_samples:
   #
        model_kwargs = {}

     #   img = next(data)  # should return an image from the dataloader "data"
        Labelmask = th.where(img[3] > 0, 1, 0)
        
        if (Labelmask * 1).sum() < 20:
            continue
        print('label', img[2], (Labelmask * 1).sum())
        number=img[4][0]
        print('number22', number)
       # if number!='003487':
       #   continue


#        if img[2] == 0:
#            viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input gesund0"))
#            viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input gesund1"))
#            viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input gesund2"))
#            viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input gesund3"))
#        elif img[2] == 1:
#            viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input krank0"))
#            viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input krank1"))
#            viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input krank2"))
#            viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input krank3"))
#            viz.image(visualize(img[3][0, ...]), opts=dict(caption="ground truth"))
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
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )


        viz.image(visualize(sample[0,0, ...]), opts=dict(caption="sampled output0"))
        viz.image(visualize(sample[0,1, ...]), opts=dict(caption="sampled output1"))
        viz.image(visualize(sample[0,2, ...]), opts=dict(caption="sampled output2"))
        viz.image(visualize(sample[0,3, ...]), opts=dict(caption="sampled output3"))
        
        difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
        viz.image(visualize(difftot), opts=dict(caption="diff"))
        plt.imshow(difftot.cpu(), cmap='jet')                 
        plt.xticks([])
        plt.yticks([])
        viz.matplot(plt)
#        viz.heatmap(visualize(abs(org[0,0, ...]-sample[0, 0,...])), opts=dict(caption="diff 0"))
#        viz.heatmap(visualize(abs(org[0, 1, ...] - sample[0, 1, ...])), opts=dict(caption="diff 1"))
#        viz.heatmap(visualize(abs(org[0, 2, ...] - sample[0, 2, ...])), opts=dict(caption="diff 2"))
#        viz.heatmap(visualize(abs(org[0, 3, ...] - sample[0, 3, ...])), opts=dict(caption="diff 3"))
#        viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
       # sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        print('sample', sample.shape, classes, 'classtarget', 'difftot', difftot.shape)
        s=th.tensor(sample)
        output=th.cat((sample, difftot[None,None,...]), dim=1)
        print('output', output.shape)
        th.save(output, './results_L250/scaling750_ddim1000_t250/' + str(number) + '_output')

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
        classifier_scale=10000,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

