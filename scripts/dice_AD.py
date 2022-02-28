import argparse
import os
import medpy
from visdom import Visdom
viz = Visdom(port=8850)
import sys
from sklearn.metrics import roc_auc_score
#from metrics import dc, jc, hd95, hd1, getHausdorff
from scipy.spatial.distance import directed_hausdorff
sys.path.append("..")
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
import nibabel as nib
from skimage.filters import threshold_otsu

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def hd(u,v):
    a=directed_hausdorff(u, v)[0]
    b=directed_hausdorff(u, v)[0]
    c=max(a,b)
    print('abc', a,b,c)
    return hd95


x=[5,10,20,50,100,250,350,500,750,1000,1500]
y1=[0.648, 0.684,0.695,0.702, 0.684,0.650,0.624,0.603,0.565,0.541,0.503]
y2=[0.980,0.984,0.987,0.987,0.986, 0.985, 0.984, 0.983, 0.981, 0.979, 0.976 ]

x_250=[5,10,20, 50, 100,250,500]
y1_250=[0.279,0.339,0.475, 0.61, 0.63, 0.658,0.641]
y2_250=[0.934, 0.952, 0.967, 0.979, 0.982, 0.983,0.983]

x_750=[5,10,20, 50, 100,250,500]
y1_750=[0.422,0.393,0.319, 0.253, 0.23, 0.201,0.204]
y2_750=[0.963, 0.959, 0.943, 0.928, 0.920, 0.917,0.918]


import matplotlib.pyplot as plt
plt.plot(x, y1,'bo-', label='L=500')
plt.plot(x_250, y1_250,'ro-', label='L=250')
plt.plot(x_750, y1_750,'ko-', label='L=750')
plt.title('Dice')
plt.xlabel('gradient scale')
plt.ylabel('avg Dice score')
plt.hlines(0.693, 0, 1500, colors=None, linestyles='dashed', label='FixedpointGAN')
plt.legend()
viz.matplot(plt)
plt.clf()
plt.plot(x, y2,'bo-', label='L=500')
plt.plot(x_250, y2_250,'ro-', label='L=250')
plt.plot(x_750, y2_750,'ko-', label='L=750')
plt.xlabel('gradient scale')
plt.ylabel('avg AUROC score')
plt.title('AUROC')
plt.hlines(0.965, 0, 1500, colors=None, linestyles='dashed', label='FixedpointGAN')
plt.legend()
viz.matplot(plt)

sys.exit('rer')

#PathDicomstripped = "./scaling20_ddim1000_t500/"
PathDicomstripped = "./results_L750/scaling500_ddim1000_t750/"

k=0
tot=0
totj=0
tothd=0
b=0
f=0
B=0
total_auc=0
i=0
for dirName, subdirList, fileList in os.walk(PathDicomstripped, topdown=True): #used to be dicomstripped
     s = dirName.split("/", -1)
     print('s', s[-1], s[-2])
#     if 't1n_3d' in subdirList:
#         path=os.path.join(dirName, 't1n_3d'))
     for filename in fileList:
            s = filename.split("_", 1)
            number=s[0]
            i+=1
            path = os.path.join(dirName, filename)
            sample=th.load(path)
#            if i%100==0:
#              viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="samplepred0"+str(number)))
#              viz.image(visualize(sample[0, 1, ...]), opts=dict(caption="samplepred1"+str(number)))
#              viz.image(visualize(sample[0, 2, ...]), opts=dict(caption="samplepred2"+str(number)))
#              viz.image(visualize(sample[0, 3, ...]), opts=dict(caption="samplepred3"+str(number)))
#              viz.heatmap(visualize(sample[0, 4, ...]), opts=dict(caption="diffpred", colormap='Jet'))


            image = np.array(sample[:,-1,...].cpu())
           # print('image', image.max(), image.min())
            thresh = threshold_otsu(image)
           # print('thresh', thresh)
            mask = th.where(th.tensor(image) > thresh, 1, 0)

          #  viz.image(visualize(mask[ 0,...]), opts=dict(caption="mask"))


            path2 = './Bratssliced/test_labels/' + str(
                number) + '-label.nii.gz'


            gt=nib.load(path2)
            Label = th.from_numpy(np.asarray(gt.dataobj).astype(dtype='float32'))
            iz = th.zeros(1, 256, 256)
            iz[:, 8:-8, 8:-8] = Label
            Labelmask = th.where(iz > 0, 1, 0)


            s=(Labelmask*1).sum()

            pixel_wise_cls = visualize(np.array(th.tensor(image).view(1, -1))[0, :])
            pixel_wise_gt = visualize(np.array(th.tensor(Labelmask).view(1, -1))[0, :])



            if (Labelmask*1).sum()>20:

                DSC=dice_score(mask.cpu(), Labelmask.cpu())
               
                tot += DSC
                k += 1
                auc = roc_auc_score(pixel_wise_gt, pixel_wise_cls)
                
                total_auc += auc
            else:
                f+=1

#print('Better', b, 'betrerdice', B/b)
print('k', k, 'f', f, 'tot', tot)
print('mean dice', tot/(k+f))
print('good dice', tot/(k))
print('mean auc',  total_auc/k)
#print('hd95', tothd/(k))
#
#print('mean jc', totj/(k+f))
#print('good jc', totj/(k))
sys.exit('rer')
 #s=1000; mean dice=0.53, auroc 0.979
 #s=500; mean dice=0.619, auroc 0.984
 #s=1500; mean dice =0.51, auroc 0.977
# s=250:
#s=750:
#s=5:
