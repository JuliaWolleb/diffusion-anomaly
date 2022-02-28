import argparse
import os
import medpy
from visdom import Visdom
viz = Visdom(port=8850)
import sys
from sklearn.metrics import roc_auc_score
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


x=[0,250,500,750,1000,1500]
y1=[0,0,0.61,0,0.53,0.51]
import matplotlib.pyplot as plt
#plt.plot(x, y1,'bo-')
#viz.matplot(plt)


number='002464'
k=0
tot=0
totj=0
tothd=0
b=0
f=0
B=0
total_auc=0
List=[]
slist=[5,10,20,50,100,250,350,500,750,1000,1500]

path2 = './Bratssliced/test_labels/' + str(
                number) + '-label.nii.gz'
gt=nib.load(path2)
Label = th.from_numpy(np.asarray(gt.dataobj).astype(dtype='float32'))
iz = th.zeros(1, 256, 256)
iz[:, 8:-8, 8:-8] = Label
Labelmask = th.where(iz > 0, 1, 0)
viz.image(visualize(Labelmask[0, 8:-8, 8:-8]), opts=dict(caption="gt"+str(number)))

path3= './Bratssliced/testing/' + str(number)
for dirName2, subdirList, fileList2 in os.walk(path3, topdown=True):
  for f in fileList2:
    print('f', f)
    seqtype = f.split('_')[3]
    org=nib.load(os.path.join(dirName2, f))
    org2=th.from_numpy(np.asarray(org.dataobj).astype(dtype='float32'))
    print('org.2', org2.shape, seqtype)
    viz.image(visualize(org2), opts=dict(caption=str(number)+str(seqtype)))


for s in slist:
  print('s',s)
  folder='./scaling'+str(s)+'_ddim1000_t500'
  filename=number+'_output'
  path=os.path.join(folder, filename)
  try:
    sample=th.load(path)
  except:
    continue
 # viz.image(visualize(sample[0, 0,8:-8, 8:-8]), opts=dict(caption=str(number)+"samplepred0_"+str(s)))
 # viz.image(visualize(sample[0, 1, 8:-8, 8:-8]), opts=dict(caption="samplepred1_"+str(s)))
 # viz.image(visualize(sample[0, 2, 8:-8, 8:-8]), opts=dict(caption="samplepred2_"+str(s)))
  viz.image(visualize(sample[0, 3,  8:-8, 8:-8]), opts=dict(caption="samplepred3_"+str(s)))
  #viz.heatmap(visualize(np.flipud(sample[0, 4, 8:-8, 8:-8].cpu())), opts=dict(caption="diffpred"+str(number), colormap='Jet'))
  plt.imshow(sample[0, 4,  8:-8, 8:-8].cpu(), cmap='jet')
  plt.xticks([])
  plt.yticks([])
  viz.matplot(plt)
  plt.savefig('./imageseries/'+str(number)+'/'+str(s)+'_diff.png')
  

